// The native-audio engine, the second implementation of the renderer's
// VoiceEngine seam (engineTypes.ts). One WebSocket to the agent's /realtime:
// the microphone goes up as raw PCM16LE and the model's own 24 kHz voice comes
// back, generation-tagged, so a barge-in can silence audio that was already
// scheduled without waiting on the server. No VAD, no STT, no TTS here — the
// model hears and speaks natively; this class is the transport plus the one
// local decision that must stay local (barge-in).
//
// The cascade (voiceEngine.ts) is the other engine; useLiveSession/useFlowOwner
// never learn which one they hold.
import type { RealtimeClientMsg, RealtimePhase, RealtimeServerMsg } from "@openlive/shared";
import { AudioPlayer } from "./audioPlayback";
import type { EnginePhase, HoldEnd, VoiceEngine as VoiceEngineContract, VoiceEngineHandlers } from "./engineTypes";
import { MIC_RATE, MicSpectrum, NoiseFloor, Pcm16Framer, rmsOf, watchMicEnded } from "./micCapture";
import { FrameRing, pcm16ToFloat32 } from "./pcm";
import { liveWsBase } from "./liveClient";
import { loadPipelineConfig } from "./pipelineConfig";
import { reconnectDelay } from "./linkStatus";
import { log } from "../log";

/** The model's output is 24 kHz PCM16LE mono; the mic's is 16 kHz (micCapture). */
export const OUTPUT_RATE = 24000;
/** Bytes of the generation header the server puts on every binary frame. */
export const HEADER_BYTES = 4;
/** The user's voice kept while the model sleeps: waking with it means what they
 *  said during the sleep is heard from its true start, not from mid-word. The
 *  agent keeps the same 1.5 s pre-roll (services/agent/src/live/gemini-bidi.ts). */
export const PRE_ROLL_MS = 1500;
/** The ScriptProcessor's buffer: 64 ms of 16 kHz audio, and one ring slot. */
export const MIC_FRAME = 1024;
/** Ring slots that hold PRE_ROLL_MS: 24 x 64 ms = 1.536 s. */
export const PRE_ROLL_FRAMES = Math.ceil((PRE_ROLL_MS / 1000) * (MIC_RATE / MIC_FRAME));

/** The agent's own first syllable, echoed back through the speakers, cannot barge in. */
const ONSET_GRACE_MS = 250;
/** A socket that stays up this long counts as recovered and resets the backoff. */
const HEALTHY_MS = 3000;
/** How many failed opens before the outage is worth a console line. */
const QUIET_TRIES = 4;

/** One server frame: `[4-byte uint32LE generation][raw PCM16LE mono 24000 Hz]`. */
export interface RealtimeFrame {
  /** The generation (epoch) this audio belongs to. */
  gen: number;
  /** The audio, still PCM16LE: pcm16ToFloat32() it, never pcmDecoder(). */
  body: Uint8Array;
}

/** Split a server frame into its generation and its audio. A frame with no audio
 *  (a header alone, or a header plus half a sample) is not playable: null. Pure. */
export function decodeRealtimeFrame(bytes: Uint8Array): RealtimeFrame | null {
  if (bytes.length <= HEADER_BYTES) return null;
  const gen = (bytes[0]! | (bytes[1]! << 8) | (bytes[2]! << 16) | (bytes[3]! << 24)) >>> 0;
  return { gen, body: bytes.subarray(HEADER_BYTES) };
}

/** The binary frame for one chunk of the model's 24 kHz audio. The inverse of
 *  decodeRealtimeFrame, for a call that has to be resumed from disk or a test. Pure. */
export function encodeRealtimeFrame(gen: number, pcm16le: Uint8Array): Uint8Array {
  const out = new Uint8Array(HEADER_BYTES + pcm16le.length);
  out[0] = gen & 0xff;
  out[1] = (gen >>> 8) & 0xff;
  out[2] = (gen >>> 16) & 0xff;
  out[3] = (gen >>> 24) & 0xff;
  out.set(pcm16le, HEADER_BYTES);
  return out;
}

/** Where the native engine connects:
 *  ws://127.0.0.1:<port>/realtime?chat=<chatId>&lang=<lang>&device=<0|1>&token=<secret>
 *  English is left out of the query, as on /live (an English call is what it was
 *  before languages existed); `device` is the desktop app's machine audio. Pure. */
export function realtimeUrl(t: { chat: string; base?: string; lang?: string; device?: boolean; token?: string }): string {
  const q = new URLSearchParams({ chat: t.chat });
  if (t.lang && t.lang !== "en") q.set("lang", t.lang);
  if (t.device) q.set("device", "1");
  if (t.token) q.set("token", t.token);
  return `${(t.base ?? liveWsBase()).replace(/\/+$/, "")}/realtime?${q}`;
}

/** What a call is connected to, and where its non-audio messages go. */
export interface RealtimeOptions {
  /** The conversation this call belongs to. */
  chat: string;
  /** ws://host:port — the agent's own port. Default: liveWsBase(), as /live uses. */
  base?: string;
  /** The session language. Default: the saved pipeline's, which is what the
   *  prompt cache was warmed with. */
  lang?: string;
  /** A call in the desktop app can reach this machine itself (its own audio in/out). */
  device?: boolean;
  /** The per-launch agent secret, handed over by the desktop app. Never logged. */
  token?: string;
  /** Everything else /realtime says — draft, task, permission, elicitation — for
   *  the surface's store. The engine owns audio and phases only. */
  onMessage?: (m: RealtimeServerMsg) => void;
  /** How the socket is opened. Injectable so a test can drive every frame by hand. */
  openSocket?: (url: string) => WebSocket;
}

/** The desktop app's per-launch secret, when there is one (the agent's secret gate). */
const agentToken = (): string | undefined =>
  (globalThis as { window?: { openlive?: { agentToken?: string } } }).window?.openlive?.agentToken;

/** A call in the desktop app can reach the machine itself, as Flow's does. */
const deviceAudio = (): boolean =>
  typeof (globalThis as { window?: { openlive?: { flow?: { device?: unknown } } } }).window?.openlive?.flow?.device === "function";

/** The phase a surface shows, from the wire phase and whether the model is talking. */
export function enginePhaseOf(phase: RealtimePhase, playing: boolean): EnginePhase {
  if (playing) return "speaking";
  switch (phase) {
    case "live": return "listening";
    case "connecting": case "reconnecting": return "thinking";
    default: return "idle"; // idle, sleeping, error
  }
}

/** The second live engine: generation-tagged native audio over /realtime. */
export class RealtimeEngine implements VoiceEngineContract {
  private player: AudioPlayer;
  private opts: RealtimeOptions;
  private ws: WebSocket | null = null;
  private phase: RealtimePhase = "idle";
  private reported: EnginePhase = "idle";
  private stopped = false;

  /** The generation this renderer accepts audio from. Bumped by a local barge-in,
   *  adopted when the server's is higher; a frame below it is a straggler from a
   *  generation the user already interrupted (docs/01-CONTRACTS.md rule 1). */
  private epoch = 0;
  /** When the reply now sounding began (0: nothing sounding). ONSET_GRACE_MS is
   *  measured from here, so the agent's own voice through the speakers can't self-barge. */
  private speakAt = 0;
  /** When the agent's voice last stopped: what the confirmation gate compares a
   *  spoken confirmation's onset against (echo/AEC leak, docs/00-BUILD-ORDER Stage 2). */
  private playbackEndMs = 0;

  private micRms = 0;               // smoothed, for the orb and the echo-safe gate
  private floor = new NoiseFloor(); // the room's ambient level, learned while the agent is quiet
  private speechOn = false;         // the local VAD is inside a segment
  private lastSpeechAt = 0;
  private lastEchoSafe = true;      // the echo check of the segment's last frame
  private hushed = false;           // an ask's audio is held back while its answer is spoken
  private speechEndMs = 550;        // the user's trailing silence, from the saved pipeline
  private muted = false;
  private gated = false;            // push to talk: the mic is shut between holds
  private ptt = false;
  private open: (() => Promise<MediaStream>) | null = null;
  private micCtx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micTrack: MediaStreamTrack | null = null;
  private micSrc: MediaStreamAudioSourceNode | null = null;
  private micNode: ScriptProcessorNode | null = null;
  private unwatchMic: (() => void) | null = null;
  private micOps: Promise<unknown> = Promise.resolve();
  private spectrum = new MicSpectrum();
  private ring = new FrameRing(PRE_ROLL_FRAMES); // the pre-roll, while sleeping
  private framer = new Pcm16Framer();            // 80 ms whole frames, no header
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private healthyTimer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;

  // Accept a pre-primed player so the audio context is unlocked during the Start
  // click (iOS blocks audio started after an await — see useLiveSession.start).
  constructor(private h: VoiceEngineHandlers, player?: AudioPlayer, opts: RealtimeOptions = { chat: "" }) {
    this.player = player ?? new AudioPlayer();
    this.opts = opts;
  }

  // ── lifecycle ───────────────────────────────────────────────────────────

  /** Opens the socket and the microphone: one per call. Rejects when the mic is
   *  wanted now and cannot be opened (the surface then re-acquires a device). */
  async start(open: () => Promise<MediaStream>): Promise<void> {
    if (this.stopped) return;
    this.open = open;
    this.player.resume();
    this.speechEndMs = loadPipelineConfig().vad.redemptionMs;
    this.connect();
    // A call opened muted or gated (push to talk) keeps the device shut, as the
    // cascade does; otherwise the mic is wanted now and a refusal must reject.
    if (this.wantsMic()) await this.listen(true);
  }

  /** Swap the mic mid-call (device change, or a lost one) without touching playback. */
  async reopen(): Promise<void> {
    const op = this.micOps.then(async () => {
      this.closeMic();
      if (this.wantsMic()) await this.openMic();
    });
    this.micOps = op.catch(() => { /* reported by the caller */ });
    await op;
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; }
    if (this.healthyTimer) { clearTimeout(this.healthyTimer); this.healthyTimer = null; }
    // Say goodbye while the socket is still here: the session stops the model
    // (and its own Gemini socket) instead of leaving it to a timeout.
    this.send({ t: "control", action: "end" });
    const ws = this.ws;
    this.ws = null;
    try { ws?.close(); } catch { /* already closed */ }
    this.closeMic();
    this.spectrum.close();
    this.player.close();
    this.ring.clear();
    this.framer.reset();
    this.speechOn = false;
    this.voiceIdle();
    this.phase = "idle";
    this.setPhase("idle");
  }

  currentPhase(): EnginePhase { return this.reported; }

  // ── the socket ─────────────────────────────────────────────────────────

  private connect() {
    const url = realtimeUrl({
      chat: this.opts.chat,
      base: this.opts.base,
      lang: this.opts.lang ?? loadPipelineConfig().language,
      device: this.opts.device ?? deviceAudio(),
      token: this.opts.token ?? agentToken(),
    });
    const ws = this.openSocket(url);
    // No socket at all (no WebSocket global, a refused address): back off like a drop.
    if (!ws) { this.attempts++; this.retryLater(); return; }
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.attempts = 0;
      if (this.healthyTimer) clearTimeout(this.healthyTimer);
      this.healthyTimer = setTimeout(() => { this.attempts = 0; }, HEALTHY_MS);
      // A generation belongs to the session behind one socket. A new socket is a
      // new session: its counter may start over, and an epoch kept from the old
      // one would drop every frame of the new call (a silently dead voice).
      this.resetGeneration();
      this.phase = "live";
      // Words said while the socket was still opening are the call's first words.
      this.drainPreRoll();
      this.refreshPhase();
    };
    ws.onclose = (ev) => this.onClose(ws, ev?.code, ev?.reason);
    ws.onerror = () => { /* onclose follows; the reconnect below handles it */ };
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return;
      if (typeof ev.data === "string") { this.onMessage(ev.data); return; }
      if (ev.data instanceof Uint8Array) this.onBinary(ev.data);
      else if (ev.data instanceof ArrayBuffer) this.onBinary(new Uint8Array(ev.data));
    };
    this.ws = ws;
  }

  /** Opens the socket to `url`, or null when it cannot be opened at all (no
   *  WebSocket, a bad address): the retry below then owns it. */
  private openSocket(url: string): WebSocket | null {
    try {
      return this.opts.openSocket ? this.opts.openSocket(url) : new WebSocket(url);
    } catch (e) {
      log.warn("live", "realtime connect:", e);
      return null;
    }
  }

  private onClose(ws: WebSocket, code?: number, reason?: string) {
    // A socket the engine already let go of (or replaced): its close is stale.
    if (this.ws !== ws) return;
    this.ws = null;
    // Held mic audio belongs to the socket that was hearing it, not to the next one.
    this.ring.clear();
    this.framer.reset();
    // Nothing of the reply may outlive the connection that was streaming it: the
    // next socket is a new session with its own generation counter.
    this.player.flush(this.epoch);
    this.voiceIdle();
    if (this.stopped) return;
    if (this.healthyTimer) { clearTimeout(this.healthyTimer); this.healthyTimer = null; }
    this.attempts++;
    if (this.attempts === QUIET_TRIES) {
      log.warn("live", `realtime disconnected${code ? ` (${code})` : ""}${reason ? `: ${reason}` : ""} — still trying`);
    }
    this.phase = "reconnecting";
    this.refreshPhase();
    this.retryLater();
  }

  private retryLater() {
    if (this.stopped) return;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => { this.retryTimer = null; this.connect(); }, reconnectDelay(Math.max(0, this.attempts - 1)));
  }

  /** A message from /realtime. Audio is the binary path; everything here is JSON. */
  private onMessage(raw: string) {
    let m: RealtimeServerMsg;
    try { m = JSON.parse(raw) as RealtimeServerMsg; } catch { return; }
    switch (m.t) {
      case "phase": this.onPhaseMsg(m.phase, m.epoch, m.reason); break;
      case "barge_in_ack": if (m.epoch > this.epoch) this.adopt(m.epoch); break;
      case "transcript": this.onTranscript(m.role, m.text, m.final); break;
      case "error":
        log.warn("live", `realtime error ${m.code}: ${m.message}`);
        this.phase = "error";
        this.refreshPhase();
        break;
      default: break; // draft/task/bound/permission/elicitation/sse: the surface's
    }
    this.opts.onMessage?.(m);
  }

  private onPhaseMsg(phase: RealtimePhase, epoch: number, reason?: string) {
    if (epoch > this.epoch) this.adopt(epoch);
    const before = this.phase;
    this.phase = phase;
    // Waking from a sleep: everything the mic heard meanwhile goes up first, from
    // its true start, so the model does not resume mid-word.
    if (phase === "live" && before !== "live") this.drainPreRoll();
    if (phase === "error") log.warn("live", `realtime: ${reason ?? "error"}`);
    this.refreshPhase();
  }

  private onTranscript(role: "user" | "assistant", text: string, final: boolean) {
    const t = text.trim();
    if (!t) return;
    if (role === "user") {
      // The interim caption greys in as they talk; the final one is the turn.
      if (final) { this.h.onPartial(""); this.h.onUserText(t, []); }
      else this.h.onPartial(t);
      return;
    }
    // A native-audio reply's only text is its output transcription, and it
    // arrives a sentence at a time: show each finished sentence as it was said.
    if (final) this.h.onAgentText(t, []);
  }

  private send(m: RealtimeClientMsg) {
    if (this.ws?.readyState !== 1) return; // 1 = OPEN
    try { this.ws.send(JSON.stringify(m)); } catch { /* closing: the reconnect owns it */ }
  }

  /** Frames may go up: the socket is open and the model is not asleep. */
  private uplink(): boolean {
    return this.ws?.readyState === 1 && this.phase !== "sleeping" && this.phase !== "error";
  }

  // ── audio down: generation-tagged playback ─────────────────────────────

  /** One binary frame from the server: `[u32le gen][PCM16LE 24 kHz]`. The one
   *  place the barge-in contract is enforced on the way in. Public: the test
   *  drives it with no socket, and a resumed call can replay a kept frame. */
  onBinary(bytes: Uint8Array) {
    if (this.stopped) return;
    const frame = decodeRealtimeFrame(bytes);
    if (!frame) return;
    // A frame already in flight when the user interrupted: below the epoch, it is
    // a straggler of speech that must not resume (docs/01-CONTRACTS.md rule 1).
    if (frame.gen < this.epoch) return;
    if (frame.gen > this.epoch) this.adopt(frame.gen);
    // An ask silenced while its answer is being spoken (holdBargeIn): the rest of
    // it is not played over the answer, but the generation is kept in step, so
    // what the model says once the answer is in plays normally.
    if (this.hushed) return;
    // The agent's own 24 kHz PCM16 — NOT pcmDecoder(), which reads Float32LE and
    // would play this as static (docs/03-EVIDENCE.md).
    const pcm = pcm16ToFloat32(frame.body);
    if (!pcm.length) return;
    if (!this.speakAt) this.speakAt = Date.now(); // the reply starts here; the onset grace runs from it
    this.player.play(pcm, frame.gen, OUTPUT_RATE, () => this.onPlayStart(frame.gen));
    this.refreshPhase();
  }

  /** A generation above ours wins: everything scheduled belongs to the
   *  interrupted one, so it goes now — not after the round trip that announced it. */
  private adopt(gen: number) {
    this.epoch = gen;
    this.speakAt = 0;
    this.hushed = false; // a new generation is a new thing said: nothing left to hold back
    this.player.flush(gen);
    this.refreshPhase();
  }

  /** A new socket is a new session: its generation counter starts where its own
   *  session does. The flush is best-effort — nothing of the old call may play
   *  over the new one, and the floor is a new room's after a reconnect. */
  private resetGeneration() {
    this.epoch = 0;
    this.speakAt = 0;
    this.hushed = false;
    this.player.flush(0);
    this.speechOn = false;
  }

  private onPlayStart(gen: number) {
    // A chunk of an interrupted reply started late: it is already flushed.
    if (gen !== this.epoch || this.stopped) return;
    this.refreshPhase();
  }

  /** The model stopped sounding: the gate's playback clock stops here. */
  private voiceIdle() {
    if (this.speakAt) this.playbackEndMs = Date.now();
    this.speakAt = 0;
  }

  // ── audio up: the mic, the pre-roll and barge-in ──────────────────────

  /** One microphone frame (mono, MIC_RATE). The ScriptProcessor's callback and
   *  the one place a barge-in is decided — locally, inside this very frame.
   *  Public so the test can drive the decision without a device. */
  micFrame(frame: Float32Array) {
    if (this.stopped || this.muted) return;
    const rms = rmsOf(frame);
    this.micRms += (rms - this.micRms) * 0.3;
    const playing = this.player.playing();
    // The room's floor is learned only while the agent is quiet: a reply's own
    // echo must never raise the gate that is there to reject it.
    if (!playing) this.floor.learn(rms);
    const gate = this.floor.gate();
    // Playback-aware: on SPEAKERS the browser's AEC leaks some of the agent's own
    // voice back into the mic, so while it plays the mic must clear the gate
    // scaled up with the playback level. Real speech over the top clears it, the
    // agent's residual echo does not (the cascade engine's same rule).
    const echoSafe = !playing || this.micRms > gate * (1 + 2 * this.player.level());
    const heard = rms > gate && echoSafe;
    this.lastEchoSafe = echoSafe;
    if (heard) {
      const began = this.noteSpeech();
      // Sleeping (the server let the model's socket go quiet): only the renderer
      // knows the user is talking again, so it asks for the wake. What they said
      // meanwhile is in the pre-roll and goes up the moment the model is live.
      if (began && this.phase === "sleeping") this.send({ t: "control", action: "wake" });
      if (playing && Date.now() - this.speakAt > ONSET_GRACE_MS) {
        // Barge-in, unless an ask is pending: holdBargeIn means the next utterance
        // IS the answer to the agent's question, and cancelling would kill the very
        // ask the user is answering. It is hushed instead (local audio only).
        if (this.h.holdBargeIn?.()) this.hush();
        else this.barge();
      }
      // After the decision, so a segment that barged in carries the clock the
      // confirmation gate times it against (docs/00-BUILD-ORDER Stage 2).
      if (began) {
        this.h.onSpeechStart?.();
        this.send({ t: "vad_state", speaking: true, onsetMs: Date.now(), ...this.playbackClock(), echoSafe });
      }
    } else this.speechQuiet();
    if (!this.uplink()) { this.ring.push(frame); this.refreshPhase(); return; }
    this.pump([frame]);
    this.refreshPhase();
  }

  /** The playback clock, when the agent's voice has sounded in this call. */
  private playbackClock(): { playbackEndMs?: number } {
    return this.playbackEndMs ? { playbackEndMs: this.playbackEndMs } : {};
  }

  /** The user's speech went on, and whether it started a segment here. */
  private noteSpeech(): boolean {
    this.lastSpeechAt = Date.now();
    if (this.speechOn) return false;
    this.speechOn = true;
    return true;
  }

  private speechQuiet() {
    if (!this.speechOn || Date.now() - this.lastSpeechAt < this.speechEndMs) return;
    this.speechOn = false;
    // The answer is over: an ask's held-back tail may be heard again.
    this.hushed = false;
    this.send({ t: "vad_state", speaking: false, onsetMs: Date.now(), ...this.playbackClock(), echoSafe: this.lastEchoSafe });
  }

  /** Local barge-in: the generation is bumped and the player flushed in this very
   *  frame, so nothing of the interrupted reply survives the decision; the server
   *  then bumps its own generation and the new audio is adopted on arrival. */
  private barge() {
    this.cut();
    // The surface stops its caption reveal and truncates the saved reply here.
    // No `spoken`: a native reply's exact cutoff is timed on the server, which
    // persists the same cut (the cascade returns its own sentence-granular text).
    this.h.onBargeIn();
  }

  /** The cut itself, with no word to the surface: silencing the reply and
   *  cancelling it on the wire. */
  private cut() {
    this.epoch++;
    this.speakAt = 0;
    this.hushed = false;
    this.playbackEndMs = Date.now();
    this.player.flush(this.epoch);
    this.send({ t: "barge_in", epoch: this.epoch, vadOnsetMs: Date.now() });
    this.refreshPhase();
  }

  /** Silence what is buffered without cancelling the model: what an ask pending
   *  over the user's answer wants. The question's tail is not heard while they
   *  reply, and the model keeps its turn and its generation.
   *
   *  `hushed` (not an epoch bump) is what holds the rest of it back: the audio of
   *  the question keeps arriving at the SAME generation, and bumping our epoch
   *  would drop it — and, because the server never agreed to a new generation,
   *  everything it said afterwards too. So the tail is skipped until the answer's
   *  segment ends, and the generation stays adoptable. */
  private hush() {
    this.hushed = true;
    this.speakAt = 0;
    this.playbackEndMs = Date.now();
    this.player.flush(this.epoch);
    this.refreshPhase();
  }

  /** Whole 80 ms frames up, carrying the samples that do not fill one. */
  private pump(frames: Float32Array[]) {
    for (const f of frames) for (const bytes of this.framer.push(f)) {
      // The view's buffer is a plain ArrayBuffer here (pcm16FromFloat32 allocates one).
      try { this.ws?.send(bytes as Uint8Array<ArrayBuffer>); } catch { /* closing: the reconnect owns it */ }
    }
  }

  /** The pre-roll goes up, oldest first: the words said while the model slept. */
  private drainPreRoll() {
    const held = this.ring.drain();
    if (held.length) this.pump([held]);
  }

  // ── the microphone ────────────────────────────────────────────────────

  private wantsMic() { return !this.muted && (!this.gated || this.ptt); }

  private listen(on: boolean): Promise<void> {
    const op = this.micOps.then(() => (on ? this.openMic() : this.closeMic()));
    this.micOps = op.catch(() => { /* reported by the caller */ });
    return op;
  }

  private async openMic() {
    if (this.micCtx || !this.open) return;
    const stream = await this.open();
    if (this.stopped) { stream.getTracks().forEach((t) => t.stop()); return; }
    this.micStream = stream;
    this.micTrack = stream.getAudioTracks()[0] ?? null;
    // "ended" fires only when the browser ends a track — unplugged, or revoked.
    this.unwatchMic = watchMicEnded(this.micTrack, () => { this.h.onMicLost?.(); });
    const ctx = new AudioContext({ sampleRate: MIC_RATE });
    const src = ctx.createMediaStreamSource(stream);
    const node = ctx.createScriptProcessor(MIC_FRAME, 1, 1);
    node.onaudioprocess = (e) => this.micFrame(e.inputBuffer.getChannelData(0).slice());
    src.connect(node);
    // A ScriptProcessor runs only on a path to the output. It writes nothing, so nothing plays.
    node.connect(ctx.destination);
    void ctx.resume();
    this.micCtx = ctx; this.micSrc = src; this.micNode = node;
    this.spectrum.attach(stream);
  }

  private closeMic() {
    this.unwatchMic?.();
    this.unwatchMic = null;
    if (this.micNode) this.micNode.onaudioprocess = null;
    try { this.micNode?.disconnect(); } catch { /* already gone */ }
    try { this.micSrc?.disconnect(); } catch { /* already gone */ }
    try { void this.micCtx?.close(); } catch { /* already gone */ }
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.spectrum.detach();
    this.micCtx = null; this.micSrc = null; this.micNode = null; this.micStream = null; this.micTrack = null;
    this.micRms = 0;
  }

  private micFailed = (e: unknown) => { log.warn("live", "realtime mic:", e); this.h.onMicLost?.(); };

  // ── the surface's commands ───────────────────────────────────────────

  /** A turn the user sends by hand: native audio has no typed turn channel — the
   *  model hears the mic. False means "nothing opened, send it yourself". */
  openTurn(): boolean { return false; }

  /** "Not for you" on the reply under way: cut like a barge-in. */
  dropReply() { if (!this.stopped && this.player.playing()) this.barge(); }

  /** No mid-thought hold in native mode: the model decides when the turn ended. */
  commitPending() { /* nothing is held */ }

  /** Side talk is the cascade's judgment; a native reply has no sentence to send. */
  sendAside(): boolean { return false; }

  beginPtt() {
    if (this.ptt || this.stopped) return;
    this.ptt = true;
    // A press while the model is talking means "listen to me now".
    if (this.player.playing() && !this.h.holdBargeIn?.()) this.barge();
    void this.listen(true).catch(this.micFailed);
  }

  /** Released: the words said while holding already went up as they were spoken. */
  async endPtt(): Promise<HoldEnd> {
    if (!this.ptt) return "heard";
    this.ptt = false;
    if (this.gated) await this.listen(false).catch(() => { /* the next hold reopens it */ });
    return "heard";
  }

  dropPtt() {
    if (!this.ptt) return;
    this.ptt = false;
    this.discard();
    if (this.gated) void this.listen(false).catch(() => { /* */ });
  }

  pttActive() { return this.ptt; }

  /** Push to talk: the mic is shut between holds, so nothing is heard without the key. */
  setGate(on: boolean) {
    this.gated = on;
    if (this.stopped) return;
    if (!this.wantsMic()) { void this.listen(false).catch(() => { /* */ }); return; }
    void this.listen(true).catch(this.micFailed);
  }

  /** The cascade renders hold cues in a TTS voice; this engine's voice is the model's own. */
  setHoldCues() { /* no local voice to cue */ }

  /** Mute: the mic is shut and the model told, so it does not wait on a user that is gone. */
  setMuted(muted: boolean) {
    this.muted = muted;
    this.send({ t: "mute", muted });
    if (this.stopped) return;
    if (muted) {
      this.speechOn = false;
      this.ring.clear();
      this.framer.reset();
      this.micRms = 0;
      void this.listen(false).catch(() => { /* */ });
      return;
    }
    if (this.wantsMic()) void this.listen(true).catch(this.micFailed);
  }

  // ── text the cascade synthesizes; the model here speaks for itself ────

  /** Assistant text arrives as transcripts (see onTranscript), never as deltas:
   *  the model's own voice is already saying it. */
  feedAgentDelta() { /* voiced natively */ }
  endAgentStep() { /* nothing held */ }
  endAgentTurn() { /* nothing held */ }
  /** The /realtime session voices its own permission asks; there is no local TTS
   *  here. The surface still shows the chip and routes the spoken answer. */
  ask() { /* voiced by the session */ }
  say() { /* no out-of-band voice in this engine */ }
  announce() { /* no out-of-band voice in this engine */ }

  /** Silence the reply and drop the rest of it. Undefined: the exact cutoff of a
   *  native reply is word-timed on the server, which persists the same cut.
   *  No onBargeIn: the surface asked for this cut and knows it happened (the
   *  cascade's cutReply does not notify either). */
  cutReply(): string | undefined {
    if (!this.stopped && this.player.playing()) this.cut();
    return undefined;
  }

  /** What was heard up to now goes nowhere: the pre-roll and the half frame too. */
  discard() {
    this.ring.clear();
    this.framer.reset();
    this.speechOn = false;
    this.hushed = false;
  }

  // ── levels and devices, for the orb ─────────────────────────────────

  micLevel() { return this.muted ? 0 : this.micRms; }
  agentLevel() { return this.player.level(); }
  setSpeaker(id: string) { this.player.setSink(id); }

  /** N octave-band magnitudes (0..1) of YOUR voice — a real spectrum for the orb. */
  micBands(n = 5): number[] { return this.muted ? new Array(n).fill(0) : this.spectrum.bands(n); }
  /** Same, for the agent's voice while it speaks. */
  agentBands(n = 5): number[] { return this.player.agentBands(n); }

  // ── helpers ─────────────────────────────────────────────────────────

  /** Reports the phase a surface should show, only when it changes. */
  private refreshPhase() {
    const playing = this.player.playing();
    if (!playing) this.voiceIdle();
    const p = enginePhaseOf(this.phase, playing);
    if (p !== this.reported) { this.reported = p; this.h.onPhase(p); }
  }

  private setPhase(p: EnginePhase) {
    if (p === this.reported) return;
    this.reported = p;
    this.h.onPhase(p);
  }
}
