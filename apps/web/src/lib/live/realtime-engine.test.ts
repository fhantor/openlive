// Guards the native engine's two contracts that no other test can: the wire
// decode ([u32le gen][PCM16LE], NOT pcmDecoder's Float32) and rule 1 of
// docs/01-CONTRACTS.md — a stale generation is dropped, a higher one is adopted
// and played, and a local barge-in flushes inside the frame it decided in.
// Plus the one list rule that keeps the cascade picker from offering the Live
// model, which speaks a different socket entirely.
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "vitest";
import { pcm16FromFloat32 } from "./micCapture.ts";
import { pcm16ToFloat32 } from "./pcm.ts";
import { HEADER_BYTES, MIC_FRAME, OUTPUT_RATE, PRE_ROLL_FRAMES, PRE_ROLL_MS, RealtimeEngine, decodeRealtimeFrame, encodeRealtimeFrame, enginePhaseOf, realtimeUrl } from "./realtimeEngine.ts";
import { GEMINI_LIVE_MODEL, cascadeModels, isLiveOnlyModel, realtimeModels } from "./pipelineConfig.ts";

/** Little-endian PCM16 bytes for signed samples. */
const pcm16 = (...samples: number[]) => {
  const out = new Uint8Array(samples.length * 2);
  samples.forEach((s, i) => { out[2 * i] = s & 0xff; out[2 * i + 1] = (s >> 8) & 0xff; });
  return out;
};

/** A player that records exactly what it was handed, and what was flushed. */
class FakePlayer {
  plays: { pcm: Float32Array; gen: number; rate: number }[] = [];
  flushes: number[] = [];
  queued = 0;   // chunks scheduled and not flushed
  loud = 0;     // playback level, for the echo-safe gate

  play(pcm: Float32Array, gen: number, rate = OUTPUT_RATE) { this.plays.push({ pcm, gen, rate }); this.queued++; }
  flush(epoch: number) { this.flushes.push(epoch); this.queued = 0; }
  playing() { return this.queued > 0; }
  level() { return this.loud; }
  setSink() { /* the fake has no device */ }
  resume() { /* no context */ }
  close() { /* no context */ }
}

/** A WebSocket the test drives by hand: sends are recorded, nothing is opened. */
class FakeWs {
  readyState = 1;
  binaryType = "";
  sent: (string | Uint8Array)[] = [];
  onopen: (() => void) | null = null;
  onclose: ((e: { code?: number; reason?: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: { data: unknown }) => void) | null = null;
  send(d: string | Uint8Array) { this.sent.push(d); }
  close() { this.readyState = 3; }
}

const handlers = (over: Record<string, unknown> = {}) => ({
  onPhase() {}, onPartial() {}, onUserText() {}, onAgentText() {}, onBargeIn() {}, onHold() {}, ...over,
} as never);

/** An engine mid-call with no microphone and no real socket. */
function liveEngine(over: Record<string, unknown> = {}) {
  const player = new FakePlayer();
  const phases: string[] = [];
  const started: string[] = [];
  const h = handlers({
    onPhase: (p: string) => phases.push(p),
    onSpeechStart: () => started.push("start"),
    onUserText: (t: string) => started.push(`user:${t}`),
    onPartial: (t: string) => started.push(`partial:${t}`),
    onAgentText: (t: string) => started.push(`agent:${t}`),
    ...over,
  });
  const eng = new RealtimeEngine(h, player as never, { chat: "c1" });
  const ws = new FakeWs();
  Object.assign(eng, { ws, phase: "live" });
  return { eng, player, ws, phases, started };
}

const sentJson = (ws: FakeWs) => ws.sent.filter((s): s is string => typeof s === "string").map((s) => JSON.parse(s));

// ── the wire frame ────────────────────────────────────────────────────────

test("decodeRealtimeFrame: a u32le generation then the PCM16LE audio", () => {
  const body = pcm16(-32768, 0, 32767);
  const frame = decodeRealtimeFrame(encodeRealtimeFrame(258, body))!;
  assert.equal(frame.gen, 258); // little-endian: 258 = 0x00000102 → [02 01 00 00]
  assert.deepEqual([...frame.body], [...body]);
  assert.equal(HEADER_BYTES, 4);
  // The biggest a uint32 body may tag, and the first generation of a call.
  assert.equal(decodeRealtimeFrame(encodeRealtimeFrame(0xffffffff, body))!.gen, 0xffffffff);
  assert.equal(decodeRealtimeFrame(encodeRealtimeFrame(0, body))!.gen, 0);
});

test("decodeRealtimeFrame: a frame with no audio is not playable", () => {
  assert.equal(decodeRealtimeFrame(new Uint8Array(0)), null);
  assert.equal(decodeRealtimeFrame(new Uint8Array([1, 2, 3])), null);      // a short header
  assert.equal(decodeRealtimeFrame(encodeRealtimeFrame(7, new Uint8Array(0))), null);
  // One stray byte past the header is half a sample: dropped by the decoder.
  assert.equal(pcm16ToFloat32(decodeRealtimeFrame(encodeRealtimeFrame(7, new Uint8Array([0x7f])))!.body).length, 0);
});

test("the frame goes to the player as PCM16 → Float32, never through pcmDecoder", () => {
  const { eng, player } = liveEngine();
  const body = pcm16(-32768, 0, 16384);
  eng.onBinary(encodeRealtimeFrame(0, body));
  assert.equal(player.plays.length, 1);
  assert.equal(player.plays[0]!.rate, OUTPUT_RATE);
  // The very samples, as pcm16ToFloat32 gives them: −1.0, 0, 0.5.
  assert.deepEqual([...player.plays[0]!.pcm], [...pcm16ToFloat32(body)]);
  assert.deepEqual([...player.plays[0]!.pcm], [-1, 0, 0.5]);
});

// ── rule 1: the generation counter is the barge-in contract ───────────────

test("a stale generation is dropped: audio already interrupted does not resume", () => {
  const { eng, player } = liveEngine();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100))); // this call's first audio: gen 0
  assert.equal(player.plays.length, 1);
  // The user barges in, then the flushed generation's frames were already in flight.
  Object.assign(eng, { epoch: 2 });
  eng.onBinary(encodeRealtimeFrame(1, pcm16(-100)));
  eng.onBinary(encodeRealtimeFrame(0, pcm16(-200)));
  assert.equal(player.plays.length, 1, "frames below the epoch must not play");
  assert.equal(player.flushes.length, 0, "a dropped straggler must not flush anything");
});

test("a higher generation is adopted, flushed and played; the same one merely plays", () => {
  const { eng, player } = liveEngine();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(1, 2)));
  assert.deepEqual(player.plays.map((p) => p.gen), [0]);
  // Generation 3 crosses the wire: whatever was scheduled is the interrupted reply.
  eng.onBinary(encodeRealtimeFrame(3, pcm16(3, 4)));
  assert.deepEqual(player.flushes, [3], "adopting must flush the old generation");
  assert.deepEqual(player.plays.map((p) => p.gen), [0, 3]);
  assert.equal(player.plays[1]!.rate, OUTPUT_RATE);
  // Audio of the adopted generation goes on playing without a second flush.
  eng.onBinary(encodeRealtimeFrame(3, pcm16(5, 6)));
  assert.deepEqual(player.flushes, [3]);
  assert.deepEqual(player.plays.map((p) => p.gen), [0, 3, 3]);
});

test("a barge_in_ack and a phase epoch above ours both adopt the server's generation", () => {
  const { eng, player } = liveEngine();
  Object.assign(eng, { ws: { readyState: 1, send() {} } });
  (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify({ t: "barge_in_ack", epoch: 4 }));
  assert.equal((eng as unknown as { epoch: number }).epoch, 4);
  // Older ack (a straggler of an earlier barge-in) is not a regression.
  (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify({ t: "barge_in_ack", epoch: 2 }));
  assert.equal((eng as unknown as { epoch: number }).epoch, 4);
  assert.deepEqual(player.flushes, [4]);
});

// ── local barge-in: one frame, and holdBargeIn is the veto ────────────────

/** A frame loud enough to clear the noise gate, at a given level. */
const loudFrame = (amp = 0.5) => new Float32Array(MIC_FRAME).fill(amp);
const quietFrame = (amp = 0.0001) => new Float32Array(MIC_FRAME).fill(amp);

test("barge-in flushes the player inside the frame it happened in, and tells the server", () => {
  const barged: string[] = [];
  const { eng, player, ws } = liveEngine({ onBargeIn: () => barged.push("barge") });
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100, 200))); // the model is talking
  assert.equal(player.playing(), true);
  // The onset grace is over: this is a real interruption, not the agent's own echo.
  Object.assign(eng, { speakAt: Date.now() - 1000 });
  eng.micFrame(loudFrame());
  // Not "on the next tick": the flush is synchronous with the decision.
  assert.deepEqual(player.flushes, [1], "the flush lands in the same frame as the decision");
  assert.equal(player.playing(), false);
  assert.equal((eng as unknown as { epoch: number }).epoch, 1);
  const barge = sentJson(ws).find((m) => m.t === "barge_in")!;
  assert.equal(barge.epoch, 1);
  assert.equal(typeof barge.vadOnsetMs, "number");
  // The surface is told in the same breath: its caption reveal stops here.
  assert.deepEqual(barged, ["barge"]);
});

test("barge-in honours holdBargeIn: an ask is pending, so no cancel — only a hush", () => {
  const barged: string[] = [];
  const { eng, player, ws } = liveEngine({ holdBargeIn: () => true, onBargeIn: () => barged.push("barge") });
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100, 200)));
  Object.assign(eng, { speakAt: Date.now() - 1000 });
  eng.micFrame(loudFrame());
  // The ask being answered must survive: no barge_in, no epoch bump, no surface cut.
  assert.deepEqual(sentJson(ws).filter((m) => m.t === "barge_in"), []);
  assert.equal((eng as unknown as { epoch: number }).epoch, 0);
  assert.deepEqual(barged, []);
  // The question's tail is silenced locally, and stays silenced over the answer.
  assert.deepEqual(player.flushes, [0]);
  eng.onBinary(encodeRealtimeFrame(0, pcm16(300)));
  assert.equal(player.plays.length, 1, "an ask's held-back tail is not played over the answer");
  // Once the answer's segment ends, the model's voice is heard again — at the
  // SAME generation, which is the server's: nothing it says would ever play if the
  // hush had bumped the epoch instead.
  Object.assign(eng, { speechEndMs: 0 });
  eng.micFrame(quietFrame());
  eng.onBinary(encodeRealtimeFrame(0, pcm16(400)));
  assert.deepEqual(player.plays.map((p) => p.gen), [0, 0]);
  assert.equal((eng as unknown as { epoch: number }).epoch, 0);
});

test("the agent's own first syllable cannot self-barge (onset grace)", () => {
  const { eng, player, ws } = liveEngine();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100, 200)));
  Object.assign(eng, { speakAt: Date.now() }); // the reply just started
  eng.micFrame(loudFrame());
  assert.deepEqual(player.flushes, []);
  assert.deepEqual(sentJson(ws).filter((m) => m.t === "barge_in"), []);
});

test("barge-in over the user's own turn is not a barge-in at all", () => {
  const { eng, player, ws } = liveEngine();
  Object.assign(eng, { speakAt: Date.now() - 1000 }); // a reply began earlier, but it is over
  eng.micFrame(loudFrame());
  assert.deepEqual(player.flushes, []);
  assert.deepEqual(sentJson(ws).filter((m) => m.t === "barge_in"), []);
});

test("an interrupted generation's late chunk never starts a new reply phase", () => {
  const { eng, player, phases } = liveEngine();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(1)));
  assert.deepEqual(phases, ["speaking"]);
  Object.assign(eng, { speakAt: Date.now() - 1000 });
  eng.micFrame(loudFrame()); // barges; the reply is now silent
  assert.equal(player.playing(), false);
  assert.equal(eng.currentPhase(), "listening", "with nothing playing, live is listening");
  // …and the flushed generation's next frame changes none of that.
  eng.onBinary(encodeRealtimeFrame(0, pcm16(2)));
  assert.equal(player.plays.length, 1);
  assert.equal(eng.currentPhase(), "listening");
});

// ── the socket: URL, mute, control ───────────────────────────────────────

test("realtimeUrl: /realtime with chat, language, device and the secret", () => {
  assert.equal(realtimeUrl({ chat: "c1", base: "ws://127.0.0.1:47833", lang: "bn" }), "ws://127.0.0.1:47833/realtime?chat=c1&lang=bn");
  // English is what a turn was before languages existed: left out, as on /live.
  assert.equal(realtimeUrl({ chat: "c1", base: "ws://h" }), "ws://h/realtime?chat=c1");
  assert.equal(realtimeUrl({ chat: "c 1", base: "ws://h", lang: "en", device: true, token: "a b" }),
    "ws://h/realtime?chat=c+1&device=1&token=a+b");
  // A trailing slash on the agent's address does not double up the path.
  assert.equal(realtimeUrl({ chat: "c1", base: "ws://h/" }), "ws://h/realtime?chat=c1");
});

test("start opens /realtime, and a mic that cannot open rejects", async () => {
  const urls: string[] = [];
  const eng = new RealtimeEngine(handlers(), new FakePlayer() as never, {
    chat: "c1", base: "ws://127.0.0.1:9", lang: "bn",
    openSocket: (u) => { urls.push(u); return new FakeWs() as never; },
  });
  await assert.rejects(eng.start(async () => { throw new Error("no mic"); }), /no mic/);
  assert.deepEqual(urls, ["ws://127.0.0.1:9/realtime?chat=c1&lang=bn"]);
});

test("mute is honoured: the mic shut, the model told, and nothing barges", async () => {
  const { eng, player, ws } = liveEngine();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100)));
  Object.assign(eng, { speakAt: Date.now() - 1000 });
  eng.setMuted(true);
  assert.deepEqual(sentJson(ws).find((m) => m.t === "mute"), { t: "mute", muted: true });
  assert.equal(eng.micLevel(), 0);
  // A muted engine hears nothing, so it cannot decide to barge either.
  eng.micFrame(loudFrame());
  assert.deepEqual(player.flushes, []);
  eng.setMuted(false);
  assert.deepEqual(sentJson(ws).filter((m) => m.t === "mute").map((m) => m.muted), [true, false]);
});

test("the speaker route and the levels reach the one player", () => {
  const { eng, player } = liveEngine();
  const sinks: string[] = [];
  Object.assign(player, { setSink: (id: string) => sinks.push(id) });
  eng.setSpeaker("speaker-2");
  assert.deepEqual(sinks, ["speaker-2"]);
  assert.equal(eng.agentLevel(), 0);
});

test("stop says goodbye on the socket it still has", () => {
  const { eng, ws } = liveEngine();
  eng.stop();
  assert.deepEqual(sentJson(ws).find((m) => m.t === "control"), { t: "control", action: "end" });
  assert.equal(ws.readyState, 3);
  assert.equal(eng.currentPhase(), "idle");
});

test("a new socket is a new session: its generation starts over, so its audio still plays", () => {
  const sockets: FakeWs[] = [];
  const player = new FakePlayer();
  const eng = new RealtimeEngine(handlers(), player as never, {
    chat: "c1", base: "ws://h",
    openSocket: () => { const s = new FakeWs(); sockets.push(s); return s as never; },
  });
  const open = (eng as unknown as { connect: () => void }).connect.bind(eng);
  // A fresh engine, its first socket open.
  open();
  const first = sockets[0]!;
  first.onopen!();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(1)));
  assert.deepEqual(player.plays.map((p) => p.gen), [0]);
  // The server's generation had advanced past the renderer's, then the wire dropped.
  eng.onBinary(encodeRealtimeFrame(7, pcm16(2)));
  assert.equal((eng as unknown as { epoch: number }).epoch, 7);
  first.readyState = 3;
  first.onclose?.({ code: 1006 });
  // A drop is reconnecting, not idle: the surface shows it as "thinking".
  assert.equal(eng.currentPhase(), "thinking");
  // Closing the OLD socket again (a late transport event) changes nothing.
  first.onclose?.({ code: 1006 });
  // The reconnect brings up a second socket, whose session counts from its own 0.
  clearTimeout((eng as unknown as { retryTimer: ReturnType<typeof setTimeout> }).retryTimer);
  open();
  const second = sockets.at(-1)!;
  second.onopen!();
  eng.onBinary(encodeRealtimeFrame(0, pcm16(3)));
  assert.deepEqual(player.plays.map((p) => p.gen), [0, 7, 0], "gen 0 of the new session must play");
  assert.equal((eng as unknown as { epoch: number }).epoch, 0);
  endTest(eng);
});

/** Let go of a test engine without a device or a real socket behind it. */
const endTest = (eng: RealtimeEngine) => {
  const timers = eng as unknown as { retryTimer?: ReturnType<typeof setTimeout>; healthyTimer?: ReturnType<typeof setTimeout> };
  clearTimeout(timers.retryTimer);
  clearTimeout(timers.healthyTimer);
  timers.retryTimer = undefined;
  timers.healthyTimer = undefined;
  eng.stop();
};

// ── the pre-roll: what the mic heard while the model slept ────────────────

test("the pre-roll holds 1.5 s and goes up from its true start on waking", () => {
  assert.ok((PRE_ROLL_FRAMES * MIC_FRAME) / 16000 >= PRE_ROLL_MS / 1000, `${PRE_ROLL_FRAMES} slots hold under 1.5 s`);
  const { eng, ws } = liveEngine();
  Object.assign(eng, { phase: "sleeping" });
  // Talking in its sleep: no audio goes up, all of it is kept. (The VAD's state
  // still does: the gate has to know they are talking to hold its echoes back.)
  for (let i = 0; i < PRE_ROLL_FRAMES; i++) eng.micFrame(new Float32Array(MIC_FRAME).fill(0.2));
  assert.deepEqual(ws.sent.filter((s): s is Uint8Array => typeof s !== "string"), []);
  // The renderer is the only side that hears them: it asks for the wake.
  assert.deepEqual(sentJson(ws).filter((m) => m.t === "control"), [{ t: "control", action: "wake" }]);
  // Woken, the held audio goes first: 24 x 64 ms = 1536 ms, whole 80 ms frames.
  (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify({ t: "phase", phase: "live", epoch: 0 }));
  const audio = () => ws.sent.filter((s): s is Uint8Array => typeof s !== "string");
  const frames = audio();
  assert.equal(frames.length, Math.floor((PRE_ROLL_FRAMES * MIC_FRAME) / (16000 * 0.08)));
  assert.ok(frames.every((f) => f.length === 16000 * 0.08 * 2), "PCM16LE mono 16 kHz: 80 ms is 2560 bytes");
  // The part that did not fill a frame is carried, not dropped: the next 64 ms of
  // mic audio completes it. Live, the frame goes up at once instead of the ring.
  eng.micFrame(new Float32Array(MIC_FRAME).fill(0.2));
  assert.equal(audio().length, frames.length + 1);
});

test("vad_state follows the mic level: speaking on the first loud frame, quiet after the hold", () => {
  const { eng, player, ws } = liveEngine();
  Object.assign(eng, { speechEndMs: 0 });
  const states = () => sentJson(ws).filter((m) => m.t === "vad_state");
  eng.micFrame(loudFrame());
  assert.equal(states()[0]!.speaking, true);
  assert.equal(states()[0]!.echoSafe, true);
  assert.equal(states()[0]!.playbackEndMs, undefined, "nothing has sounded yet");
  eng.micFrame(quietFrame());
  assert.equal(states().at(-1)!.speaking, false);
  // A held confirmation is timed against when the agent's voice stopped: a barge-in
  // stops it now, and the segment's end carries that clock to the gate.
  eng.onBinary(encodeRealtimeFrame(0, pcm16(100)));
  assert.equal(player.playing(), true);
  Object.assign(eng, { speakAt: Date.now() - 1000 });
  eng.micFrame(loudFrame());  // barges: the voice stops inside this frame
  eng.micFrame(quietFrame()); // the segment ends: its state carries the clock
  assert.equal(typeof states().at(-1)!.playbackEndMs, "number");
});

// ── phases, transcripts ─────────────────────────────────────────────────

test("the wire phase becomes the phase a surface shows", () => {
  assert.equal(enginePhaseOf("live", false), "listening");
  assert.equal(enginePhaseOf("connecting", false), "thinking");
  assert.equal(enginePhaseOf("reconnecting", false), "thinking");
  assert.equal(enginePhaseOf("sleeping", false), "idle");
  assert.equal(enginePhaseOf("idle", false), "idle");
  assert.equal(enginePhaseOf("error", false), "idle");
  // The model's voice always wins over the wire phase.
  assert.equal(enginePhaseOf("sleeping", true), "speaking");
  assert.equal(enginePhaseOf("live", true), "speaking");
});

test("transcripts: the user's caption greys in, the reply's arrives a sentence at a time", () => {
  const { eng, started } = liveEngine();
  const msg = (m: unknown) => (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify(m));
  msg({ t: "transcript", role: "user", text: "শোনো", final: false });
  msg({ t: "transcript", role: "user", text: "শোনো, হেডারটা বদলাও", final: true });
  msg({ t: "transcript", role: "assistant", text: "ঠিক আছে।", final: true });
  msg({ t: "transcript", role: "assistant", text: "   ", final: true }); // nothing said: dropped
  // The final clears the greyed caption first, so the turn does not show twice.
  assert.deepEqual(started, ["partial:শোনো", "partial:", "user:শোনো, হেডারটা বদলাও", "agent:ঠিক আছে।"]);
});

test("a phase error is reported, and every message still reaches the surface", () => {
  const seen: { t: string }[] = [];
  const eng = new RealtimeEngine(handlers(), new FakePlayer() as never, { chat: "c1", onMessage: (m) => seen.push(m) });
  Object.assign(eng, { ws: new FakeWs(), phase: "live" });
  (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify({ t: "error", code: "x", message: "y" }));
  (eng as unknown as { onMessage: (s: string) => void }).onMessage(JSON.stringify({ t: "draft_state", draft: null }));
  (eng as unknown as { onMessage: (s: string) => void }).onMessage("not json");
  assert.deepEqual(seen.map((m) => m.t), ["error", "draft_state"]);
});

// ── the Live model stays out of the cascade picker ──────────────────────

/** The two model lists a provider returns, as /api/models hands them over. */
const MODELS = [
  { id: "gemini-3.8-live", display_name: "Gemini 3.8 Live" },
  { id: "gemini-2.5-flash", display_name: "Gemini 2.5 Flash" },
  { id: "gemini-3.1-flash-lite", display_name: "Gemini 3.1 Flash Lite" },
  { id: "gpt-5-mini", display_name: "GPT-5 mini" },
];

test("isLiveOnlyModel: the native-audio ids, and only them", () => {
  assert.equal(isLiveOnlyModel("gemini-3.8-live"), true);
  assert.equal(isLiveOnlyModel("  models/gemini-2.0-flash-live-001  "), true);
  assert.equal(isLiveOnlyModel(GEMINI_LIVE_MODEL), true);
  assert.equal(isLiveOnlyModel("gemini-2.5-flash"), false);
  assert.equal(isLiveOnlyModel("gemini-3.1-flash-lite"), false);
  assert.equal(isLiveOnlyModel("gpt-5"), false);
  assert.equal(isLiveOnlyModel("live"), false);      // a name, not this suffix
  assert.equal(isLiveOnlyModel("gemini-live-chat"), false);
});

test("cascadeModels: the cascade picker no longer lists gemini-3.8-live", () => {
  // The exact mapping both pickers do (`{ value: m.id, label: m.display_name }`).
  const options = cascadeModels(MODELS).map((m) => ({ value: m.id, label: m.display_name }));
  assert.deepEqual(options.map((o) => o.value), ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gpt-5-mini"]);
  assert.ok(!options.some((o) => o.value === "gemini-3.8-live"));
  // Nothing else is hidden: a chat model survives, and the input is not mutated.
  const ids = MODELS.map((m) => m.id);
  assert.deepEqual(cascadeModels(MODELS).map((m) => m.id), ids.filter((id) => !isLiveOnlyModel(id)));
  assert.deepEqual(MODELS.map((m) => m.id), ids);
  // A picker row keyed by something other than `id` filters the same way.
  assert.deepEqual(cascadeModels(["gemini-3.8-live", "gpt-5"], (s) => s), ["gpt-5"]);
});

test("realtimeModels: the native engine's own picker still SHOWS gemini-3.8-live, first", () => {
  const ids = realtimeModels(MODELS).map((m) => m.id);
  assert.equal(ids[0], "gemini-3.8-live");
  assert.deepEqual(ids.slice(1), ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gpt-5-mini"]);
  assert.ok(realtimeModels(MODELS).length === MODELS.length, "the live picker hides nothing");
  // A list with no Live model is left as it is.
  assert.deepEqual(realtimeModels(["gpt-5", "gemini-2.5-flash"], (s) => s), ["gpt-5", "gemini-2.5-flash"]);
});

test("every cascade picker calls the filter on the list it maps", () => {
  const SRC = join(import.meta.dirname, "..", "..");
  const read = (p: string) => readFileSync(join(SRC, p), "utf8");
  const settings = read("components/settings/ModelsSettings.tsx");
  const quick = read("components/live/ModelQuickPick.tsx");
  const bar = read("components/live/AgentBar.tsx");
  // The filter is applied to the FETCHED list in the options the picker renders —
  // a filter nothing calls would satisfy the pass condition nowhere (§C Stage 7).
  assert.match(settings, /const options: SearchOption\[\] = cascadeModels\(models\)\.map\(/);
  assert.match(quick, /\.\.\.cascadeModels\(models\)\.map\(\(m\) => \(\{ id: m\.id, name: m\.display_name \}\)\)/);
  assert.match(bar, /\{ id: "", label: "Recommended" \}, \.\.\.cascadeModels\(models\)\.map\(/);
  for (const src of [settings, quick, bar]) assert.match(src, /import \{ cascadeModels \} from "@\/lib\/live\/pipelineConfig"/);
  // One rule for both sides of the socket: no picker hardcodes a model id.
  // (Comments explain which one is hidden; no id is written in code.)
  const code = (settings + quick + bar).replace(/\/\/[^\n]*/g, "");
  assert.equal(/gemini-[\w.-]*-live/.test(code), false);
});

// ── the engine is the seam's second implementation ──────────────────────

test("the engine exposes every command the VoiceEngine seam has the surface use", () => {
  const eng = new RealtimeEngine(handlers(), new FakePlayer() as never, { chat: "c1" });
  Object.assign(eng, { ws: new FakeWs(), phase: "live" });
  for (const fn of ["start", "reopen", "stop", "currentPhase", "openTurn", "dropReply", "commitPending", "sendAside",
    "beginPtt", "endPtt", "dropPtt", "pttActive", "setGate", "setHoldCues", "setMuted", "feedAgentDelta", "endAgentStep",
    "endAgentTurn", "ask", "say", "announce", "cutReply", "discard", "micLevel", "agentLevel", "setSpeaker", "micBands", "agentBands"]) {
    assert.equal(typeof (eng as unknown as Record<string, unknown>)[fn], "function", fn);
  }
  // The model always decides the turn: nothing to hold, nothing to send by hand.
  assert.equal(eng.openTurn(), false);
  assert.equal(eng.sendAside(), false);
  assert.equal(eng.cutReply(), undefined);
  // pcm16FromFloat32 is the encoder this engine's frames come from; a decode→encode
  // round trip is exact, so a replayed frame is the same audio.
  const samples = new Float32Array([-1, 0, 0.5]);
  assert.deepEqual([...pcm16ToFloat32(pcm16FromFloat32(samples))], [...samples]);
});
