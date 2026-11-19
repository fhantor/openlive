// The microphone half of a live engine, kept out of the turn loop: opening the
// device, framing what it hears, watching a track the browser ends on its own,
// tapping it for the orb's spectrum, and learning the room's ambient floor.
// Pure plumbing (except getUserMedia/the analyser), so it is shared by the local
// cascade engine and the native Gemini one rather than written twice.
import { octaveBands } from "./spectrum";

/** The mic's samples are mono at this rate everywhere in the live engines. */
export const MIC_RATE = 16000;

/** Browser AEC on: the agent's own voice is cancelled from the mic, which is what
 *  makes barge-in on speakers work at all. */
export const MIC_CONSTRAINTS: MediaTrackConstraints = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };

/** The constraints to open the chosen device with (`deviceId` unset: the system
 *  default), or with a device that is only preferred, so a saved device that is
 *  gone does not fail the call. */
export function micConstraints(deviceId?: string): MediaTrackConstraints {
  return deviceId ? { ...MIC_CONSTRAINTS, deviceId: { ideal: deviceId } } : { ...MIC_CONSTRAINTS };
}

/** Opens the microphone. Rejects when the mic is wanted now and cannot be opened. */
export function openMicStream(deviceId?: string): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({ audio: micConstraints(deviceId) });
}

// vad-web 0.0.31 starts a segment preSpeechPadMs (default 800) before speech was
// detected, in 512-sample (32 ms) frames: 25 of them plus the frame that tripped it.
// A streamed utterance sends the same audio from the same point.
export const PRE_SPEECH_FRAMES = Math.floor(800 / 32) + 1;

/** Shorter than any word: a blip, not a turn. */
export const MIN_UTTER_SAMPLES = MIC_RATE * 0.25;

export const RMS_GATE = 0.006;      // reject near-silence; low enough to hear a soft talker
const AMBIENT_MAX = 0.03;           // the learned floor can never rise enough to swallow speech
const AMBIENT_LEARN = 0.05;         // how far one idle frame moves it toward the room
const GATE_MARGIN = 1.6;            // the gate sits this far above the learned floor

/** Root-mean-square of a frame: what the engine gates on. */
export function rmsOf(a: Float32Array): number { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * a[i]!; return Math.sqrt(s / a.length); }

/** The room's ambient level, learned only while idle, and what it was before each
 *  of the last PRE_SPEECH_FRAMES frames (a segment opens with the floor as it was
 *  before Silero fired, not the first words of the talker). */
export class NoiseFloor {
  private floors: number[] = Array(PRE_SPEECH_FRAMES).fill(0.002);
  private at = 0;
  private floor = 0.002;

  get value() { return this.floor; }
  set value(v: number) { this.floor = v; }

  /** Ages the history: once per frame, before learning from that frame. */
  nextFrame() {
    this.floors[this.at] = this.floor;
    this.at = (this.at + 1) % PRE_SPEECH_FRAMES;
  }

  /** The floor as it stood before this frame — the room a segment opened in. */
  onsetFloor(): number { return this.floors[this.at]!; }

  /**
   * Learn the room's ambient noise floor WHILE IDLE (never during the user's own
   * speech), so the reject-gate rises in a loud room / around a TV and stops
   * background chatter tripping a turn — but stays at the fixed floor in a quiet
   * room so a soft talker is still heard. ponytail: a real room needs this
   * calibration; the clamp keeps it from ever rising high enough to swallow speech.
   */
  learn(rms: number) { this.floor = Math.min(AMBIENT_MAX, this.floor + (rms - this.floor) * AMBIENT_LEARN); }

  // Reject threshold: the fixed floor, or a margin above the learned room noise
  // (whichever is higher), capped so it can't rise enough to reject real speech.
  gate(): number { return Math.min(AMBIENT_MAX, Math.max(RMS_GATE, this.floor * GATE_MARGIN)); }
}

/** Everything a hold hears, 16 kHz mono, from the moment the microphone opens. */
export type Tape = { stop(lateMs: number): Promise<Float32Array>; peek(): Float32Array };
/** For a hold that began before the VAD could: loading it takes a moment, and
 *  words said meanwhile would be gone. A ScriptProcessor, not a worklet, since a
 *  worklet's module load is the very wait this covers. */
export function tapeOf(stream: MediaStream): Tape {
  const ctx = new AudioContext({ sampleRate: MIC_RATE });
  const node = ctx.createScriptProcessor(1024, 1, 1);
  const parts: Float32Array[] = [];
  let total = 0;
  node.onaudioprocess = (e) => { const p = e.inputBuffer.getChannelData(0).slice(); parts.push(p); total += p.length; };
  ctx.createMediaStreamSource(stream).connect(node);
  // A ScriptProcessor runs only on a path to the output. It writes nothing, so nothing plays.
  node.connect(ctx.destination);
  void ctx.resume();
  const all = () => {
    const out = new Float32Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
  };
  return {
    // Cut at the release, `lateMs` ago, on the context's clock: what came after it is not the hold's.
    async stop(lateMs) {
      const end = Math.max(0, Math.round((ctx.currentTime - lateMs / 1000) * MIC_RATE));
      for (let i = 0; i < 6 && total < end; i++) await new Promise((r) => setTimeout(r, 50)); // the last buffer, in flight
      node.onaudioprocess = null;
      void ctx.close();
      return all().subarray(0, end);
    },
    /** All of it so far, for a caption while the hold goes on. O(n) in its samples. */
    peek: all,
  };
}

/** The one listener a mic track needs. "ended" fires only when the browser ends a
 *  track — unplugged, or its permission revoked — never for our own stop().
 *  Returns the remover, so the caller can let go of the track with it. */
export function watchMicEnded(track: MediaStreamTrack | null, onEnded: () => void): () => void {
  track?.addEventListener("ended", onEnded);
  return () => track?.removeEventListener("ended", onEnded);
}

// Tap the mic stream with an AnalyserNode for a live frequency spectrum. Runs
// alongside the VAD (a MediaStream feeds many consumers); the analyser isn't
// connected onward, so nothing is played back (no echo).
export class MicSpectrum {
  private ctx: AudioContext | null = null;
  private src: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private freq: Uint8Array | null = null;

  attach(stream: MediaStream) {
    try {
      this.detach();
      if (!this.ctx) this.ctx = new AudioContext();
      const ctx = this.ctx;
      const a = ctx.createAnalyser();
      a.fftSize = 256;
      a.smoothingTimeConstant = 0.6;
      this.src = ctx.createMediaStreamSource(stream);
      // analyser → muted gain → destination: some engines only process an analyser
      // that's in a path to the destination. Gain 0 → silent (no echo).
      const mute = ctx.createGain();
      mute.gain.value = 0;
      this.src.connect(a);
      a.connect(mute);
      mute.connect(ctx.destination);
      this.analyser = a;
      this.freq = new Uint8Array(a.frequencyBinCount);
    } catch { this.analyser = null; /* spectrum is best-effort */ }
  }

  /** Let go of the device: nothing reads it any more. */
  detach() {
    try { this.src?.disconnect(); } catch { /* */ }
    this.src = null;
  }

  /** N octave-band magnitudes (0..1) of YOUR voice — a real spectrum for the orb. */
  bands(n = 5): number[] {
    if (!this.analyser || !this.freq) return new Array(n).fill(0);
    this.analyser.getByteFrequencyData(this.freq as Uint8Array<ArrayBuffer>);
    return octaveBands(this.freq, n);
  }

  close() {
    try { this.src?.disconnect(); } catch { /* */ }
    try { void this.ctx?.close(); } catch { /* */ }
    this.src = null; this.analyser = null; this.freq = null; this.ctx = null;
  }
}

/** Raw little-endian PCM16 bytes for Float32 samples in [-1, 1]: the wire format
 *  of /realtime, and the encoder for pcm16ToFloat32(). NOT pcmDecoder()'s
 *  little-endian Float32 — sending Float32 where PCM16 is expected is static. */
export function pcm16FromFloat32(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    // One scale for both signs, so a decode→encode round trip is exact; the
    // positive end clamps at 32767, which is the only code that overflows.
    const s = Math.max(-0x8000, Math.min(0x7fff, Math.round(Math.max(-1, Math.min(1, samples[i]!)) * 0x8000)));
    out[2 * i] = s & 0xff;
    out[2 * i + 1] = (s >> 8) & 0xff;
  }
  return out;
}

export const PCM16_FRAME_MS = 80; // inside /realtime's 60–100 ms per frame

/** Coalesces arbitrary mic frames into whole PCM16LE frames of `ms`, carrying the
 *  samples that do not fill one into the next. 16 kHz mono, no header. */
export class Pcm16Framer {
  private size: number;
  private carry = new Float32Array(0);

  constructor(ms = PCM16_FRAME_MS) {
    this.size = Math.max(1, Math.round((MIC_RATE * ms) / 1000));
  }

  /** Every whole frame `frame` completed, oldest first. */
  push(frame: Float32Array): Uint8Array[] {
    const all = new Float32Array(this.carry.length + frame.length);
    all.set(this.carry);
    all.set(frame, this.carry.length);
    const whole = all.length - (all.length % this.size);
    this.carry = all.slice(whole);
    const out: Uint8Array[] = [];
    for (let off = 0; off < whole; off += this.size) out.push(pcm16FromFloat32(all.subarray(off, off + this.size)));
    return out;
  }

  /** The tail the frames so far left over, when the utterance ends. */
  flush(): Uint8Array {
    const tail = this.carry;
    this.carry = new Float32Array(0);
    return tail.length ? pcm16FromFloat32(tail) : new Uint8Array(0);
  }

  reset() { this.carry = new Float32Array(0); }
}
