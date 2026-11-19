// Raw PCM plumbing for the voice engines. Pure, so it is unit-tested.

/** Decodes a stream of little-endian Float32 bytes whose network chunks can end
 *  mid-sample: the 1-3 trailing bytes are carried into the next chunk. One copy
 *  per chunk, O(bytes). */
export function pcmDecoder(): (bytes: Uint8Array) => Float32Array {
  let carry = new Uint8Array(0);
  return (bytes) => {
    // A fresh buffer is 4-byte aligned, which a Float32Array view requires.
    const all = new Uint8Array(carry.length + bytes.length);
    all.set(carry);
    all.set(bytes, carry.length);
    const whole = all.length - (all.length % 4);
    carry = all.slice(whole);
    return new Float32Array(all.buffer, 0, whole / 4);
  };
}

/** Decodes little-endian **PCM16** bytes (mono: /realtime's frames, Gemini's
 *  24 kHz output) to Float32 samples in [-1, 1). NOT pcmDecoder(): that one reads
 *  Float32LE, and feeding PCM16 through it is static that sounds like a transport
 *  bug. One scale for both signs (/32768, as `readInt16LE(...) / 32768` elsewhere
 *  in this repo) so a sample survives a decode→encode round trip exactly. */
export function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  // An odd trailing byte is a frame cut in half: it cannot be read, so it is
  // dropped rather than misread as the low byte of the next sample.
  const whole = bytes.length - (bytes.length % 2);
  const out = new Float32Array(whole / 2);
  for (let i = 0, o = 0; i < whole; i += 2, o++) {
    const s = (bytes[i]! | (bytes[i + 1]! << 8)) << 16 >> 16; // sign-extend 16 bits
    out[o] = s / 0x8000;
  }
  return out;
}

/** The last `size` mic frames, so speech the VAD only recognised a few frames in
 *  can still be sent from its true start. push is O(1); drain is O(samples held). */
export class FrameRing {
  private frames: (Float32Array | undefined)[];
  private next = 0;
  private count = 0;

  constructor(size: number) { this.frames = new Array(size); }

  push(frame: Float32Array) {
    this.frames[this.next] = frame;
    this.next = (this.next + 1) % this.frames.length;
    this.count = Math.min(this.count + 1, this.frames.length);
  }

  /** Every held frame, oldest first, as one array; the ring is left empty. */
  drain(): Float32Array {
    const n = this.frames.length;
    const held: Float32Array[] = [];
    for (let i = this.count; i > 0; i--) held.push(this.frames[(this.next - i + n) % n]!);
    const out = new Float32Array(held.reduce((s, f) => s + f.length, 0));
    let off = 0;
    for (const f of held) { out.set(f, off); off += f.length; }
    this.clear();
    return out;
  }

  clear() { this.frames.fill(undefined); this.count = 0; }
}
