// Guards the PCM16 path (Gemini's 24 kHz output, and /realtime's frames) and its
// inverse. Its whole point is that it is NOT pcmDecoder(): that one reads
// little-endian Float32, and 24 kHz PCM16 through it is static.
import assert from "node:assert";
import { test } from "vitest";
import { pcm16ToFloat32 } from "./pcm.ts";
import { pcm16FromFloat32 } from "./micCapture.ts";

/** Little-endian PCM16 bytes for signed samples. */
const pcm16 = (...samples: number[]) => {
  const out = new Uint8Array(samples.length * 2);
  samples.forEach((s, i) => { out[2 * i] = s & 0xff; out[2 * i + 1] = (s >> 8) & 0xff; });
  return out;
};

/** What a sample becomes inside the Float32Array the decoder returns. */
const f32 = (x: number) => Math.fround(x);

test("pcm16ToFloat32: the full-scale codes map to ±1 and silence to 0", () => {
  const out = pcm16ToFloat32(pcm16(-32768, 0, 32767));
  assert.equal(out.length, 3);
  assert.equal(out[0], -1.0);
  assert.equal(out[1], 0);
  assert.ok(Math.abs(out[2]! - 1) < 1e-4, `32767 -> ${out[2]}`);
  // −32768 is the one code with no positive twin, so it is the only exact −1;
  // 32767 sits a hair under +1, which is what "≈1.0" means here.
  assert.deepEqual([...pcm16ToFloat32(pcm16(-32768))], [-1.0]);
  assert.equal(out[2], f32(32767 / 32768));
});

test("pcm16ToFloat32: mid-scale and negative codes decode to the right sign and ratio", () => {
  assert.deepEqual([...pcm16ToFloat32(pcm16(16384, -16384, 1, -1))], [0.5, -0.5, f32(1 / 32768), f32(-1 / 32768)]);
  const sign = pcm16ToFloat32(pcm16(-100, 100));
  assert.ok(sign[0]! < 0 && sign[1]! > 0);
});

test("pcm16ToFloat32: little-endian, not big-endian", () => {
  // 0x0100 read LE from "00 01"; read BE from "01 00" it is 1.
  assert.deepEqual([...pcm16ToFloat32(new Uint8Array([0x00, 0x01]))], [f32(256 / 32768)]);
  assert.deepEqual([...pcm16ToFloat32(new Uint8Array([0x01, 0x00]))], [f32(1 / 32768)]);
});

test("pcm16ToFloat32: an odd trailing byte is dropped, not misread", () => {
  // A network chunk can end mid-sample: the half sample cannot be decoded.
  const bytes = new Uint8Array([...pcm16(-32768, 0, 16384), 0x7f]);
  const out = pcm16ToFloat32(bytes);
  assert.equal(out.length, 3);
  assert.deepEqual([...out], [-1.0, 0, 0.5]);
  // The same as the aligned slice: the stray byte reads into nothing.
  assert.deepEqual([...out], [...pcm16ToFloat32(bytes.subarray(0, 6))]);
  // A lone byte decodes to nothing at all.
  assert.equal(pcm16ToFloat32(new Uint8Array([0x40])).length, 0);
  assert.equal(pcm16ToFloat32(new Uint8Array(0)).length, 0);
});

test("pcm16ToFloat32 ≠ pcmDecoder: Float32 bytes through it are not PCM16", () => {
  // The trap this file exists for: the same Float32 bytes must not decode alike.
  const floatBytes = new Uint8Array(new Float32Array([0.5, -0.25]).buffer);
  const asPcm16 = pcm16ToFloat32(floatBytes);
  assert.equal(asPcm16.length, 4); // 8 bytes = 4 PCM16 samples, not 2 floats
  assert.notDeepEqual([...asPcm16], [0.5, -0.25]);
});

test("pcm16FromFloat32: encodes the inverse, clamped to full scale", () => {
  const bytes = pcm16FromFloat32(new Float32Array([-1, 0, 1, -0.5, 0.5]));
  assert.equal(bytes.length, 10);
  assert.deepEqual([...pcm16ToFloat32(bytes)], [-1, 0, f32(32767 / 32768), -0.5, 0.5]);
  // Out-of-range samples clamp instead of wrapping to the far side.
  assert.deepEqual([...pcm16FromFloat32(new Float32Array([-2, 2]))], [...pcm16FromFloat32(new Float32Array([-1, 1]))]);
});
