// File: scripts/probes/probe-gemini-bidi.mjs
// Stage 0 probe: measure the four Gemini Live facts the realtime engine design
// depends on. Run against the real key:
//
//   node scripts/probes/probe-gemini-bidi.mjs                 # key from env or OpenLive's store
//   node scripts/probes/probe-gemini-bidi.mjs --bn-pcm bn.pcm # use a captured 16 kHz PCM16LE mono file
//   node scripts/probes/probe-gemini-bidi.mjs --no-bn-script-hint  # bare transcription, no script instruction
//
// The key is never printed. It is read from GEMINI_API_KEY / GOOGLE_API_KEY,
// --key / --key-file, or OpenLive's own encrypted store (secrets/providers.json)
// through the resolver the agent uses at runtime.
//
// Measured 2026-11-19 on this key: bare transcription comes back romanized
// ("Shono amar app er header title ta bodle..."); the system instruction alone
// is flaky (2/3); `inputAudioTranscription.languageCodes:["bn-IN"]` was 3/3 in
// Bengali script and is the default here. The Live docs say languageCodes is
// unsupported, so the client must treat a config rejection as a fallback to
// romanized text (Banglish), not as fatal.
//
// Matrix printed (exactly the names the build order requires):
//   initialHistorySilent   seeding via historyConfig produces no model turn
//   midTurnFalseSilent     mid-session clientContent turnComplete:false is silent
//   resumePreservedContext a handle survives a forced close and keeps context
//   bengaliTranscriptOk    a Bengali audio turn transcribes to Bengali text

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Node's built-in WebSocket (Node >= 22) keeps this probe dependency-free: the
// repo's `ws` lives under services/agent and is not resolvable from scripts/.
class WsClient {
  constructor(url) {
    this.raw = new WebSocket(url);
    this.raw.binaryType = "arraybuffer";
    this.listeners = new Map();
    this.raw.addEventListener("open", () => this.emit("open"));
    this.raw.addEventListener("message", (ev) => this.emit("message", ev.data));
    this.raw.addEventListener("error", (ev) => this.emit("error", ev.error ?? new Error(ev.message ?? "websocket error")));
    this.raw.addEventListener("close", (ev) => this.emit("close", ev.code, ev.reason));
  }
  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
  }
  off(event, fn) {
    this.listeners.set(event, (this.listeners.get(event) ?? []).filter((f) => f !== fn));
  }
  once(event, fn) {
    const wrap = (...args) => {
      this.off(event, wrap);
      fn(...args);
    };
    this.on(event, wrap);
  }
  emit(event, ...args) {
    for (const fn of [...(this.listeners.get(event) ?? [])]) fn(...args);
  }
  send(data) {
    this.raw.send(data);
  }
  close(code, reason) {
    this.raw.close(code, reason);
  }
  terminate() {
    this.raw.close();
  }
}

const isValidSetupFrame = (frame) => frame && typeof frame === "object" && "setupComplete" in frame;

function parseFrame(raw) {
  if (typeof raw === "string") return JSON.parse(raw);
  if (raw instanceof ArrayBuffer) return JSON.parse(Buffer.from(raw).toString("utf8"));
  if (ArrayBuffer.isView(raw)) return JSON.parse(Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength).toString("utf8"));
  return JSON.parse(String(raw));
}

const MODEL = "models/gemini-3.8-live";
const WS_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const TTS_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-3.8-flash-tts"];
const BENGALI_LINE = "শোনো, আমার অ্যাপের হেডার টাইটেলটা বদলে ওপেনলাইভ প্রো করো।";
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(SCRIPT_DIR, "../..");

function parseArgs(argv) {
  const out = { timeoutMs: 20_000, turnTimeoutMs: 15_000, bnScriptHint: true, bnLang: "bn-IN" };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--key" && argv[i + 1]) out.key = argv[++i];
    else if (a === "--key-file" && argv[i + 1]) out.keyFile = argv[++i];
    else if (a === "--bn-pcm" && argv[i + 1]) out.bnPcm = argv[++i];
    else if (a === "--bn-pcm-out" && argv[i + 1]) out.bnPcmOut = argv[++i];
    else if (a === "--bn-lang" && argv[i + 1]) out.bnLang = argv[++i];
    else if (a === "--no-bn-lang") out.bnLang = null;
    else if (a === "--bn-script-hint") out.bnScriptHint = true;
    else if (a === "--no-bn-script-hint") out.bnScriptHint = false;
    else if (a === "--only" && argv[i + 1]) out.only = argv[++i];
    else if (a === "--model" && argv[i + 1]) out.model = argv[++i];
    else if (a === "--timeout" && argv[i + 1]) out.timeoutMs = Number(argv[++i]);
  }
  return out;
}

// ── key resolution (never printed) ───────────────────────────────────────────

function homesToTry() {
  const homes = [];
  if (process.env.OPENLIVE_HOME?.trim()) homes.push(path.resolve(process.env.OPENLIVE_HOME.trim()));
  if (process.env.OPENLIVE_DATA_DIR?.trim()) homes.push(path.resolve(process.env.OPENLIVE_DATA_DIR.trim()));
  homes.push(path.join(os.homedir(), ".openlive"));
  homes.push(path.join(REPO_ROOT, "data"));
  return [...new Set(homes)];
}

async function keyFromOpenLiveStore() {
  let homeModule;
  try {
    homeModule = await import(new URL("../../packages/shared/src/home/index.mjs", import.meta.url));
  } catch {
    return null;
  }
  for (const home of homesToTry()) {
    const P = homeModule.layout(home);
    if (!fs.existsSync(P.providers) || !fs.existsSync(P.encKey)) continue;
    let rows;
    try {
      rows = JSON.parse(fs.readFileSync(P.providers, "utf8"));
    } catch {
      continue;
    }
    if (!Array.isArray(rows)) continue;
    const row = rows.find((r) => r.kind === "google" || /gemini/i.test(String(r.name ?? "")));
    if (!row?.apiKeyCiphertext) continue;
    try {
      const key = homeModule.decrypt(homeModule.loadKey(P.encKey), row.apiKeyCiphertext).trim();
      if (key) return key;
    } catch {
      // Wrong home/key pair; keep looking.
    }
  }
  return null;
}

function keyFromFile(file) {
  if (!file || !fs.existsSync(file)) return null;
  const raw = fs.readFileSync(file, "utf8").trim();
  if (raw.startsWith("AIza")) return raw;
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed.google === "string") return parsed.google.trim();
    if (typeof parsed.gemini === "string") return parsed.gemini.trim();
    if (typeof parsed.apiKey === "string") return parsed.apiKey.trim();
  } catch {
    /* not plain JSON */
  }
  return null;
}

async function resolveApiKey(args) {
  const candidates = [
    args.key?.trim(),
    process.env.GEMINI_API_KEY?.trim(),
    process.env.GOOGLE_API_KEY?.trim(),
    keyFromFile(args.keyFile),
    await keyFromOpenLiveStore(),
  ];
  const key = candidates.find(Boolean);
  if (!key) {
    throw new Error(
      "No Gemini API key found. Set GEMINI_API_KEY/GOOGLE_API_KEY, pass --key/--key-file, " +
        "or configure Google Gemini in OpenLive (its encrypted store is read, never printed).",
    );
  }
  return key;
}

// ── Gemini Live plumbing ─────────────────────────────────────────────────────

function openSession(apiKey, setupObj, timeoutMs) {
  return new Promise((resolve, reject) => {
    const ws = new WsClient(`${WS_URL}?key=${encodeURIComponent(apiKey)}`);
    const events = [];
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { ws.terminate(); } catch { /* ignore */ }
      reject(new Error(`setupComplete timeout after ${timeoutMs} ms`));
    }, timeoutMs);

    ws.once("open", () => ws.send(JSON.stringify({ setup: setupObj })));
    ws.on("message", (raw) => {
      let frame;
      try {
        frame = parseFrame(raw);
      } catch {
        return;
      }
      events.push(frame);
      if (!settled && isValidSetupFrame(frame)) {
        settled = true;
        clearTimeout(timer);
        resolve({ ws, events });
      }
    });
    ws.once("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(err);
    });
    ws.once("close", (code, reasonBuf) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`closed before setupComplete (${code}): ${reasonBuf?.toString("utf8") ?? ""}`));
    });
  });
}

/** Collect frames for a window, or until predicate says stop. */
function collectFrames(ws, durationMs, stopPredicate) {
  return new Promise((resolve) => {
    const frames = [];
    let closed = null;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      ws.off("message", onMessage);
      ws.off("close", onClose);
      resolve({ frames, closed });
    };
    const onMessage = (raw) => {
      try {
        const frame = parseFrame(raw);
        frames.push(frame);
        if (stopPredicate?.(frame, frames)) finish();
      } catch {
        /* ignore malformed frames */
      }
    };
    const onClose = (code, reasonBuf) => {
      closed = { code, reason: reasonBuf?.toString("utf8") ?? "" };
      finish();
    };
    const timer = setTimeout(finish, durationMs);
    ws.on("message", onMessage);
    ws.on("close", onClose);
  });
}

const outputText = (frames) =>
  frames.map((f) => f.serverContent?.outputTranscription?.text ?? "").join("").trim();
const inputText = (frames) =>
  frames.map((f) => f.serverContent?.inputTranscription?.text ?? "").join("").trim();
const hasAudioOrOutput = (frames) =>
  frames.some(
    (f) =>
      Boolean(f.serverContent?.outputTranscription?.text) ||
      Boolean(f.serverContent?.modelTurn?.parts?.some((p) => p.inlineData?.data)),
  );

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function probeInitialHistory(apiKey, model, args) {
  const { ws } = await openSession(
    apiKey,
    {
      model,
      generationConfig: { responseModalities: ["AUDIO"] },
      systemInstruction: { parts: [{ text: "Keep answers under 15 words." }] },
      outputAudioTranscription: {},
      historyConfig: { initialHistoryInClientContent: true },
    },
    args.timeoutMs,
  );

  try {
    ws.send(
      JSON.stringify({
        clientContent: {
          turns: [
            { role: "user", parts: [{ text: "Remember the access code AZURE-FALCON-94." }] },
            { role: "model", parts: [{ text: "Access code AZURE-FALCON-94 stored." }] },
          ],
          turnComplete: true,
        },
      }),
    );
    const seedWindow = await collectFrames(ws, 2200);
    const initialHistorySilent = !seedWindow.closed && !hasAudioOrOutput(seedWindow.frames);

    ws.send(
      JSON.stringify({
        realtimeInput: { text: "What is the exact access code from earlier? Answer with just the code." },
      }),
    );
    const answerWindow = await collectFrames(ws, args.turnTimeoutMs, (f) => f.serverContent?.turnComplete === true);
    const transcript = outputText(answerWindow.frames);
    return {
      initialHistorySilent,
      recalled: /AZURE[-\s]?FALCON[-\s]?94/i.test(transcript),
      seedFrames: seedWindow.frames.length,
      transcript,
      closed: answerWindow.closed,
    };
  } finally {
    try { ws.close(); } catch { /* ignore */ }
  }
}

async function probeMidTurnFalse(apiKey, model, args) {
  const { ws } = await openSession(
    apiKey,
    {
      model,
      generationConfig: { responseModalities: ["AUDIO"] },
      systemInstruction: { parts: [{ text: "Keep answers under 15 words." }] },
      outputAudioTranscription: {},
      contextWindowCompression: { slidingWindow: {} },
    },
    args.timeoutMs,
  );

  try {
    ws.send(
      JSON.stringify({
        clientContent: {
          turns: [{ role: "user", parts: [{ text: "[STATE_SYNC] The current task CODENAME is MANGO-77." }] }],
          turnComplete: false,
        },
      }),
    );
    const silentWindow = await collectFrames(ws, 2200);
    const midTurnFalseSilent = !silentWindow.closed && !hasAudioOrOutput(silentWindow.frames);

    ws.send(JSON.stringify({ realtimeInput: { text: "What is the CODENAME from the state sync? Just the word." } }));
    const recallWindow = await collectFrames(ws, args.turnTimeoutMs, (f) => f.serverContent?.turnComplete === true);
    const transcript = outputText(recallWindow.frames);
    return {
      midTurnFalseSilent,
      recalled: /MANGO[-\s]?77/i.test(transcript),
      transcript,
      closed: recallWindow.closed,
    };
  } finally {
    try { ws.close(); } catch { /* ignore */ }
  }
}

async function probeSessionResumption(apiKey, model, args) {
  let resumeHandle = "";
  let tokensBefore = 0;
  const first = await openSession(
    apiKey,
    {
      model,
      generationConfig: { responseModalities: ["AUDIO"] },
      systemInstruction: { parts: [{ text: "Keep answers under 15 words." }] },
      outputAudioTranscription: {},
      contextWindowCompression: { slidingWindow: {} },
      sessionResumption: {},
    },
    args.timeoutMs,
  );

  try {
    first.ws.send(
      JSON.stringify({
        realtimeInput: { text: "Remember the session token RIVER-31 and say OK." },
      }),
    );
    const turn = await collectFrames(first.ws, args.turnTimeoutMs, (f) => {
      if (f.usageMetadata?.promptTokenCount) tokensBefore = f.usageMetadata.promptTokenCount;
      if (f.sessionResumptionUpdate?.resumable && f.sessionResumptionUpdate.newHandle) {
        resumeHandle = f.sessionResumptionUpdate.newHandle;
      }
      return Boolean(resumeHandle) && f.serverContent?.turnComplete === true;
    });
    if (!resumeHandle) {
      const late = await collectFrames(first.ws, 4000, (f) => {
        if (f.sessionResumptionUpdate?.resumable && f.sessionResumptionUpdate.newHandle) {
          resumeHandle = f.sessionResumptionUpdate.newHandle;
          return true;
        }
        return false;
      });
      if (!resumeHandle && turn.closed) {
        return { resumePreservedContext: false, why: `closed before a handle: ${JSON.stringify(turn.closed)}` };
      }
    }
  } finally {
    try { first.ws.close(1000, "probe_forced_close"); } catch { /* ignore */ }
  }

  if (!resumeHandle) {
    return {
      resumePreservedContext: false,
      why: "server issued no resumable handle (sessionResumptionUpdate resumable:true + newHandle never arrived)",
    };
  }

  await sleep(750);
  const second = await openSession(
    apiKey,
    {
      model,
      generationConfig: { responseModalities: ["AUDIO"] },
      systemInstruction: { parts: [{ text: "Keep answers under 15 words." }] },
      outputAudioTranscription: {},
      contextWindowCompression: { slidingWindow: {} },
      sessionResumption: { handle: resumeHandle },
    },
    args.timeoutMs,
  );

  try {
    let tokensAfter = 0;
    second.ws.send(
      JSON.stringify({ realtimeInput: { text: "What session token did I ask you to remember? Answer with just the token." } }),
    );
    const answer = await collectFrames(second.ws, args.turnTimeoutMs, (f) => {
      if (f.usageMetadata?.promptTokenCount) tokensAfter = f.usageMetadata.promptTokenCount;
      return f.serverContent?.turnComplete === true;
    });
    const transcript = outputText(answer.frames);
    return {
      resumePreservedContext: /RIVER[-\s]?31/i.test(transcript),
      handleIssued: true,
      tokensBefore,
      tokensAfter,
      transcript,
      closed: answer.closed,
    };
  } finally {
    try { second.ws.close(); } catch { /* ignore */ }
  }
}

// ── Bengali audio (generated with Gemini TTS when no capture is supplied) ────

function resamplePcm16(pcm, fromRate, toRate) {
  if (fromRate === toRate) return pcm;
  const input = new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.byteLength / 2));
  const outLength = Math.floor((input.length * toRate) / fromRate);
  const out = Buffer.alloc(outLength * 2);
  const ratio = fromRate / toRate;
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = pos - lo;
    const value = Math.max(-32768, Math.min(32767, Math.round(input[lo] * (1 - frac) + input[hi] * frac)));
    out.writeInt16LE(value, i * 2);
  }
  return out;
}

async function generateBengaliPcm(apiKey, model) {
  let lastError = "";
  for (const ttsModel of TTS_MODELS) {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${ttsModel}:generateContent`,
      {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          contents: [{ parts: [{ text: BENGALI_LINE }] }],
          generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Kore" } } },
          },
        }),
      },
    );
    if (!res.ok) {
      lastError = `${ttsModel}: HTTP ${res.status} ${(await res.text()).slice(0, 240)}`;
      continue;
    }
    const body = await res.json();
    const inline = body.candidates?.[0]?.content?.parts?.find((p) => p.inlineData)?.inlineData;
    if (!inline?.data) {
      lastError = `${ttsModel}: no inline audio (${JSON.stringify(body).slice(0, 240)})`;
      continue;
    }
    const rate = Number(/rate=(\d+)/.exec(inline.mimeType ?? "")?.[1] ?? 24000);
    const pcm24 = Buffer.from(inline.data, "base64");
    return { pcm: resamplePcm16(pcm24, rate, 16000), source: `tts:${ttsModel}@${rate}Hz`, text: BENGALI_LINE };
  }
  throw new Error(`Bengali TTS generation failed — ${lastError}`);
}

async function probeBengali(apiKey, model, args) {
  let pcm;
  let source;
  if (args.bnPcm) {
    pcm = fs.readFileSync(path.resolve(args.bnPcm));
    source = `file:${args.bnPcm}`;
  } else {
    const generated = await generateBengaliPcm(apiKey, model);
    pcm = generated.pcm;
    source = generated.source;
    if (args.bnPcmOut) fs.writeFileSync(path.resolve(args.bnPcmOut), pcm);
  }

  const { ws } = await openSession(
    apiKey,
    {
      model,
      generationConfig: { responseModalities: ["AUDIO"] },
      systemInstruction: {
        parts: [
          {
            text:
              "Always answer naturally in Bengali. Keep it under 15 words." +
              (args.bnScriptHint
                ? " The user speaks Bengali; transcribe their speech verbatim in Bengali script (বাংলা লিপি), never romanized."
                : ""),
          },
        ],
      },
      inputAudioTranscription: args.bnLang ? { languageCodes: [args.bnLang] } : {},
      outputAudioTranscription: {},
      contextWindowCompression: { slidingWindow: {} },
    },
    args.timeoutMs,
  );

  try {
    const chunk = 3200; // 100 ms of 16 kHz PCM16LE
    for (let off = 0; off < pcm.length; off += chunk) {
      ws.send(
        JSON.stringify({
          realtimeInput: {
            audio: {
              data: pcm.subarray(off, Math.min(off + chunk, pcm.length)).toString("base64"),
              mimeType: "audio/pcm;rate=16000",
            },
          },
        }),
      );
      await sleep(90);
    }
    ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    const answer = await collectFrames(ws, Math.max(args.turnTimeoutMs, 20_000), (f) => f.serverContent?.turnComplete === true);
    const heard = inputText(answer.frames);
    const said = outputText(answer.frames);
    const bengali = /[\u0980-\u09FF]/;
    return {
      bengaliTranscriptOk: bengali.test(heard) && /হেডার|টাইটেল|ওপেনলাইভ|OpenLive/i.test(heard),
      audioSource: source,
      inputTranscript: heard,
      outputTranscript: said,
      outputIsBengali: bengali.test(said),
      closed: answer.closed,
    };
  } finally {
    try { ws.close(); } catch { /* ignore */ }
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv);
  const apiKey = await resolveApiKey(args);
  const model = args.model
    ? args.model.startsWith("models/")
      ? args.model
      : `models/${args.model}`
    : MODEL;

  const only = args.only ?? "all";
  const want = (name) => only === "all" || only === name;
  const skip = { initialHistorySilent: true, midTurnFalseSilent: true, resumePreservedContext: true, bengaliTranscriptOk: true };

  const [initRes, midRes, resumeRes, bnRes] = await Promise.allSettled([
    want("init") ? probeInitialHistory(apiKey, model, args) : Promise.resolve(skip),
    want("mid") ? probeMidTurnFalse(apiKey, model, args) : Promise.resolve(skip),
    want("resume") ? probeSessionResumption(apiKey, model, args) : Promise.resolve(skip),
    want("bn") ? probeBengali(apiKey, model, args) : Promise.resolve(skip),
  ]);

  const settle = (r) => (r.status === "fulfilled" ? r.value : { error: String(r.reason?.message ?? r.reason) });
  const init = settle(initRes);
  const mid = settle(midRes);
  const resume = settle(resumeRes);
  const bn = settle(bnRes);

  const matrix = {
    probe: "gemini-bidi",
    model,
    at: new Date().toISOString(),
    initialHistorySilent: init.initialHistorySilent === true,
    midTurnFalseSilent: mid.midTurnFalseSilent === true,
    resumePreservedContext: resume.resumePreservedContext === true,
    bengaliTranscriptOk: bn.bengaliTranscriptOk === true,
    details: { initialHistory: init, midTurnFalse: mid, sessionResumption: resume, bengali: bn },
  };
  console.log(JSON.stringify(matrix, null, 2));
  if (!Object.values({ a: matrix.initialHistorySilent, b: matrix.midTurnFalseSilent, c: matrix.resumePreservedContext, d: matrix.bengaliTranscriptOk }).every(Boolean)) {
    process.exitCode = 2;
  }
}

main().catch((err) => {
  console.error(`probe failed: ${err?.message ?? err}`);
  process.exit(1);
});
