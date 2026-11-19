// Stage 5 pass condition, proved over a LOOPBACK SOCKET (no network, no key, no real model, no
// child agent): a real `ws` client speaks to the real `/realtime` upgrade in `ws.ts`, whose
// `RealtimeSession` talks to a local `GeminiBidiClient` stand-in and a fake ACP host.
//
//   1. 16 kHz PCM16 up -> the native socket; 24 kHz PCM16 down -> `[u32le generation]`+PCM;
//   2. `barge_in` bumps the generation, and so does a Gemini `interrupted`;
//   3. a spoken confirmation (validator + model nomination) dispatches ACP EXACTLY once, and a
//      replay of the same draft never dispatches again;
//   4. a reconnect while `verifying` demotes cleanly, with zero duplicate dispatches;
//   5. `task_finished` wakes a sleeping socket once, and only when idle;
//   6. clean transcript + tool cards are persisted, `[LEDGER_COLD_START]`/`[STATE_SYNC]` never;
//   7. the six Live tool handlers answer from memory in well under their budget and never await
//      the coding agent (a hung agent must not freeze the conversation).
import { EventEmitter } from "node:events";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WebSocket } from "ws";
import { GEMINI_TOOL_NAMES } from "@openlive/shared";
import { afterAll, afterEach, beforeAll, describe, expect, test, vi } from "vitest";

// The db resolves its data dir at import time, so this has to be set first.
const home = mkdtempSync(join(tmpdir(), "ol-realtime-"));
process.env.OPENLIVE_HOME = home;

/** Every fake the module mocks push into; `vi.hoisted` keeps it reachable from the factories. */
const fake = vi.hoisted(() => ({
  clients: [] as unknown[],
  hosts: [] as unknown[],
}));

/** A BidiGenerateContent stand-in with the exact public surface `RealtimeSession` uses. */
class FakeGemini extends EventEmitter {
  readonly cfg: Record<string, any>;
  phase = "idle";
  busy = false;
  muted = false;
  closed = false;
  connects = 0;
  /** Raw 16 kHz PCM the session forwarded for the native socket. */
  mic: Buffer[] = [];
  /** `functionResponses` the session answered with, in order. */
  responses: Array<Record<string, any>> = [];
  silentSyncs: Array<Array<Record<string, any>>> = [];
  texts: string[] = [];
  wakeCalls = 0;
  sleepCalls = 0;
  coldHistory: Array<Record<string, any>> = [];
  private callId = 0;

  constructor(cfg: Record<string, any>) {
    super();
    this.cfg = cfg;
    fake.clients.push(this);
  }

  currentPhase(): string { return this.phase; }
  currentResumeHandle(): string { return ""; }
  currentCheckpointSeq(): number { return 0; }
  isTurnBusy(): boolean { return this.busy; }

  async connect(): Promise<void> {
    this.connects += 1;
    // What the real client does on a cold connect: ask for the bounded ledger header.
    this.coldHistory = (await this.cfg.getColdHistory?.()) ?? [];
    this.phase = "live";
    this.emit("phase", { phase: "live" });
  }
  async sleep(reason?: string): Promise<void> {
    this.sleepCalls += 1;
    this.phase = "sleeping";
    this.emit("phase", { phase: "sleeping", reason });
  }
  async wake(preRoll?: Buffer[]): Promise<void> {
    this.wakeCalls += 1;
    if (preRoll?.length) this.mic.push(...preRoll);
    this.phase = "live";
    this.emit("phase", { phase: "live" });
  }
  async close(): Promise<void> {
    this.closed = true;
    this.phase = "closed";
    this.emit("phase", { phase: "closed" });
  }
  pushMicPcm16(chunk: Buffer): void { this.mic.push(Buffer.from(chunk)); }
  setMuted(m: boolean): void { this.muted = m; }
  sendAudioStreamEnd(): void {}
  sendRealtimeText(text: string): void { this.texts.push(text); }
  sendSilentSync(turns: Array<Record<string, any>>): void { this.silentSyncs.push(turns); }
  sendToolResponses(responses: Array<Record<string, any>>): boolean {
    this.responses.push(...responses);
    return true;
  }

  // ── the test's side of the mock server ──
  /** One 24 kHz model turn's audio, alive until `turnComplete`. */
  say(pcm: Buffer): void { this.busy = true; this.emit("audio", pcm); }
  interrupted(): void { this.emit("interrupted"); }
  turnComplete(): void { this.busy = false; this.emit("turnComplete"); }
  transcript(text: string): void { this.emit("inputTranscription", { text, finished: true }); }
  /** A tool call, answered synchronously (the whole point of the non-awaiting discipline). */
  callTool(name: string, args: Record<string, unknown>): { response: Record<string, any>; spentMs: number } {
    const id = `call_${++this.callId}`;
    const before = this.responses.length;
    const t0 = performance.now();
    this.emit("toolCall", { calls: [{ id, name, args }], socketSeq: 1 });
    const spentMs = performance.now() - t0;
    const sent = this.responses[before] as Record<string, any> | undefined;
    if (!sent) throw new Error(`no toolResponse was sent for ${name}`);
    return { response: sent.response as Record<string, any>, spentMs };
  }
}

/** A coding agent that is never spawned: it records dispatch, and can be held "hung" on purpose. */
class FakeAcpHost {
  readonly ports: Record<string, any>;
  id: string | null = "hermes";
  cwd = "";
  disposed = false;
  meta: Record<string, any> | null = null;
  task: { taskId: string; spec: string; signal: AbortSignal; startedAt: number } | null = null;
  /** ACP events the next turn emits, in order. */
  script: Array<Record<string, any>> = [];
  /** false: `runTurn` hangs until `finish()` — the "frozen coding agent" case. */
  autoResolve = true;
  pendingRun: (() => void) | null = null;
  lastInput: { text?: string; frames?: unknown[] } | null = null;
  calls = { restore: 0, bind: 0, startTask: 0, runTurn: 0, endTask: 0, steer: 0, dispose: 0, ask: 0 };
  answeredPerm: string[] = [];
  answeredElicit: string[] = [];
  guidanceDrains: string[] = [];

  constructor(ports: Record<string, any>) {
    this.ports = ports;
    fake.hosts.push(this);
  }

  get ready(): Promise<void> { return Promise.resolve(); }
  get settling(): Promise<void> { return Promise.resolve(); }
  get running(): boolean { return this.id !== null; }
  get taskRunning(): boolean { return !!this.task && !this.task.signal.aborted; }
  get startMs(): number { return 0; }
  get resumeKind(): undefined { return undefined; }
  get bindStarted(): boolean { return this.calls.bind > 0; }
  boundState(): { agentId: string | null; cwd: string; agentActive: boolean } {
    return { agentId: this.id, cwd: this.cwd, agentActive: !!this.id };
  }
  activeTaskSignal(): AbortSignal { return this.task?.signal ?? AbortSignal.abort(); }
  async restore(): Promise<void> { this.calls.restore += 1; }
  async applyBind(id: string | null, cwd?: string): Promise<void> {
    this.calls.bind += 1;
    this.id = id;
    if (cwd !== undefined) this.cwd = cwd;
    this.ports.send?.({ t: "bound_state", agentId: id, cwd: this.cwd, agentActive: !!id });
  }
  startTask(taskId: string, spec: string, signal: AbortSignal): { taskId: string; spec: string; signal: AbortSignal; startedAt: number } {
    this.calls.startTask += 1;
    this.task = { taskId, spec, signal, startedAt: Date.now() };
    return this.task;
  }
  currentTask(): { taskId: string; spec: string; signal: AbortSignal; startedAt: number } | null { return this.task; }
  endTask(): void { this.calls.endTask += 1; this.task = null; }
  async runTurn(
    input: { text: string; frames: unknown[] },
    emit: (e: Record<string, any>) => Promise<void>,
    _signal: AbortSignal,
  ): Promise<boolean> {
    this.calls.runTurn += 1;
    this.lastInput = input;
    for (const e of this.script.splice(0)) await emit(e);
    if (this.autoResolve) return true;
    await new Promise<void>((resolve) => { this.pendingRun = resolve; });
    return true;
  }
  /** Let the hung turn finish, as a real agent completing its task would. */
  finish(): void { const r = this.pendingRun; this.pendingRun = null; r?.(); }
  async steerTask(_taskId: string, item: Record<string, any>): Promise<Record<string, any>> {
    this.calls.steer += 1;
    return { mechanism: "mcp_guidance", delivered: [item], cancelled: false };
  }
  drainGuidance(taskId: string): { items: unknown[]; text: string } {
    this.guidanceDrains.push(taskId);
    return { items: [], text: "NO_NEW_GUIDANCE" };
  }
  markDelivered(): void {}
  answerPermission(reqId: string, optionId: string): void { this.answeredPerm.push(`${reqId}:${optionId}`); }
  answerElicitation(reqId: string, action: string): void { this.answeredElicit.push(`${reqId}:${action}`); }
  async askPermission(_q: string, _o: unknown[], _t?: string): Promise<string> { this.calls.ask += 1; return "allow"; }
  async askElicitation(): Promise<{ action: "accept" }> { return { action: "accept" }; }
  async setModel(): Promise<void> {}
  async setMode(): Promise<void> {}
  async setOption(): Promise<void> {}
  cut(): void {}
  toolsChanged(): void {}
  hide(emit: unknown): unknown { return emit; }
  dispose(): void { this.calls.dispose += 1; this.disposed = true; }
}

/** An in-process socket for the whitebox session tests (same events a real `ws` emits). */
class FakeSocket extends EventEmitter {
  readonly OPEN = 1;
  readyState = 1;
  sent: Array<string | Buffer> = [];
  send(data: string | Buffer): void { this.sent.push(data); }
  close(): void { this.readyState = 3; this.emit("close"); }
}

vi.mock("./gemini-bidi.js", async (importOriginal) => {
  const real = await importOriginal<Record<string, unknown>>();
  return {
    ...real,
    createGeminiLiveClient: async (cfg: Record<string, any>) => new FakeGemini(cfg),
  };
});

vi.mock("./acp-host.js", async (importOriginal) => {
  const real = await importOriginal<Record<string, unknown>>();
  return { ...real, AcpHost: FakeAcpHost };
});

// `/live` and Flow are heavyweight and irrelevant here; the upgrade gate only needs them to exist.
vi.mock("./session.js", () => ({ LiveSession: class { constructor(_ws: unknown) {} async start() {} } }));
vi.mock("./flow-ws.js", () => ({ FlowLiveSession: class { constructor(_ws: unknown) {} } }));

const { attachLiveWs } = await import("./ws.ts");
const { RealtimeSession, TOOL_BUDGET_MS, cleanForChat } = await import("./realtime-session.ts");
const { closeDbForTests, listMessages } = await import("@openlive/db");

const SECRET = "realtime-test-secret";
const SPEC = "Change the header title to 'OpenLive Pro'";
const SUMMARY = "হেডার শিরোনাম বদলে 'OpenLive Pro' করবো";
const READBACK = "হেডার শিরোনাম বদলে 'OpenLive Pro' করবো — এটা কি পাঠাবো?";
const YES = "হ্যাঁ, পাঠাও।";

let server: Server;
let port = 0;

beforeAll(async () => {
  process.env.OPENLIVE_AGENT_SECRET = SECRET;
  server = createServer((_req, res) => { res.statusCode = 404; res.end(); });
  attachLiveWs(server);
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  port = (server.address() as AddressInfo).port;
});

afterAll(async () => {
  delete process.env.OPENLIVE_AGENT_SECRET;
  for (const c of openConns.splice(0)) await c.close().catch(() => {});
  await new Promise((r) => server.close(r));
  closeDbForTests();
  rmSync(home, { recursive: true, force: true });
});

// ── helpers ────────────────────────────────────────────────────────────────

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitUntil(pred: () => boolean, what: string, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pred()) return;
    await sleep(5);
  }
  throw new Error(`timed out waiting for ${what}`);
}

const latestClient = () => fake.clients[fake.clients.length - 1] as FakeGemini;
const latestHost = () => fake.hosts[fake.hosts.length - 1] as FakeAcpHost;

interface Conn {
  ws: WebSocket;
  messages: Array<Record<string, any>>;
  binaries: Buffer[];
  send: (m: Record<string, unknown>) => void;
  binary: (b: Buffer) => void;
  waitFor: (pred: () => boolean, what: string, timeoutMs?: number) => Promise<void>;
  close: () => Promise<void>;
}

const openConns: Conn[] = [];

/** A real WebSocket to the real `/realtime` upgrade, with the same ?token= the desktop sends. */
function connect(chat: string, lang = "en"): Promise<Conn> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/realtime?chat=${encodeURIComponent(chat)}&lang=${lang}&device=0&token=${SECRET}`);
    const messages: Array<Record<string, any>> = [];
    const binaries: Buffer[] = [];
    ws.on("message", (data, isBinary) => {
      if (isBinary) binaries.push(Buffer.from(data as Buffer));
      else {
        try { messages.push(JSON.parse(String(data))); } catch { /* not JSON: ignore */ }
      }
    });
    ws.once("error", reject);
    ws.once("open", () => {
      const conn: Conn = {
        ws,
        messages,
        binaries,
        send: (m) => ws.send(JSON.stringify(m)),
        binary: (b) => ws.send(b),
        waitFor: (pred, what, timeoutMs) => waitUntil(pred, what, timeoutMs),
        close: () => new Promise<void>((done) => {
          if (ws.readyState === WebSocket.CLOSED) return done();
          ws.once("close", () => done());
          try { ws.close(); } catch { done(); }
        }),
      };
      openConns.push(conn);
      resolve(conn);
    });
  });
}

async function live(chat: string): Promise<Conn> {
  const c = await connect(chat);
  await c.waitFor(() => c.messages.some((m) => m.t === "phase" && m.phase === "live"), "the live phase");
  return c;
}

/** Ask the upgrade for a path and report what the gate did. */
function attempt(path: string): Promise<string> {
  return new Promise((resolve) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}${path}`);
    const done = (r: string) => { try { ws.terminate(); } catch { /* already gone */ } resolve(r); };
    ws.on("open", () => done("accepted"));
    ws.on("unexpected-response", (_q, r) => done(`rejected ${r.statusCode}`));
    ws.on("error", (e) => done(`error ${e.message}`));
    setTimeout(() => done("timeout"), 2000);
  });
}

/** propose -> read-back -> the person's spoken yes. Stops in `verifying` (0 dispatches). */
async function speakConfirmation(c: Conn, gem: FakeGemini): Promise<{ draftId: string; version: number }> {
  const proposed = gem.callTool("propose_instruction", {
    summary_bn: SUMMARY,
    english_spec: SPEC,
    target: "new_task",
    open_questions: [],
  }).response;
  expect(proposed).toMatchObject({ ok: true, state: "proposing" });
  const draftId = String(proposed.draft_id);
  const version = Number(proposed.version);
  await c.waitFor(() => c.messages.some((m) => m.t === "draft_state" && m.draft?.state === "proposing"), "the staged draft");

  const armed = gem.callTool("request_confirmation", { draft_id: draftId, version, readback_bn: READBACK }).response;
  expect(armed).toMatchObject({ ok: true, armed: true });
  await c.waitFor(() => c.messages.some((m) => m.t === "draft_state" && m.draft?.state === "confirming"), "the armed read-back");

  // The renderer's VAD: speech started well after our read-back finished.
  c.send({ t: "vad_state", speaking: true, onsetMs: 5_000, playbackEndMs: 4_000, echoSafe: true });
  await sleep(25); // the vad_state crosses the socket before the transcript does
  gem.transcript(YES);
  await c.waitFor(() => c.messages.some((m) => m.t === "draft_state" && m.draft?.state === "verifying"), "the validated confirmation");
  return { draftId, version };
}

function dispatches(chat: string): Array<Record<string, any>> {
  try {
    const file = join(home, "data", "ledgers", `${chat}.jsonl`);
    return readFileSync(file, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Record<string, any>)
      .filter((r) => r.kind === "task_dispatched");
  } catch {
    return [];
  }
}

afterEach(async () => {
  for (const c of openConns.splice(0)) await c.close().catch(() => {});
});

// ── the route and the binary transport ────────────────────────────────────

describe("/realtime upgrade + the binary transport", () => {
  test("the same secret gate covers /realtime — a pathname check, not a second auth story", async () => {
    expect(await attempt("/realtime?chat=gate")).toMatch(/rejected 401|error.*401/);
    const c = await live("rt-gate");
    expect(c.messages.some((m) => m.t === "phase" && m.phase === "live")).toBe(true);
    await c.close();
  });

  test("?lang= reaches the model as words", async () => {
    // `?lang=` is already gated by `languageSchema` in ws.ts; pick one it carries. (Bengali is not
    // in LANGUAGE_CODES yet — the protocol tells the model to speak it by default, so a call
    // without a supported code still speaks Bengali.)
    const c = await connect("rt-lang", "hi");
    await c.waitFor(() => c.messages.some((m) => m.t === "phase" && m.phase === "live"), "the live phase");
    const gem = latestClient();
    expect(gem.cfg.systemInstruction).toContain("Always reply in Hindi");
    await c.close();
  });

  test("a tool declaration exists for every shared GEMINI_TOOL_NAME, and for nothing else", async () => {
    const c = await live("rt-declarations");
    const gem = latestClient();
    // The declarations the client actually puts in its setup frame, from the shared list.
    const { ORCHESTRATOR_FUNCTION_DECLARATIONS } = await import("./gemini-bidi.ts");
    expect(ORCHESTRATOR_FUNCTION_DECLARATIONS.map((d: { name: string }) => d.name)).toEqual([...GEMINI_TOOL_NAMES]);
    // …and each one is answered rather than reported as unknown.
    for (const name of GEMINI_TOOL_NAMES) {
      const { response } = gem.callTool(name, { summary_bn: "সামারি", english_spec: SPEC, target: "new_task", draft_id: "d", version: 1, reason: "r", request_id: "x", decision: "deny" });
      expect(response.error ?? "").not.toContain("Unknown tool");
    }
    await c.close();
  });

  test("16 kHz PCM16 goes up; 24 kHz PCM16 comes back as [u32le generation]+PCM", async () => {
    const c = await live("rt-binary");
    const gem = latestClient();
    // The client sends whole 60-100 ms frames; the session must forward them unchanged and never
    // re-frame them itself (the "frames forwarded uncoalesced" rollback).
    const first = Buffer.alloc(320, 0x11);
    const second = Buffer.alloc(320, 0x22);
    c.binary(first);
    c.binary(second);
    await c.waitFor(() => gem.mic.length === 2, "the mic frames");
    expect(Buffer.concat(gem.mic)).toEqual(Buffer.concat([first, second]));

    const pcm24 = Buffer.alloc(480, 0x33);
    gem.say(pcm24);
    await c.waitFor(() => c.binaries.length === 1, "the downlink audio frame");
    const frame = c.binaries[0]!;
    expect(frame.length).toBe(4 + pcm24.length);
    expect(frame.readUInt32LE(0)).toBe(0); // the first generation
    expect(frame.subarray(4)).toEqual(pcm24);
  });

  test("barge_in bumps the generation, and so does a Gemini interrupted", async () => {
    const c = await live("rt-barge");
    const gem = latestClient();
    gem.say(Buffer.alloc(480, 1));
    await c.waitFor(() => c.binaries.length === 1, "the first frame");
    expect(c.binaries[0]!.readUInt32LE(0)).toBe(0);

    // The person talks over the model: the renderer flushed playback and told us.
    c.send({ t: "barge_in", epoch: 0, vadOnsetMs: 1234 });
    await c.waitFor(() => c.messages.some((m) => m.t === "barge_in_ack"), "the barge-in ack");
    expect(c.messages.find((m) => m.t === "barge_in_ack")!.epoch).toBe(1);
    gem.say(Buffer.alloc(480, 2));
    await c.waitFor(() => c.binaries.length === 2, "audio after the client barge-in");
    expect(c.binaries[1]!.readUInt32LE(0)).toBe(1);

    // Server-side interruption (VAD/EIC): the same generation moves, so in-flight frames drop.
    gem.interrupted();
    await c.waitFor(() => c.messages.some((m) => m.t === "phase" && m.reason === "interrupted"), "the interruption phase");
    gem.say(Buffer.alloc(480, 3));
    await c.waitFor(() => c.binaries.length === 3, "audio after the interruption");
    expect(c.binaries[2]!.readUInt32LE(0)).toBe(2);
  });
});

// ── the gate: one dispatch, never two ─────────────────────────────────────

describe("a spoken confirmation dispatches ACP exactly once", () => {
  test("validator + nomination dispatch once; a replay, a reconnect and a stale version cannot", async () => {
    const c = await live("rt-dispatch");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false; // the coding agent is still working while we poke the conversation

    const { draftId, version } = await speakConfirmation(c, gem);
    // Key 2 alone never dispatches.
    expect(host.calls.startTask).toBe(0);
    expect(dispatches("rt-dispatch")).toHaveLength(0);

    const confirmed = gem.callTool("confirm_instruction", { draft_id: draftId, version }).response;
    expect(confirmed).toMatchObject({ ok: true, dispatched: true });
    expect(host.calls.startTask).toBe(1);
    expect(host.calls.runTurn).toBe(1);
    // Rule 2: the coding agent receives the English spec, never the Bengali dictation.
    expect(host.lastInput!.text).toBe(SPEC);
    expect(host.lastInput!.text).not.toContain(SUMMARY);
    expect(dispatches("rt-dispatch")).toHaveLength(1);

    // A second nomination in the same window is the trust failure the gate exists to stop.
    const again = gem.callTool("confirm_instruction", { draft_id: draftId, version }).response;
    expect(again).toMatchObject({ ok: true, dispatched: false, reason: "already_dispatched" });
    expect(host.calls.startTask).toBe(1);
    expect(dispatches("rt-dispatch")).toHaveLength(1);

    // A StaleCard Send (key 3) still needs key 2's version: the locked card cannot re-send.
    c.send({ t: "draft_action", action: "send", draftId, version });
    await sleep(40);
    expect(host.calls.startTask).toBe(1);

    // get_status is a memory read: it answers while the agent is hung, and never calls it.
    const before = host.calls.runTurn;
    const status = gem.callTool("get_status", { scope: "detailed" }).response;
    expect(status.ok).toBe(true);
    expect(String(status.digest)).toContain("task_");
    expect(host.calls.runTurn).toBe(before);
    expect(host.pendingRun).not.toBeNull();

    // cancel_task is idempotent.
    const cancelled = gem.callTool("cancel_task", { reason: "the user asked to stop" }).response;
    expect(cancelled).toMatchObject({ ok: true, cancelled: true });
    const cancelledAgain = gem.callTool("cancel_task", { reason: "the user asked to stop" }).response;
    expect(cancelledAgain).toMatchObject({ ok: true, cancelled: false });

    host.finish();
    await waitUntil(() => host.calls.endTask >= 1, "the task to settle");
  });

  test("a reconnect while verifying demotes the draft and dispatches nothing", async () => {
    const c1 = await live("rt-reconnect");
    const gem1 = latestClient();
    const host1 = latestHost();
    const { draftId, version } = await speakConfirmation(c1, gem1);
    expect(dispatches("rt-reconnect")).toHaveLength(0);

    await c1.close();
    await waitUntil(() => host1.disposed, "the dead socket's host to be disposed");
    expect(host1.calls.startTask).toBe(0);

    // The reconnect is a NEW session over the same ledger; the half-verified draft is gone, and
    // the old card's Send cannot recreate a dispatch from it.
    const c2 = await live("rt-reconnect");
    const gem2 = latestClient();
    const host2 = latestHost();
    await c2.waitFor(() => c2.messages.some((m) => m.t === "draft_state" && m.draft === null), "no resurrected draft");
    c2.send({ t: "draft_action", action: "send", draftId, version });
    await sleep(40);
    const stale = gem2.callTool("confirm_instruction", { draft_id: draftId, version }).response;
    expect(stale.ok).toBe(false);
    expect(host2.calls.startTask).toBe(0);
    expect(dispatches("rt-reconnect")).toHaveLength(0);

    // A fresh, complete confirmation on the new socket still dispatches exactly once.
    const fresh = await speakConfirmation(c2, gem2);
    const confirmed = gem2.callTool("confirm_instruction", { draft_id: fresh.draftId, version: fresh.version }).response;
    expect(confirmed.dispatched).toBe(true);
    expect(host2.calls.startTask).toBe(1);
    expect(dispatches("rt-reconnect")).toHaveLength(1);
  });

  test("a model nomination alone cannot dispatch — the validator decides", async () => {
    const c = await live("rt-nominate");
    const gem = latestClient();
    const host = latestHost();

    const proposed = gem.callTool("propose_instruction", { summary_bn: SUMMARY, english_spec: SPEC, target: "new_task" }).response;
    const draftId = String(proposed.draft_id);
    const version = Number(proposed.version);
    gem.callTool("request_confirmation", { draft_id: draftId, version, readback_bn: READBACK });

    // The model "heard" a yes before the transcript caught up: nomination only, refused.
    const early = gem.callTool("confirm_instruction", { draft_id: draftId, version }).response;
    expect(early).toMatchObject({ ok: false, dispatched: false, reason: "awaiting_spoken_confirmation" });
    expect(host.calls.startTask).toBe(0);
    expect(dispatches("rt-nominate")).toHaveLength(0);

    // When the person's own words do arrive, the recorded nomination makes BOTH keys agree, and
    // the dispatch happens exactly once without a second nomination.
    c.send({ t: "vad_state", speaking: true, onsetMs: 5_000, playbackEndMs: 4_000, echoSafe: true });
    await sleep(25);
    gem.transcript(YES);
    await waitUntil(() => host.calls.startTask === 1, "the dispatch the recorded nomination earned");
    expect(host.calls.runTurn).toBe(1);
    expect(dispatches("rt-nominate")).toHaveLength(1);
  });

  test("a mid-task correction is a steer, not a second dispatch, and pushes steer_applied", async () => {
    const c = await live("rt-steer");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false;

    const first = await speakConfirmation(c, gem);
    gem.callTool("confirm_instruction", { draft_id: first.draftId, version: first.version });
    expect(host.calls.startTask).toBe(1);

    // "Actually, only the header — not the README." That is a NEW draft aimed at the running task.
    const steer = gem.callTool("propose_instruction", {
      summary_bn: "শুধু হেডারটা বদলাও, README নোটটা লাগবে না",
      english_spec: "Only change the header title; do not add the README note",
      target: "steer_active_task",
      steer_kind: "steer_pivot",
    }).response;
    expect(steer).toMatchObject({ ok: true, state: "proposing" });
    gem.callTool("request_confirmation", { draft_id: steer.draft_id, version: steer.version, readback_bn: "শুধু হেডারটা — পাঠাবো?" });
    c.send({ t: "vad_state", speaking: true, onsetMs: 20_000, playbackEndMs: 19_000, echoSafe: true });
    await sleep(25);
    gem.transcript("হ্যাঁ, পাঠাও।");
    await c.waitFor(() => c.messages.some((m) => m.t === "draft_state" && m.draft?.state === "verifying"), "the steer armed");
    const applied = gem.callTool("confirm_instruction", { draft_id: steer.draft_id, version: steer.version }).response;

    // A steer goes through the host's steering row; it never starts a second ACP task.
    expect(applied.dispatched).toBe(true);
    expect(host.calls.steer).toBe(1);
    expect(host.calls.startTask).toBe(1);
    expect(dispatches("rt-steer")).toHaveLength(1); // still exactly one dispatch for this chat
    await waitUntil(() => gem.texts.length === 1, "the steer_applied push");
    expect(gem.texts[0]).toContain("correction was applied");

    host.finish();
    await waitUntil(() => host.calls.endTask >= 1, "the task to settle");
  });

  test("a socket that drops while verifying demotes the draft before it can dispatch", async () => {
    const ws = new FakeSocket();
    const host = new FakeAcpHost({ send: () => {} });
    const gem = new FakeGemini({});
    const session = new RealtimeSession(ws as never, "rt-demote", "en", false, {
      host: host as never,
      createClient: async () => gem as never,
    });
    await session.start();
    await waitUntil(() => session.snapshot().phase === "live", "the direct session to go live");

    const proposed = gem.callTool("propose_instruction", { summary_bn: SUMMARY, english_spec: SPEC, target: "new_task" }).response;
    gem.callTool("request_confirmation", { draft_id: proposed.draft_id, version: proposed.version, readback_bn: READBACK });
    ws.emit("message", Buffer.from(JSON.stringify({ t: "vad_state", speaking: true, onsetMs: 5_000, playbackEndMs: 4_000, echoSafe: true })), false);
    gem.transcript(YES);
    expect(session.snapshot().draft?.state).toBe("verifying");

    // The socket dies between key 2 and key 1.
    ws.emit("close");
    expect(session.snapshot().draft?.state).toBe("confirming"); // demoted, not verifying
    expect(host.calls.startTask).toBe(0);
  });
});

// ── the four pushed events ────────────────────────────────────────────────

describe("the pushed events wake a sleeping socket only when idle", () => {
  test("task_finished wakes the socket once; needs_input waits for an idle boundary", async () => {
    const c = await live("rt-wake");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false;

    const { draftId, version } = await speakConfirmation(c, gem);
    gem.callTool("confirm_instruction", { draft_id: draftId, version });
    expect(host.calls.startTask).toBe(1);

    // The call went to sleep while the coding agent worked.
    gem.phase = "sleeping";
    gem.emit("phase", { phase: "sleeping" });

    host.finish(); // task_finished arrives while the socket is asleep and the model is idle
    await waitUntil(() => gem.texts.length === 1, "the pushed task_finished");
    expect(gem.wakeCalls).toBe(1);
    expect(gem.texts[0]).toContain("[ORCHESTRATOR EVENT]");
    expect(gem.texts[0]).toContain("finished");
    // The ledger says one dispatch; the socket was woken once, not per event.
    expect(dispatches("rt-wake")).toHaveLength(1);
    await c.waitFor(() => c.messages.some((m) => m.t === "task_status" && m.task?.status === "done"), "the done chip");

    // A permission ask is a pushed event too — but the model is SPEAKING, so it must wait.
    gem.busy = true;
    host.ports.send({
      t: "permission",
      reqId: "perm_1",
      question: "Hermes wants to run the tests. Allow it?",
      options: [
        { id: "allow", label: "Allow", kind: "allow_once" },
        { id: "deny", label: "Deny", kind: "reject_once" },
      ],
    });
    await c.waitFor(() => c.messages.some((m) => m.t === "permission" && m.id === "perm_1"), "the permission ask");
    expect(gem.texts).toHaveLength(1); // queued, not spoken over the model
    expect(gem.wakeCalls).toBe(1);

    // answer_agent is valid only while that ask is pending.
    const tooEarly = gem.callTool("answer_agent", { request_id: "nope", decision: "approve" }).response;
    expect(tooEarly).toMatchObject({ ok: false, reason: "no_pending_request" });
    const answered = gem.callTool("answer_agent", { request_id: "perm_1", decision: "approve" }).response;
    expect(answered.ok).toBe(true);
    expect(host.answeredPerm).toContain("perm_1:allow");

    gem.busy = false;
    gem.turnComplete(); // now idle -> the queued event is delivered
    await waitUntil(() => gem.texts.length === 2, "the queued needs_input");
    expect(gem.wakeCalls).toBe(1); // already live: one wake, not one per event
    expect(gem.texts[1]).toContain("needs an answer");
  });
});

// ── persistence: clean transcript + tool cards, never ledger headers ─────

describe("chat history", () => {
  test("clean transcript and tool cards are persisted; ledger headers never are", async () => {
    const c = await live("rt-history");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false;
    host.script = [
      {
        type: "acp_tool_call",
        call: { id: "tool_1", title: "Editing app.tsx", kind: "edit", status: "completed", content: [], locations: [] },
      },
      { type: "text_delta", text: "The header is now OpenLive Pro." },
    ];

    const { draftId, version } = await speakConfirmation(c, gem);
    gem.callTool("confirm_instruction", { draft_id: draftId, version });
    // The turn emits its script before it parks on the hung promise; wait for the park.
    await waitUntil(() => host.pendingRun !== null, "the coding agent to be running");
    host.finish();
    await waitUntil(
      () => listMessages("rt-history").some((m) => m.role === "assistant" && m.content.some((b) => b.type === "acp_tool")),
      "the persisted tool card",
    );
    const saved = listMessages("rt-history");
    expect(saved.some((m) => m.role === "user" && m.content.some((b) => b.type === "text" && b.text.includes("হ্যাঁ")))).toBe(true);

    // The model's own speech is a clean assistant turn.
    gem.emit("outputTranscription", { text: "হেডারটা বদলে দিয়েছি।" });
    gem.turnComplete();
    await waitUntil(
      () => listMessages("rt-history").some((m) => m.role === "assistant" && m.content.some((b) => b.type === "text" && b.text.includes("হেডারটা"))),
      "the assistant transcript",
    );

    // Even if the model echoes an internal header into its speech, it is not stored.
    expect(cleanForChat("[STATE_SYNC resume=ok checkpoint_seq=3]\nreal words only")).toBe("real words only");
    gem.emit("outputTranscription", { text: "[STATE_SYNC resume=ok checkpoint_seq=3]\nreal words only" });
    gem.turnComplete();
    await waitUntil(
      () => listMessages("rt-history").some((m) => m.content.some((b) => b.type === "text" && b.text === "real words only")),
      "the filtered assistant turn",
    );
    expect(JSON.stringify(listMessages("rt-history"))).not.toMatch(/\[(?:LEDGER_COLD_START|STATE_SYNC)\]/);

    // A reconnect cold-seeds the model with the ledger header — and persists none of it.
    await c.close();
    const c2 = await live("rt-history");
    const gem2 = latestClient();
    expect(JSON.stringify(gem2.coldHistory)).toContain("[LEDGER_COLD_START]");
    expect(JSON.stringify(listMessages("rt-history"))).not.toMatch(/\[(?:LEDGER_COLD_START|STATE_SYNC)\]/);
    // The cold header is what a resumed model is told not to re-dispatch; the ledger has one row.
    expect(dispatches("rt-history")).toHaveLength(1);
    await c2.close();
  });
});

// ── the non-awaiting handlers + the task outliving its socket ────────────

describe("every Live handler returns from memory", () => {
  test("all six answer inside the budget, even with a hung coding agent", async () => {
    const c = await live("rt-tools");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false;

    const t0 = performance.now();
    const proposed = gem.callTool("propose_instruction", { summary_bn: SUMMARY, english_spec: SPEC, target: "new_task" });
    expect(proposed.response.ok).toBe(true);
    const draftId = String(proposed.response.draft_id);
    const version = Number(proposed.response.version);
    const armed = gem.callTool("request_confirmation", { draft_id: draftId, version, readback_bn: READBACK });
    expect(armed.response.ok).toBe(true);
    const status = gem.callTool("get_status", { scope: "summary" });
    expect(status.response.ok).toBe(true);
    const cancelNone = gem.callTool("cancel_task", { reason: "nothing is running" });
    expect(cancelNone.response).toMatchObject({ ok: true, cancelled: false });
    const answerNone = gem.callTool("answer_agent", { request_id: "none", decision: "deny" });
    expect(answerNone.response.ok).toBe(false);
    const nominateNone = gem.callTool("confirm_instruction", { draft_id: "nope", version: 9 });
    expect(nominateNone.response.ok).toBe(false);
    const spent = performance.now() - t0;

    // Six sequential handlers, a loopback round trip each, all far inside one model turn.
    expect(spent).toBeLessThan(TOOL_BUDGET_MS * 6);
    for (const [name, result] of [["propose_instruction", proposed], ["request_confirmation", armed], ["get_status", status]] as const) {
      expect(result.spentMs, `${name} handler must return from memory`).toBeLessThan(TOOL_BUDGET_MS);
    }
    await sleep(20);
    expect(host.pendingRun).toBeNull(); // nothing above dispatched, so nothing hangs
  });

  test("the hosted MCP gets user_guidance and report_progress from AcpHost's tool port", async () => {
    const c = await live("rt-mcp");
    const gem = latestClient();
    const host = latestHost();
    host.autoResolve = false;

    // This object is what `AcpHost.ensureMcp` publishes through the one `serveMcp` server.
    const tools = host.ports.tools();
    expect(tools.list.map((t: any) => t.name).sort()).toEqual(["report_progress", "user_guidance"]);
    const ctx = { signal: new AbortController().signal, context: null } as never;

    // `user_guidance` drains through the host (never the agent) — empty before any task.
    const noTask = await tools.resolve("user_guidance")!.execute({}, ctx);
    expect((noTask.content[0] as any).text).toBe("NO_NEW_GUIDANCE");

    const { draftId, version } = await speakConfirmation(c, gem);
    gem.callTool("confirm_instruction", { draft_id: draftId, version });
    await waitUntil(() => host.currentTask() !== null, "the running task");
    const drained = await tools.resolve("user_guidance")!.execute({}, ctx);
    expect((drained.content[0] as any).text).toBe("NO_NEW_GUIDANCE");
    expect(host.guidanceDrains.at(-1)).toBe(host.currentTask()!.taskId);

    // `report_progress` writes the step into the ledger, and `get_status` reads it back from memory.
    await tools.resolve("report_progress")!.execute({ status: "editing the header component", detail: "renamed it" }, ctx);
    const status = gem.callTool("get_status", { scope: "summary" }).response;
    expect(String(status.digest)).toContain("editing the header component");

    host.finish();
    await waitUntil(() => host.calls.endTask >= 1, "the task to settle");
  });

  test("a task outlives the socket that dispatched it, then cleans up", async () => {
    const c1 = await live("rt-orphan");
    const gem1 = latestClient();
    const host1 = latestHost();
    host1.autoResolve = false;
    const { draftId, version } = await speakConfirmation(c1, gem1);
    gem1.callTool("confirm_instruction", { draft_id: draftId, version });
    expect(host1.calls.startTask).toBe(1);

    // The renderer reloads mid-task: the voice socket dies, the coding task does not.
    await c1.close();
    await waitUntil(() => gem1.closed, "the voice socket to close");
    expect(host1.disposed).toBe(false);

    // The reconnect watches the ledger instead of spawning a second agent on the same project.
    const c2 = await live("rt-orphan");
    const gem2 = latestClient();
    const host2 = latestHost();
    await waitUntil(() => c2.messages.some((m) => m.t === "task_status"), "the reconnected status");
    expect(host2.calls.restore).toBe(0);
    const staged = gem2.callTool("propose_instruction", { summary_bn: SUMMARY, english_spec: SPEC, target: "new_task" }).response;
    expect(staged.ok).toBe(true); // staging is always allowed
    const nominate = gem2.callTool("confirm_instruction", { draft_id: String(staged.draft_id), version: Number(staged.version) }).response;
    expect(nominate).toMatchObject({ dispatched: false, reason: "previous_task_running" });

    // The orphaned task ends: the reconnecting socket binds its agent and the orphan is disposed.
    host1.finish();
    await waitUntil(() => host1.disposed && host2.calls.restore === 1, "the orphan hand-off");
    await c2.close();
  });
});
