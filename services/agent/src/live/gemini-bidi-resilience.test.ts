// Stage 3 pass condition: the five faults that drop a real Gemini Live call, each replayed
// against a local `ws` mock server (no network, no key, no real API).
//   1. a 1008 inactivity close sleeps the call instead of failing it, and wake() drains pre-roll
//   2. a mid-turn 1006 reconnects with the last *resumable* handle, never the one sent mid-work
//   3. a goAway waits for the current turn to finish before the sockets are swapped
//   4. a 1007 stale handle is cleared once and the retry is cold (never four tries with the same handle)
//   5. a resumed socket whose promptTokenCount collapses triggers the cold-header reseed
import { afterEach, expect, test } from "vitest";
import { WebSocketServer, type WebSocket as WsSocket } from "ws";
import { GEMINI_TOOL_NAMES } from "@openlive/shared";
import { GeminiBidiClient, MIC_COALESCE_BYTES, type GeminiBidiConfig } from "./gemini-bidi.js";

interface MockConn {
  index: number;
  /** The `setup` object the client sent on this socket. */
  setup: Record<string, any>;
  /** Every non-setup frame the client sent on this socket, in order. */
  frames: Record<string, any>[];
  closed: { code: number; reason: string } | null;
  send: (frame: unknown) => void;
  close: (code: number, reason?: string) => void;
  terminate: () => void;
  /** Concatenated PCM16 the client streamed on this socket. */
  audio: () => Buffer;
  clientContent: () => Record<string, any>[];
  realtimeInput: () => Record<string, any>[];
}

interface Mock {
  url: string;
  conns: MockConn[];
  script: (index: number, handler: (conn: MockConn) => void) => void;
  stop: () => Promise<void>;
}

const raws = (data: unknown): string => {
  if (Buffer.isBuffer(data)) return data.toString("utf8");
  if (Array.isArray(data)) return Buffer.concat(data as Buffer[]).toString("utf8");
  return Buffer.from(data as ArrayBuffer).toString("utf8");
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitUntil(pred: () => boolean, what: string, timeoutMs = 4000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pred()) return;
    await sleep(10);
  }
  throw new Error(`timed out waiting for ${what}`);
}

/** A local BidiGenerateContent stand-in: every socket is scripted by connection index. */
async function startMock(): Promise<Mock> {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise<void>((resolve) => wss.once("listening", () => resolve()));
  const port = (wss.address() as { port: number }).port;
  const conns: MockConn[] = [];
  const scripts = new Map<number, (conn: MockConn) => void>();

  wss.on("connection", (socket: WsSocket) => {
    const index = conns.length;
    let sawSetup = false;
    const frames: Record<string, any>[] = [];
    const listeners: Array<(frame: Record<string, any>) => void> = [];
    let closed: { code: number; reason: string } | null = null;

    const conn: MockConn = {
      index,
      setup: {},
      frames,
      get closed() { return closed; },
      send: (frame) => { if (socket.readyState === socket.OPEN) socket.send(JSON.stringify(frame)); },
      close: (code, reason) => socket.close(code, reason),
      terminate: () => socket.terminate(),
      audio: () => Buffer.concat(frames.flatMap((f) => (f.realtimeInput?.audio?.data
        ? [Buffer.from(String(f.realtimeInput.audio.data), "base64")]
        : []))),
      clientContent: () => frames.filter((f) => f.clientContent),
      realtimeInput: () => frames.filter((f) => f.realtimeInput),
    };

    socket.on("message", (data) => {
      let frame: Record<string, any>;
      try { frame = JSON.parse(raws(data)) as Record<string, any>; } catch { return; }
      if (!sawSetup) {
        sawSetup = true;
        conn.setup = (frame.setup ?? {}) as Record<string, any>;
        conns.push(conn);
        scripts.get(index)?.(conn);
        return;
      }
      frames.push(frame);
      for (const fn of listeners) fn(frame);
    });
    socket.on("close", (code, reason) => { closed = { code, reason: reason.toString() }; });
  });

  return {
    url: `ws://127.0.0.1:${port}`,
    conns,
    script: (index, handler) => { scripts.set(index, handler); },
    stop: async () => {
      for (const socket of wss.clients) { try { socket.terminate(); } catch { /* already gone */ } }
      await new Promise<void>((resolve) => wss.close(() => resolve()));
    },
  };
}

const cleanups: Array<() => Promise<void> | void> = [];
afterEach(async () => {
  for (const fn of cleanups.splice(0).reverse()) await fn();
});

async function trackedClient(mock: Mock, over: Partial<GeminiBidiConfig> = {}): Promise<GeminiBidiClient> {
  const client = new GeminiBidiClient({
    apiKey: "test-not-a-real-key",
    model: "gemini-3.8-live",
    systemInstruction: "ORCHESTRATOR PROTOCOL",
    wsUrl: mock.url,
    setupTimeoutMs: 4000,
    ...over,
  });
  // Keep a thrown 'error' event from becoming an unhandled exception in the test run.
  client.on("error", () => {});
  cleanups.push(async () => { await client.close(); });
  return client;
}

const audioPart = (bytes = 480) => ({ inlineData: { mimeType: "audio/pcm;rate=24000", data: Buffer.alloc(bytes, 7).toString("base64") } });

test("a 1008 inactivity close sleeps the call and wake() drains the pre-roll onto the new socket", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => conn.send({ setupComplete: {} }));
  const client = await trackedClient(mock);
  const phases: string[] = [];
  client.on("phase", (e: { phase: string }) => phases.push(e.phase));

  await client.connect();
  expect(client.currentPhase()).toBe("live");

  // The documented live setup message, as the server must be told it.
  const first = mock.conns[0]!;
  expect(first.setup.model).toBe("models/gemini-3.8-live");
  expect(first.setup.generationConfig).toMatchObject({ responseModalities: ["AUDIO"] });
  expect(first.setup.inputAudioTranscription).toEqual({});
  expect(first.setup.outputAudioTranscription).toEqual({});
  expect(first.setup.contextWindowCompression).toEqual({ slidingWindow: {} });
  expect(first.setup.sessionResumption).toEqual({});
  expect(first.setup.historyConfig).toEqual({ initialHistoryInClientContent: true });
  expect((first.setup.tools?.[0]?.functionDeclarations ?? []).map((d: { name: string }) => d.name)).toEqual([...GEMINI_TOOL_NAMES]);
  // Cold seeding is a clientContent, and it goes out before any audio can.
  await waitUntil(() => first.clientContent().length === 1, "the cold seed");
  expect(first.realtimeInput()).toEqual([]);

  // The fault the current plugin fails the whole call on: 1008 inactivity is survivable.
  first.close(1008, "inactivity close");
  await waitUntil(() => client.currentPhase() === "sleeping", "sleeping");
  expect(phases).toContain("sleeping");

  // Audio captured while asleep is pre-roll, not sent on the dead socket.
  const early = Buffer.alloc(MIC_COALESCE_BYTES, 0x11);
  const late = Buffer.alloc(MIC_COALESCE_BYTES, 0x22);
  client.pushMicPcm16(early);
  client.pushMicPcm16(late);
  expect(first.realtimeInput()).toEqual([]);

  mock.script(1, (conn) => conn.send({ setupComplete: {} }));
  const wakeRing = Buffer.alloc(1280, 0x33);
  await client.wake([wakeRing]);
  await waitUntil(() => mock.conns.length === 2 && client.currentPhase() === "live", "the wake");
  const woken = mock.conns[1]!;
  await waitUntil(() => woken.audio().length === early.length + late.length + wakeRing.length, "the drained pre-roll");

  // Nothing was lost across the sleep, and every frame stays inside the 60-100 ms window.
  const drained = woken.audio();
  for (const part of [early, late, wakeRing]) expect(drained.includes(part)).toBe(true);
  for (const frame of woken.realtimeInput().filter((f) => f.realtimeInput?.audio)) {
    const bytes = Buffer.from(String(frame.realtimeInput.audio.data), "base64").length;
    expect(bytes).toBeGreaterThan(0);
    expect(bytes % 2).toBe(0);
    expect(bytes).toBeLessThanOrEqual(MIC_COALESCE_BYTES);
    expect(frame.realtimeInput.audio.mimeType).toBe("audio/pcm;rate=16000");
  }
  expect(phases.at(-1)).toBe("live");
});

test("a mid-turn 1006 reconnects with the last resumable handle, never the one sent during the work", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  const part = audioPart();
  mock.script(0, (conn) => {
    conn.send({ setupComplete: {} });
    conn.send({ sessionResumptionUpdate: { resumable: true, newHandle: "GOOD-HANDLE", lastConsumedClientMessageIndex: 17 } });
    // During generation or a tool call the server reports resumable:false and points BEFORE the work.
    conn.send({ sessionResumptionUpdate: { resumable: false, newHandle: "MID-TURN-HANDLE" } });
    conn.send({ serverContent: { modelTurn: { parts: [part] } } });
    // A transport drop (1006) has no close frame to send: terminate the TCP socket mid-turn.
    setTimeout(() => conn.terminate(), 40);
  });
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock);
  const heard: string[] = [];
  client.on("audio", (pcm: Buffer) => heard.push(pcm.toString("base64")));

  await client.connect();
  await waitUntil(() => mock.conns.length === 2, "the reconnect", 6000);
  const resumed = mock.conns[1]!;
  expect(resumed.setup.sessionResumption).toEqual({ handle: "GOOD-HANDLE" });
  expect(resumed.setup.historyConfig).toBeUndefined(); // a resume is not a cold start
  await waitUntil(() => client.currentPhase() === "live", "live after the resume");

  expect(client.currentResumeHandle()).toBe("GOOD-HANDLE");
  expect(client.currentCheckpointSeq()).toBe(17);
  expect(heard).toEqual([part.inlineData.data]);
});

test("a goAway waits for turnComplete before the sockets are swapped", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => {
    conn.send({ setupComplete: {} });
    conn.send({ serverContent: { modelTurn: { parts: [audioPart()] } } });
    conn.send({ goAway: { timeLeft: "30s" } });
  });
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock);
  const away = new Promise<{ timeLeft?: string }>((resolve) => client.once("goAway", resolve));
  await client.connect();
  expect(await away).toMatchObject({ timeLeft: "30s" });
  await sleep(300);

  // goAway is a deadline, not an order to cut speech: a live turn keeps the old socket.
  expect(mock.conns.length).toBe(1);
  expect(mock.conns[0]!.closed).toBeNull();
  expect(client.currentPhase()).toBe("live");

  // turnComplete is the quiet boundary.
  mock.conns[0]!.send({ serverContent: { turnComplete: true } });
  await waitUntil(() => mock.conns.length === 2, "the rotation");
  await waitUntil(() => client.currentPhase() === "live", "live after rotating");
  expect(mock.conns[0]!.closed?.code).toBe(1000);
  // The replacement socket is a real cold start, so it reseeds instead of losing the call.
  expect(mock.conns[1]!.setup.sessionResumption).toEqual({});
  expect(mock.conns[1]!.setup.historyConfig).toEqual({ initialHistoryInClientContent: true });
  await waitUntil(() => mock.conns[1]!.clientContent().length === 1, "the cold seed on the new socket");
});

test("a 1007 stale handle is cleared once and the retry is cold", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  const coldHeader = () => [
    { role: "user" as const, parts: [{ text: "[LEDGER_COLD_START]\nDISPATCHED_TASKS (NEVER_RE_DISPATCH):\n- task_7 status=running\n[END]" }] },
  ];
  // (a) the socket is accepted with the handle, then the server rejects what the handle points at.
  mock.script(0, (conn) => {
    conn.send({ setupComplete: {} });
    setTimeout(() => conn.close(1007, "resumption handle is stale"), 20);
  });
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock, { resumeHandle: "STALE-HANDLE", getColdHistory: coldHeader });
  await client.connect();
  expect(mock.conns[0]!.setup.sessionResumption).toEqual({ handle: "STALE-HANDLE" });
  expect(mock.conns[0]!.setup.historyConfig).toBeUndefined();

  await waitUntil(() => mock.conns.length === 2, "the cold retry");
  const retried = mock.conns[1]!;
  expect(retried.setup.sessionResumption).toEqual({});
  expect(retried.setup.historyConfig).toEqual({ initialHistoryInClientContent: true });
  await waitUntil(() => retried.clientContent().length === 1, "the cold header");
  expect(retried.clientContent()[0]!.clientContent.turns[0].parts[0].text).toContain("[LEDGER_COLD_START]");
  expect(client.currentResumeHandle()).toBe("");
  // Exactly one retry: the stale handle is never presented a second time (the plugin retried it four times).
  expect(mock.conns.filter((c) => c.setup.sessionResumption?.handle === "STALE-HANDLE")).toHaveLength(1);
  await waitUntil(() => client.currentPhase() === "live", "live after the cold retry");

  // (b) a handle rejected during setup itself is cleared just the same: connect() falls back cold
  // once, instead of presenting the same rejected handle for another attempt.
  mock.script(2, (conn) => conn.close(1007, "resumption handle is stale"));
  mock.script(3, (conn) => conn.send({ setupComplete: {} }));
  const rejected = await trackedClient(mock, { resumeHandle: "STALE-HANDLE-2" });
  await rejected.connect();
  expect(mock.conns[2]!.setup.sessionResumption).toEqual({ handle: "STALE-HANDLE-2" });
  expect(mock.conns[3]!.setup.sessionResumption).toEqual({});
  expect(mock.conns[3]!.setup.historyConfig).toEqual({ initialHistoryInClientContent: true });
  expect(rejected.currentResumeHandle()).toBe("");
  expect(rejected.currentPhase()).toBe("live");
  expect(mock.conns.filter((c) => c.setup.sessionResumption?.handle === "STALE-HANDLE-2")).toHaveLength(1);
});

test("the token-count canary turns a silent context loss into a cold header", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => {
    conn.send({ setupComplete: {} });
    conn.send({ usageMetadata: { promptTokenCount: 1500 } });
    conn.send({ sessionResumptionUpdate: { resumable: true, newHandle: "HANDLE-BEFORE-LOSS" } });
    setTimeout(() => conn.terminate(), 30);
  });
  // The resumed socket says setupComplete, but the prompt token count collapsed: resumption kept no context.
  mock.script(1, (conn) => {
    conn.send({ setupComplete: {} });
    conn.send({ usageMetadata: { promptTokenCount: 200 } });
    setTimeout(() => conn.terminate(), 30);
  });
  mock.script(2, (conn) => conn.send({ setupComplete: {} }));

  const coldHistory = () => [{ role: "user" as const, parts: [{ text: "[LEDGER_COLD_START]\ncanary reseed\n[END]" }] }];
  const client = await trackedClient(mock, { getColdHistory: coldHistory });
  const lost: Array<{ before: number; after: number }> = [];
  client.on("resumptionContextLost", (e: { before: number; after: number }) => lost.push(e));

  await client.connect();
  await waitUntil(() => mock.conns.length === 3, "the cold reseed", 8000);
  expect(mock.conns[1]!.setup.sessionResumption).toEqual({ handle: "HANDLE-BEFORE-LOSS" });
  expect(lost).toMatchObject([{ before: 1500, after: 200 }]);

  const reseeded = mock.conns[2]!;
  expect(reseeded.setup.sessionResumption).toEqual({});
  expect(reseeded.setup.historyConfig).toEqual({ initialHistoryInClientContent: true });
  await waitUntil(() => reseeded.clientContent().length === 1, "the cold header after the canary");
  expect(reseeded.clientContent()[0]!.clientContent.turns[0].parts[0].text).toContain("[LEDGER_COLD_START]");
  expect(client.currentResumeHandle()).toBe("");
  await waitUntil(() => client.currentPhase() === "live", "live after the cold reseed");
});

test("mic audio is coalesced into whole 60-100 ms frames, and the tail survives a socket swap", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => conn.send({ setupComplete: {} }));
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock);
  await client.connect();
  const first = mock.conns[0]!;
  const sizes = () => first.realtimeInput()
    .flatMap((f) => (f.realtimeInput?.audio?.data ? [Buffer.from(String(f.realtimeInput.audio.data), "base64").length] : []));
  /** Frames cross the real socket, so wait for the mock to have received them. */
  const expectSizes = async (want: number[]) => {
    await waitUntil(() => sizes().length >= want.length, `frames ${JSON.stringify(want)}`);
    expect(sizes()).toEqual(want);
  };

  // A 60-100 ms envelope at 16 kHz PCM16LE: MIC_COALESCE_BYTES is 80 ms of it.
  expect(MIC_COALESCE_BYTES / 32).toBeGreaterThanOrEqual(60);
  expect(MIC_COALESCE_BYTES / 32).toBeLessThanOrEqual(100);

  // 10 ms chunks in (320 B each): eight of them make one 80 ms frame, not eight runt frames.
  for (let i = 0; i < 7; i++) client.pushMicPcm16(Buffer.alloc(320, i + 1));
  await sleep(150); // longer than the flush timer: still nothing whole, so still no runt
  expect(sizes()).toEqual([]);
  client.pushMicPcm16(Buffer.alloc(320, 8));
  await expectSizes([MIC_COALESCE_BYTES]);

  // A sub-frame remainder waits for the next chunk instead of going out as a runt.
  client.pushMicPcm16(Buffer.alloc(1000, 9)); // below one frame: held
  await sleep(150);
  await expectSizes([MIC_COALESCE_BYTES]);
  client.pushMicPcm16(Buffer.alloc(2000, 9)); // 3 kB -> one whole frame, 440 B held
  await expectSizes([MIC_COALESCE_BYTES, MIC_COALESCE_BYTES]);
  // Every frame that went out is exactly one coalesced window, never a fragment of one.
  expect(sizes()).toEqual([MIC_COALESCE_BYTES, MIC_COALESCE_BYTES]);

  // Sleeping drops the stale remainder instead of sending it late (the renderer's ring reseeds on wake).
  client.pushMicPcm16(Buffer.alloc(640, 4));
  await client.sleep("test_sleep");
  await sleep(150);
  await expectSizes([MIC_COALESCE_BYTES, MIC_COALESCE_BYTES]);
  expect(client.currentPhase()).toBe("sleeping");

  await client.wake();
  await waitUntil(() => client.currentPhase() === "live" && mock.conns.length === 2, "the woken socket");
  await sleep(120);
  expect(mock.conns[1]!.realtimeInput().flatMap((f) => (f.realtimeInput?.audio?.data ? [1] : []))).toEqual([]);
});

test("a goAway rotation forces the sub-frame remainder out instead of dropping the last words", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => conn.send({ setupComplete: {} }));
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock);
  await client.connect();
  const first = mock.conns[0]!;
  const sizes = () => first.realtimeInput()
    .flatMap((f) => (f.realtimeInput?.audio?.data ? [Buffer.from(String(f.realtimeInput.audio.data), "base64").length] : []));

  // A sub-frame is in flight when the goAway arrives at an already-idle boundary: that is a quiet
  // boundary too, so the tail must go out on the socket being replaced rather than be dropped.
  client.pushMicPcm16(Buffer.alloc(700, 5));
  await sleep(100);
  expect(sizes()).toEqual([]);
  first.send({ goAway: { timeLeft: "30s" } });
  await waitUntil(() => mock.conns.length === 2, "the rotation");
  await waitUntil(() => sizes().length === 1, "the forced tail");
  expect(sizes()).toEqual([700]);
});

test("every serverContent event is surfaced, and a tool response goes only to the socket that asked", async () => {
  const mock = await startMock();
  cleanups.push(mock.stop);
  mock.script(0, (conn) => {
    conn.send({ setupComplete: {} });
    conn.send({ toolCall: { functionCalls: [{ id: "call_a", name: "get_status", args: { scope: "summary" } }] } });
    conn.send({ serverContent: { modelTurn: { parts: [audioPart()] } } });
    conn.send({ serverContent: { outputTranscription: { text: "ঠিক আছে", finished: false } } });
    conn.send({ serverContent: { inputTranscription: { text: "হ্যাঁ পাঠাও", finished: true } } });
    conn.send({ serverContent: { interrupted: true } });
    conn.send({ serverContent: { generationComplete: true } });
    conn.send({ serverContent: { turnComplete: true } });
  });
  mock.script(1, (conn) => conn.send({ setupComplete: {} }));

  const client = await trackedClient(mock);
  const seen: string[] = [];
  for (const name of ["interrupted", "generationComplete", "turnComplete", "inputTranscription", "outputTranscription", "audio"]) {
    client.on(name, (payload: unknown) => seen.push(`${name}:${JSON.stringify(payload)}`));
  }
  const toolEvents: Array<{ calls: Array<{ id: string; name: string }>; socketSeq: number }> = [];
  client.on("toolCall", (e: { calls: Array<{ id: string; name: string }>; socketSeq: number }) => toolEvents.push(e));

  await client.connect();
  const sock = mock.conns[0]!;
  // The scripted burst fires as the socket opens; make sure the last of it has landed.
  sock.send({ serverContent: { turnComplete: true } });
  await waitUntil(() => seen.filter((s) => s.startsWith("turnComplete:")).length >= 2, "both turnCompletes");

  expect(seen.some((s) => s.startsWith("interrupted:"))).toBe(true);
  expect(seen.some((s) => s.startsWith("generationComplete:"))).toBe(true);
  expect(seen.some((s) => s.startsWith("audio:"))).toBe(true);
  expect(seen.some((s) => s.startsWith('inputTranscription:{"text":"হ্যাঁ পাঠাও"'))).toBe(true);
  expect(seen.some((s) => s.startsWith('outputTranscription:{"text":"ঠিক আছে"'))).toBe(true);
  expect(toolEvents.map((e) => e.calls.map((c) => c.name))).toEqual([["get_status"]]);
  expect(client.isTurnBusy()).toBe(false);

  const responses = () => sock.frames.filter((f) => f.toolResponse);
  const currentSeq = toolEvents[0]!.socketSeq;
  // The call this socket issued is answered on this socket.
  expect(client.sendToolResponses([{ id: "call_a", name: "get_status", response: { step: "editing" } }], currentSeq)).toBe(true);
  await waitUntil(() => responses().length === 1, "the tool response");
  expect(responses()[0]!.toolResponse.functionResponses).toMatchObject([{ id: "call_a", name: "get_status" }]);

  // A call issued by a previous socket is never answered on this one (the plugin's cross-socket bug).
  sock.send({ toolCall: { functionCalls: [{ id: "call_b", name: "propose_instruction", args: {} }] } });
  await waitUntil(() => toolEvents.length === 2, "the second toolCall");
  expect(client.sendToolResponses([{ id: "call_b", name: "propose_instruction", response: {} }], currentSeq - 1)).toBe(false);
  expect(responses()).toHaveLength(1);
  // An explicitly tagged correct seq still sends it, and a call with no tag is sent untagged.
  expect(client.sendToolResponses([{ id: "call_b", name: "propose_instruction", response: {} }], currentSeq)).toBe(true);
  sock.send({ toolCall: { functionCalls: [{ id: "call_c", name: "get_status", args: {} }] } });
  await waitUntil(() => toolEvents.length === 3, "the third toolCall");
  expect(client.sendToolResponses([{ id: "call_c", name: "get_status", response: {} }])).toBe(true);
  await waitUntil(() => responses().length === 3, "the remaining tool responses");
});
