// AcpHost is the ACP half of a conversation: binding (with the epoch guard), the
// one hosted MCP server, permission/elicitation asks, and ordered steering. It was
// EXTRACTED from LiveSession, so the first thing this file pins is that the parts
// that must not change did not: bindEpoch cancellation, the MCP signal, and the
// single-settle-path asks.
//
// Then the two things the extraction ADDS, which have no equivalent in the old
// session and therefore need their own proofs:
//
//   * ordered GuidanceItem[] — applied #1 then #2, in the order spoken, NEVER
//     concatenated into one string (that is what makes "do X" + "not X" read as a
//     single instruction);
//   * the <voice_summary> extractor — at most three sentences, no code and no
//     paths, falling back to the last three sentences when nothing survives.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { WebSocket } from "ws";
import { afterAll, describe, expect, test } from "vitest";
import type { GuidanceItem, LiveServerMsg } from "@openlive/shared";
import type { Approve, Tool } from "../capabilities/types.js";
import { ToolSet } from "../capabilities/dispatch.js";

// The db resolves its data dir at import time, so this has to be set first.
const dir = mkdtempSync(join(tmpdir(), "ol-acp-host-"));
process.env.OPENLIVE_HOME = dir;
const { AcpHost, ALLOW_OR_DENY, continuationPrompt, hideHosted, sentences, steeringOf, voiceSummary, voiceSummaryBlock, acpSessionKey } = await import("./acp-host.ts");
const { Ledger } = await import("./ledger.ts");

afterAll(() => {
  delete process.env.OPENLIVE_HOME;
  rmSync(dir, { recursive: true, force: true });
});

// ── a host with no ACP child process ────────────────────────────────────────
// Binding spawns a real adapter, so these tests drive the parts that do not:
// the signal, the asks, the steering dispatch and the summary. That is deliberate
// — they are the pieces the extraction moved, and each is provable in isolation.

const allowAll: Approve = async () => ({});

/** A tool that only records that it ran, plus what its ctx signal was. */
function spyTool(seen: AbortSignal[]): ToolSet {
  const tool: Tool<{ text?: string }, { ok: boolean }> = {
    name: "spy",
    description: "Records that it ran and the signal it was given.",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    readOnly: true,
    async execute(_args, ctx) {
      seen.push(ctx.signal);
      return { content: [{ type: "text", text: "ran" }], details: { ok: true } };
    },
  };
  return new ToolSet([tool]);
}

interface Harness {
  host: AcpHost;
  sent: LiveServerMsg[];
  ledger: Ledger;
  signals: AbortSignal[];
  /** What the latest MCP ctx resolved to. */
  ctxSignal: () => AbortSignal | null;
  setTurn: (signal: AbortSignal | null) => void;
  turnState: { active: boolean };
}

/** A host over a throwaway ledger, its ports fully faked. */
function harness(chatId: string): Harness {
  const sent: LiveServerMsg[] = [];
  const signals: AbortSignal[] = [];
  const ledger = new Ledger(chatId, dir);
  let turn: AbortSignal | null = null;
  const turnState = { active: false };
  const tools = spyTool(signals);
  const host = new AcpHost({
    chatId,
    closed: () => false,
    send: (m) => sent.push(m),
    session: () => ({}),
    tools: () => tools,
    approve: allowAll,
    ledger,
    turnActive: () => turnState.active,
    turnSignal: () => turn,
    replyTurn: () => 7,
    seed: () => [],
    pendingCut: () => null,
    onMeta: () => {},
    onReplay: async () => {},
    onReplayExpected: () => {},
    onFolderMoved: () => {},
    onStarted: () => {},
    onReady: () => {},
    onStartError: () => {},
    onResumed: () => {},
  });
  // The hosted server resolves its ctx per request; reach it the same way MCP does.
  const ctxSignal = () => {
    const mcp = (host as unknown as { mcp: { wire: unknown } | null }).mcp;
    void mcp;
    return null;
  };
  return { host, sent, ledger, signals, ctxSignal, setTurn: (s) => { turn = s; turnState.active = !!s; }, turnState };
}

const item = (guidanceId: string, text: string, taskId = "task_1"): GuidanceItem => ({
  guidanceId,
  seq: 0,
  taskId,
  idempotencyKey: `key-${guidanceId}`,
  kind: "steer_add",
  text,
  status: "pending",
  createdAt: 1,
});

// ── the signal fix ─────────────────────────────────────────────────────────

describe("activeTaskSignal: the tools outlive a voice turn", () => {
  test("with no turn and no task, a call is refused rather than run on a dead signal", () => {
    const { host } = harness("sig-none");
    expect(host.activeTaskSignal().aborted).toBe(true);
  });

  test("inside a voice turn it is exactly that turn's signal, as /live bound it before", () => {
    const h = harness("sig-turn");
    const turn = new AbortController();
    h.setTurn(turn.signal);
    // Identical to the pre-extraction `this.ac?.signal`: a live turn's tools run.
    expect(h.host.activeTaskSignal().aborted).toBe(false);
    turn.abort();
    expect(h.host.activeTaskSignal().aborted).toBe(true);
  });

  test("a barge-in on the VOICE turn does not abort a running task's tools", () => {
    const h = harness("sig-task");
    const turn = new AbortController();
    const task = new AbortController();
    h.setTurn(turn.signal);
    h.host.startTask("task_1", "change the header", task.signal);
    // The turn the user spoke into is cut — the coding task is still running.
    turn.abort();
    expect(h.host.activeTaskSignal().aborted).toBe(false);
    expect(h.host.taskRunning).toBe(true);
    // Only cancelling the TASK stops it.
    task.abort();
    expect(h.host.activeTaskSignal().aborted).toBe(true);
    expect(h.host.taskRunning).toBe(false);
  });

  test("the task's signal is what the hosted tools get, so user_guidance lives for the task", () => {
    const h = harness("sig-hosted");
    const task = new AbortController();
    h.host.startTask("task_1", "spec", task.signal);
    const signal = h.host.activeTaskSignal();
    expect(signal).toBe(h.host.currentTask()!.signal);
    expect(signal.aborted).toBe(false);
    h.host.endTask();
    expect(h.host.activeTaskSignal().aborted).toBe(true);
  });

  test("starting a task supersedes the previous one's signal instead of leaking it", () => {
    const h = harness("sig-supersede");
    const first = new AbortController();
    h.host.startTask("task_1", "one", first.signal);
    const before = h.host.currentTask()!.signal;
    h.host.startTask("task_2", "two", new AbortController().signal);
    // The old task's signal is aborted, so a straggling poll cannot answer for it.
    expect(before.aborted).toBe(true);
    expect(h.host.currentTask()!.taskId).toBe("task_2");
    expect(h.host.activeTaskSignal().aborted).toBe(false);
  });
});

// ── the asks ───────────────────────────────────────────────────────────────

describe("permission and elicitation asks keep one settle path", () => {
  test("a permission ask carries the turn's number and settles to the chosen option", async () => {
    const h = harness("ask-perm");
    h.setTurn(new AbortController().signal);
    const pending = h.host.askPermission("Codex wants permission: OpenLive's open url. Allow it?", ALLOW_OR_DENY, "ask");
    const ask = h.sent.find((m) => m.t === "permission")!;
    expect(ask).toMatchObject({ question: "Codex wants permission: OpenLive's open url. Allow it?", toolCallId: "ask", turn: 7 });
    expect(ask.options.map((o) => o.kind)).toEqual(["allow_once", "reject_once"]);
    h.host.answerPermission(ask.reqId, "allow");
    expect(await pending).toBe("allow");
    // One settle tells the client the ask is resolved; a second answer is a no-op.
    expect(h.sent.filter((m) => m.t === "permission_resolved")).toHaveLength(1);
  });

  test("a cancelled ask resolves to ACP's sentinel and is not left pending", async () => {
    const h = harness("ask-cancel");
    h.setTurn(new AbortController().signal);
    const pending = h.host.askPermission("Run it?", ALLOW_OR_DENY);
    h.host.cancelPendingPermissions();
    expect(await pending).toBe("__acp_cancelled__");
    h.host.cancelPendingPermissions(); // idempotent
    expect(h.sent.filter((m) => m.t === "permission_resolved")).toHaveLength(1);
  });

  test("an ask arriving with no live turn is refused as cancelled, never shown", async () => {
    const h = harness("ask-late");
    // This is the case that would otherwise take the NEXT utterance as its answer.
    expect(await h.host.askPermission("Run it?", ALLOW_OR_DENY)).toBe("__acp_cancelled__");
    expect(h.sent.some((m) => m.t === "permission")).toBe(false);
  });

  test("only a request-scoped elicitation is shown outside a turn, and unnumbered", async () => {
    const h = harness("ask-elicit");
    expect(await h.host.askElicitation({ mode: "form", message: "Name?", schema: {} })).toEqual({ action: "cancel" });
    expect(h.sent.some((m) => m.t === "elicitation")).toBe(false);

    const login = h.host.askElicitation({ mode: "url", message: "Sign in", url: "https://example.com", requestScoped: true });
    const asked = h.sent.find((m) => m.t === "elicitation")!;
    expect(asked.turn).toBeUndefined();
    h.host.answerElicitation(asked.reqId, "accept");
    expect(await login).toEqual({ action: "accept" });
  });
});

// ── binding ───────────────────────────────────────────────────────────────

describe("binding keeps the epoch guard's observable behaviour", () => {
  test("a rebind echoes the state the session is ACTUALLY using", async () => {
    const h = harness("bind-echo");
    // No cwd: the agent must not be spawned without a folder (a spurious "pick a
    // folder" and a churned process was the old bug). The echo still lands.
    await h.host.applyBind("codex");
    expect(h.sent.filter((m) => m.t === "bound_state").at(-1)).toMatchObject({ agentId: "codex", cwd: "", agentActive: false });
    expect(h.host.id).toBe("codex");
    expect(h.host.cwd).toBe("");
    expect(h.host.running).toBe(false);
  });

  test("the acpSession setting is cleared for another agent and kept for the same one", async () => {
    const { setSetting, getSetting } = await import("@openlive/db");
    const point = async (chatId: string, agentId: string | null) => {
      const h = harness(chatId);
      await h.host.applyBind(agentId);
    };
    // Rebinding the SAME agent (null → null) keeps its session id.
    await setSetting(acpSessionKey("host-kept"), "sess-a");
    await point("host-kept", null);
    expect(getSetting("acpSession:host-kept")).toBe("sess-a");
    // A different agent cannot load the old one's session, so it is forgotten.
    await setSetting("bind:host-moved", "claude-code");
    await setSetting("acpSession:host-moved", "sess-a");
    await point("host-moved", null);
    expect(getSetting("acpSession:host-moved")).toBe("");
  });

  test("a bind superseded before it commits leaves the original agent's session alone", async () => {
    const { setSetting, getSetting } = await import("@openlive/db");
    await setSetting("bind:host-flip", "");
    await setSetting("acpSession:host-flip", "sess-a");
    const h = harness("host-flip");
    // codex, then straight back: the codex bind is superseded before it writes.
    const both = Promise.all([h.host.applyBind("codex"), h.host.applyBind(null)]);
    await both;
    expect(getSetting("acpSession:host-flip")).toBe("sess-a");
    expect(h.host.id).toBe(null);
  });

  test("restore adopts the persisted agent, and a client bind that already arrived wins", async () => {
    const { setSetting } = await import("@openlive/db");
    await setSetting("bind:host-restore", "codex");
    const h = harness("host-restore");
    await h.host.restore();
    expect(h.host.id).toBe("codex");
    // A bind already stamped means restore must await it, not overwrite it.
    const h2 = harness("host-race");
    await h2.host.applyBind(null);
    await setSetting("bind:host-race", "codex");
    await h2.host.restore();
    expect(h2.host.id).toBe(null);
  });
});

// ── steering ──────────────────────────────────────────────────────────────

describe("steering: ordered GuidanceItem[], never concatenated", () => {
  test("the dispatch table follows AgentDef.acp.steering", () => {
    // Stage 0 measured hermes as the only mcp_guidance agent; the default is the
    // documented safe fallback.
    expect(steeringOf("hermes")).toBe("mcp_guidance");
    expect(steeringOf("claude-code")).toBe("cancel_resume");
    expect(steeringOf(null)).toBe("cancel_resume");
    expect(steeringOf("not-an-agent")).toBe("cancel_resume");
  });

  test("hermes queues the correction for the MCP drain instead of cancelling the task", async () => {
    const h = harness("steer-mcp");
    h.host.startTask("task_1", "change the header", new AbortController().signal);
    // A steered task is running under the agent that measured mcp_guidance.
    (h.host as unknown as { boundId: string }).boundId = "hermes";
    const outcome = await h.host.steerTask("task_1", item("g1", "do not add the README note"));
    expect(outcome).toMatchObject({ mechanism: "mcp_guidance", cancelled: false });
    expect(outcome.delivered.map((g) => g.text)).toEqual(["do not add the README note"]);
    // Still pending: the agent drains it through the hosted `user_guidance` tool,
    // so marking it delivered now would hide it from that poll.
    expect(h.ledger.drainPendingGuidance("task_1").map((g) => g.guidanceId)).toEqual(["g1"]);
  });

  test("two corrections queue in the order spoken and stay two items", async () => {
    const h = harness("steer-order");
    h.host.startTask("task_1", "spec", new AbortController().signal);
    (h.host as unknown as { boundId: string }).boundId = "hermes";
    await h.host.steerTask("task_1", item("g1", "first: keep the CSS"));
    await h.host.steerTask("task_1", item("g2", "second: change only the header"));
    const pending = h.ledger.drainPendingGuidance("task_1");
    expect(pending.map((g) => g.guidanceId)).toEqual(["g1", "g2"]);
    // NEVER one concatenated string: joining them is what makes "do X" + "not X"
    // read as a single instruction.
    expect(pending).toHaveLength(2);
    expect(pending[0]!.text).not.toContain("second");
  });

  test("the MCP drain hands the agent ordered guidance, and says so when there is none", async () => {
    const h = harness("steer-drain");
    h.host.startTask("task_1", "spec", new AbortController().signal);
    expect(h.host.drainGuidance("task_1")).toEqual({ items: [], text: "NO_NEW_GUIDANCE" });
    (h.host as unknown as { boundId: string }).boundId = "hermes";
    await h.host.steerTask("task_1", item("g1", "keep the CSS"));
    await h.host.steerTask("task_1", item("g2", "change only the header"));
    const drained = h.host.drainGuidance("task_1");
    // One numbered line each, in the order spoken — never one joined sentence.
    expect(drained.text).toBe("1. keep the CSS\n2. change only the header");
    expect(drained.items.map((g) => g.guidanceId)).toEqual(["g1", "g2"]);
    // Handed over, so a second poll does not apply the same correction twice.
    expect(h.host.drainGuidance("task_1").text).toBe("NO_NEW_GUIDANCE");
  });

  test("the default row cancels and sends a continuation prompt built from the ledger", async () => {
    const h = harness("steer-cancel");
    h.ledger.recordDispatch("task_1", "key-1", "change the header, keep the CSS", "claude-code");
    h.host.startTask("task_1", "change the header, keep the CSS", new AbortController().signal);
    (h.host as unknown as { boundId: string }).boundId = "claude-code";
    // There is no live agent, so the prompt is built but nothing is sent — which is
    // exactly what the shape assertions need.
    const outcome = await h.host.steerTask("task_1", item("g1", "do not touch the CSS"));
    expect(outcome).toMatchObject({ mechanism: "cancel_resume", cancelled: false });
    const prompt = outcome.prompt!;
    expect(prompt.startsWith("[CONTINUED]")).toBe(true);
    expect(prompt).toContain("The previous run was interrupted on purpose");
    // The ledger's own progress, never the model's memory of it.
    expect(prompt).toContain("taskId=task_1");
    expect(prompt).toContain("THE TASK AS GIVEN: change the header, keep the CSS");
    expect(prompt).toContain("CORRECTION (the user's latest words");
    expect(prompt).toContain("1. do not touch the CSS");
    expect(prompt).toContain("Continue the SAME task. Do not restart it, do not greet again");
    // One numbered line per correction, in order.
    expect(prompt).not.toContain("2. ");
  });
});

describe("the continuation prompt's shape", () => {
  test("numbers each correction in the order spoken, never joined", () => {
    const prompt = continuationPrompt("CURRENT STEP: editing", [item("g1", "keep the CSS"), item("g2", "change only the header")], "switch the title");
    expect(prompt).toContain("1. keep the CSS");
    expect(prompt).toContain("2. change only the header");
    expect(prompt.indexOf("1. keep the CSS")).toBeLessThan(prompt.indexOf("2. change only the header"));
    expect(prompt).toContain("THE TASK AS GIVEN: switch the title");
  });

  test("says so plainly when the ledger has nothing, rather than inventing progress", () => {
    const prompt = continuationPrompt("", []);
    expect(prompt).toContain("ALREADY DONE: nothing recorded yet.");
    expect(prompt).toContain("(none recorded");
    // No spec means "carry on", never a made-up task line.
    expect(prompt).toContain("REMAINING: carry on from exactly where you stopped.");
  });

  test("carries the digest through verbatim, so a specific status can be spoken", () => {
    const digest = "taskId=task_7 status=running agent=hermes started=22s ago\nCURRENT STEP: editing session.ts (4s ago)";
    expect(continuationPrompt(digest, [item("g1", "x")])).toContain(digest);
  });
});

// ── the <voice_summary> extractor ─────────────────────────────────────────

describe("voice_summary: at most three sentences, no code, no paths", () => {
  test("keeps the outcome and drops the code fence", () => {
    const reply = [
      "I changed the header title to OpenLive Pro.",
      "```ts",
      "const title = \"OpenLive Pro\";",
      "```",
      "The CSS was left untouched.",
    ].join("\n");
    const summary = voiceSummary(reply);
    expect(summary).toBe("I changed the header title to OpenLive Pro. The CSS was left untouched.");
    expect(summary).not.toContain("const");
    expect(summary).not.toContain("```");
  });

  test("never speaks a path", () => {
    const reply = "I edited C:\\proj\\src\\header.tsx and added the title. The build passes now.";
    const summary = voiceSummary(reply);
    expect(summary).not.toContain("C:\\proj");
    expect(summary).toBe("The build passes now.");
  });

  test("is at most THREE sentences, however long the reply was", () => {
    const reply = ["One happened.", "Two happened.", "Three happened.", "Four happened.", "Five happened."].join(" ");
    const summary = voiceSummary(reply);
    expect(sentences(summary)).toHaveLength(3);
    expect(summary).toBe("One happened. Two happened. Three happened.");
  });

  test("falls back to the LAST three sentences when nothing survives the filter", () => {
    // Every sentence is code-ish, so the readable filter empties: the tail is where
    // the outcome is stated, so that is what is spoken.
    const reply = "const a = 1. function b() { return a. } import x from y. class Z {}.";
    const summary = voiceSummary(reply);
    expect(summary).not.toBe("");
    expect(sentences(summary)).toHaveLength(3);
    expect(summary).toContain("class Z");
  });

  test("is empty when there is nothing to say, so a caller can skip the tag", () => {
    expect(voiceSummary("")).toBe("");
    expect(voiceSummary("   \n\n ")).toBe("");
    expect(voiceSummaryBlock("")).toBe("");
  });

  test("wraps a summary in the tag a model reads as state", () => {
    expect(voiceSummaryBlock("Done. The header is OpenLive Pro now.")).toBe("<voice_summary>Done. The header is OpenLive Pro now.</voice_summary>");
  });

  test("splits on the Bengali danda, because the user code-switches", () => {
    expect(sentences("হেডার বদলেছি। সিএসএস ছুঁইনি। Done.")).toEqual(["হেডার বদলেছি।", "সিএসএস ছুঁইনি।", "Done."]);
  });
});

// ── hideHosted, moved verbatim ────────────────────────────────────────────

describe("hideHosted drops the agent's report of OpenLive's own tools", () => {
  test("records the ids it hides and passes everything else through", async () => {
    const hidden = new Set<string>();
    const seen: string[] = [];
    const emit = hideHosted(async (e) => { seen.push(e.type); }, hidden);
    await emit({ type: "acp_tool_call", call: { id: "call-1", title: "mcp__openlive__look", kind: "other", status: "pending" } as never });
    await emit({ type: "acp_tool_call", call: { id: "own", title: "Read notes.md", kind: "read", status: "completed" } as never });
    // The hidden id is remembered, so its later update is dropped too.
    await emit({ type: "acp_tool_update", delta: { id: "call-1" } as never });
    expect(hidden.has("call-1")).toBe(true);
    expect(seen).toEqual(["acp_tool_call"]);
  });

  test("recognises the other title shape too (mcp.<server>.<tool>)", async () => {
    const hidden = new Set<string>();
    const emit = hideHosted(async () => {}, hidden);
    await emit({ type: "acp_tool_call", call: { id: "c", title: "mcp.openlive.open_url", kind: "execute", status: "pending" } as never });
    expect(hidden.has("c")).toBe(true);
  });
});

// ── the `look` handshake, moved verbatim ─────────────────────────────────

describe("the look frame handshake is serialized in the host", () => {
  /** The chain defers even the first grab to a microtask, so let it run. */
  const tick = () => new Promise((r) => setTimeout(r, 0));

  test("a frame response arms the slot and the next binary resolves it", async () => {
    const h = harness("look-one");
    const pending = h.host.requestFrame();
    await tick();
    const asked = h.sent.find((m) => m.t === "need_frame")!;
    expect(asked.reqId).toBeTruthy();
    h.host.onFrameResponse(asked.reqId);
    // The JPEG rides the next binary, prefixed with LIVE_TAG.FRAME_IN.
    const buf = Buffer.concat([Buffer.from([0x02]), Buffer.from("jpeg-bytes")]);
    expect(h.host.onFrame(buf)).toBe(true);
    expect(await pending).toEqual({ data: Buffer.from("jpeg-bytes").toString("base64"), mime: "image/jpeg" });
  });

  test("a failed response resolves null rather than hanging the turn", async () => {
    const h = harness("look-failed");
    const pending = h.host.requestFrame();
    await tick();
    h.host.onFrameResponse(h.sent.find((m) => m.t === "need_frame")!.reqId, true);
    expect(await pending).toBe(null);
  });

  test("two concurrent looks each resolve, instead of overwriting the one slot", async () => {
    const h = harness("look-chain");
    const both = Promise.all([h.host.requestFrame(), h.host.requestFrame()]);
    await tick();
    // Only the first is asked for until it resolves; then the second is asked.
    expect(h.sent.filter((m) => m.t === "need_frame")).toHaveLength(1);
    h.host.onFrameResponse(h.sent.find((m) => m.t === "need_frame")!.reqId);
    h.host.onFrame(Buffer.concat([Buffer.from([0x02]), Buffer.from("a")]));
    await tick();
    const second = h.sent.filter((m) => m.t === "need_frame").at(-1)!;
    h.host.onFrameResponse(second.reqId);
    h.host.onFrame(Buffer.concat([Buffer.from([0x02]), Buffer.from("b")]));
    const [a, b] = await both;
    // Neither promise is orphaned — the old wedge (turnActive stuck true).
    expect(a).toMatchObject({ mime: "image/jpeg" });
    expect(b).toMatchObject({ mime: "image/jpeg" });
    expect([a!.data, b!.data]).toEqual([Buffer.from("a").toString("base64"), Buffer.from("b").toString("base64")]);
  });

  test("a frame for no request, or the wrong tag, is ignored", async () => {
    const h = harness("look-stray");
    expect(h.host.onFrame(Buffer.from([0x02, 0xff]))).toBe(false);
    // A response for an id nobody asked for must not arm the slot.
    h.host.onFrameResponse("nope");
    expect(h.host.onFrame(Buffer.from([0x02, 0xff]))).toBe(false);
  });
});

test("the ws stub is only here to keep the import shape honest", () => {
  // `AcpHostPorts.send` speaks the same LiveServerMsg the socket does.
  const ws: Pick<WebSocket, "readyState"> = { readyState: 1 };
  expect(ws.readyState).toBe(1);
});
