import { randomUUID } from "node:crypto";
import type { WebSocket } from "ws";
import type { SseEvent, MessageBlock, LiveServerMsg } from "@openlive/shared";
import { LIVE_TAG, liveClientMsgSchema, agentLabel, classifyError, startReasonOf, toolSummary, withReplyLanguage, type AgentMetaWire, type LanguageCode } from "@openlive/shared";
import { createChat, addMessage, updateMessageContent, listMessages, renameChat, getSetting, setSetting, deleteChatsBefore } from "@openlive/db";
import { historyKeep, KEEP_MS } from "@openlive/flow-store/shared";
import { layout, resolveHome } from "@openlive/shared/home";
import type { Message } from "@openlive/harness";
import type { Approve, Emit, Session, Tool } from "../capabilities/types.js";
import { registry } from "../capabilities/registry.js";
import { CHAT } from "../capabilities/profiles.js";
import { offerOf, ToolSet } from "../capabilities/dispatch.js";
import { bridgedDevice, DEVICE_TIMEOUT_MS } from "../capabilities/device.js";
import { computer } from "../computer/helper.js";
import { slashSkill } from "../skills/tools.js";
import { finalizeToolBlocks, foldBlock, newFoldCtx, type FoldCtx } from "../block-emit.js";
import { LiveTurnRunner } from "./turn-runner.js";
import { narrationEnabled, wrapEmitWithNarration, createCommentaryGate } from "./narrator.js";
import { boundAgent, type AgentId, type ElicitationAnswer, type ElicitationAsk, type PermissionAskOption, type ReplayMessage } from "../agents/index.js";
import { AcpHost, ALLOW_OR_DENY } from "./acp-host.js";
import { Ledger } from "./ledger.js";
import { emitFact, type AgentFactProps } from "../telemetry/emit.js";
import { brainOf, modelFact, toolTally, reportReply, reportTurnError, TurnTimer, type Brain as BrainIdent } from "../telemetry/facts.js";
import { resolveLive } from "../providers.js";
import { log } from "../log.js";
import { cancelledText, sentAside } from "../turn.js";
import { hearReminders, liveSockets, type ReminderMsg } from "../reminders/fire.js";

type Frame = { data: string; mime: string };
type TurnFrame = Frame & { source: "camera" | "screen" | "attachment" };
const HISTORY_TURNS = 20; // recent messages to rehydrate on reconnect
const BRIDGE_TIMEOUT_MS = 5_000;
/** Tools whose work shows on its own: the worker's searches, and the checklist. */
const OWN_UI = new Set(["delegate", "update_todos"]);

// Some providers (e.g. MiniMax) leak control-token fragments like "[e[" into the
// text stream. Scrub ONLY those bracket-pair fragments — NOT every bracket: coding
// agents legitimately say things like `arr[0]`, and blanket-stripping brackets
// corrupted saved transcripts (`arr 0`). The live client strips the same noise for
// display/TTS.
function scrubControlTokens(blocks: MessageBlock[]): void {
  for (const b of blocks) {
    if (b.type === "text") b.text = b.text.replace(/[[\]][a-z0-9~!]{0,3}[[\]]/gi, " ").replace(/[ \t]{2,}/g, " ");
  }
}

// Strip OpenLive's OWN injected context out of a replayed user message so a resumed
// transcript reads as what the user actually said — not the voice/vision preamble,
// the "[Context — earlier…]" recap, or the "[The user is sharing…]" frame note. Each
// is a single `[…]` block with no internal `]`, so we match its opening marker and
// cut to the block's close — wherever it sits, and robust even if an agent's replay
// collapses the blank lines between blocks (a paragraph split would then over-strip).
const OPENLIVE_INJECTED = /\[(You're being used through OpenLive|How the user wants you to behave|Context \u2014 earlier in this voice conversation|The user is sharing their|A vision model is looking at the user's|Always reply in|The user cut you off|The user cancelled|This was first taken for the user talking)[^\]]*\]/g;
export function stripInjectedContext(blocks: MessageBlock[]): MessageBlock[] {
  return blocks
    .map((b) => (b.type === "text"
      ? { ...b, text: b.text.replace(OPENLIVE_INJECTED, "").replace(/\n{3,}/g, "\n\n").trim() }
      : b))
    .filter((b) => b.type !== "text" || b.text.length > 0);
}

// Replace the assistant's spoken text in `blocks` with exactly what the client
// says was voiced on a barge-in (live replies are plain text — a clean swap).
function truncateSpokenText(blocks: MessageBlock[], spoken: string): void {
  const s = spoken.trim();
  let placed = false;
  for (const b of blocks) {
    if (b.type !== "text") continue;
    if (!placed) { b.text = s; placed = true; } else b.text = "";
  }
  if (!placed && s) blocks.unshift({ type: "text", text: s });
}

// One live call — THIN. The browser runs the whole voice stack (VAD, STT, turn
// detection, TTS) on-device; this server only receives the final user text + the
// freshest camera frame, runs the LLM turn, streams reply text back, and PERSISTS
// the conversation.
export class LiveSession {
  private runner: LiveTurnRunner;
  private toolSession: Session;
  /** ACP for this conversation — binding, the hosted MCP server, asks, steering.
   *  Extracted so /live and /realtime share one implementation; the session keeps
   *  only what is about being a VOICE CALL. */
  private readonly host: AcpHost;
  private lastMeta: AgentMetaWire | null = null; // last agent_meta sent — re-sent on a same-bind rebind
  private agentStartMs = 0; // what the bound agent took to start; 0 without one
  // How the agent's session came up. It is known in the lobby, before the call's
  // record opens, so each turn's fact carries it in.
  private resumed: AgentFactProps<"call">["resumed"];
  private warmAc: AbortController | null = null; // aborts the cache-warm request on teardown
  private ac: AbortController | null = null;
  private turnActive = false;
  // An utterance (with its frames) that arrived mid-turn (barge-in), drained when the
  // current turn settles. Frames are queued too so a barge-in with the camera on
  // doesn't lose what the user was showing.
  private queued: { text: string; frames: TurnFrame[]; lang?: LanguageCode; turn?: number; wordsAt?: number[]; speaker?: string; aside?: boolean; typed?: boolean } | null = null;
  private bargeSpoken: string | null = null; // on barge-in, the text the client actually SPOKE
  private cutSaved: Promise<unknown> = Promise.resolve(); // the last cut written for a coding agent
  // The client's number for the utterance the running turn answers, echoed on its
  // events.
  private replyTurn: number | undefined;
  // The last saved reply, while the client may still be voicing it: a model streams
  // its text far faster than speech, so most barge-ins land after the turn is over.
  private lastReply: { id: string; blocks: MessageBlock[]; byRunner: boolean } | null = null;
  private startup: Promise<void> = Promise.resolve();
  private expectReplay = false;               // this chat was empty when a resume began → persist recovered turns
  private cameraOn = false;
  private screenOn = false;
  private titled = false;                  // rename the chat from the first user turn
  private closed = false;

  // `look` tool ↔ client hi-res frame handshake: the pending slot lives in the host.
  // OS bridge (clipboard / open_url / the device) ↔ client handshake. The client
  // runs the action via Electron and replies; on the web it replies "not available".
  private bridgePending = new Map<string, (out: string) => void>();
  // The call's tools, served to a coding agent over MCP: the same set the
  // built-in brain runs, so one implementation answers both.
  private tools: ToolSet;
  private approve: Approve;
  /** The running turn's emit, so a tool shows its chip in that turn. */
  private toolEmit: Emit = () => {};
  /** A timer or reminder that went off, for the client to show and say. */
  private readonly hear = (m: ReminderMsg) => this.send(m);

  /** `device`: the client is the desktop app, which can reach the ol-input addon. */
  constructor(private ws: WebSocket, private chatId: string, private lang?: LanguageCode, device = false) {
    // What the call can reach. The client answers every bridge op; the camera
    // frame for the device comes from the share the call already holds.
    this.toolSession = {
      clipboard: { read: () => this.bridge("clipboard_read"), write: (text) => this.bridge("clipboard_write", text) },
      openUrl: (url) => this.bridge("open_url", url),
      share: { showing: () => (this.screenOn ? "screen" : this.cameraOn ? "camera" : null), frame: () => this.requestFrame() },
      // Read live, so the file tools track a folder change mid-call. `host` is
      // built further down the constructor, so this must stay a thunk all the
      // way to first use — dereferencing it here throws on every /live upgrade.
      workspace: () => this.host.cwd,
      emit: (e) => this.toolEmit(e),
      elicit: (req) => this.askElicitation(req),
      ...(device && { device: bridgedDevice((arg) => this.bridge("flow_device", arg, DEVICE_TIMEOUT_MS), async () => (this.cameraOn ? this.requestFrame() : null)) }),
      ...(device && computer.available() && { computer }),
    };
    const call = this;
    this.approve = CHAT.approval((question) => call.askPermission(question, ALLOW_OR_DENY).then((id) => id === "allow"));
    // ACP, hosted by its own object: the same code /realtime will use. Everything
    // it needs from the session is a callback, so the two never import each other.
    this.host = new AcpHost({
      chatId: this.chatId,
      closed: () => this.closed,
      send: (m) => this.send(m),
      session: () => this.toolSession,
      tools: () => this.tools,
      approve: this.approve,
      tally: toolTally("call"),
      ledger: new Ledger(this.chatId, layout(resolveHome()).data, () => Date.now()),
      turnActive: () => this.turnActive,
      turnSignal: () => this.ac?.signal ?? null,
      replyTurn: () => this.replyTurn,
      seed: () => this.rehydrate(),
      pendingCut: () => this.pendingCut(),
      onMeta: (meta) => { this.lastMeta = meta; },
      onReplay: (msgs) => this.ingestReplay(msgs),
      onReplayExpected: (wanted) => { this.expectReplay = wanted; },
      onResumed: (how) => { this.resumed = how; },
      onFolderMoved: () => { this.tools = this.callTools(); this.runner.tools = this.tools; },
      onStarted: (ms) => { this.agentStartMs = ms; },
      onReady: () => this.send({ t: "sse", event: { type: "status", text: "ready" } }),
      onStartError: (e) => this.reportStartError(e),
    });

    // Tools and the runner last: the skill provider reads `host.cwd`, so `host`
    // has to exist first, and `runner` takes the resolved set.
    this.tools = this.callTools();
    this.runner = new LiveTurnRunner(this.tools, this.toolSession, { approve: this.approve, tally: toolTally("call") });

    ws.on("message", (data: Buffer, isBinary: boolean) => {
      if (isBinary) this.onBinary(data);
      else this.onText(data.toString()).catch((e) => log.error("live", "text:", e));
    });
    ws.on("close", () => this.dispose());
    ws.on("error", () => this.dispose());
    hearReminders(this.hear);
  }

  start(): Promise<void> { return (this.startup = this.boot()); }

  private async boot() {
    // Persist the chat row + rehydrate recent history (so a reconnect mid-call
    // doesn't make the agent forget what was already said).
    if (this.chatId) {
      // Older chats go to what Settings > Chat keeps as a call starts, as well as
      // when History is read, so Keep nothing holds without History ever opening.
      const keep = historyKeep(getSetting("chatHistory"));
      if (keep !== "forever") await deleteChatsBefore(new Date(Date.now() - KEEP_MS[keep]).toISOString(), this.chatId).catch((e) => log.warn("live", "prune chats:", e));
      await createChat(this.chatId);
      const prior = this.rehydrate();
      if (prior.length) this.runner.seed(prior);
    }
    // Resume a persisted bind — but ONLY if the client's own bind hasn't already
    // arrived (the host keeps the epoch guard and the same reasoning).
    await this.host.restore();
    if (this.host.running) return;
    // Warm the prompt cache + connection in the background so the first spoken turn
    // answers fast. Tell the client when it's done (drives the "Warming up…" spinner);
    // always signal ready, even on failure, so the indicator never sticks.
    this.warmAc = new AbortController();
    // A turn waiting on this start primes the cache itself; a warm-up would send it twice.
    void (this.turnActive ? Promise.resolve() : this.runner.warm(this.warmAc.signal, this.lang))
      .catch(() => {})
      .finally(() => { if (!this.closed) this.send({ t: "sse", event: { type: "status", text: "ready" } }); });
  }

  /** Stored messages → harness messages (text only — enough for continuity). */
  private rehydrate(): Message[] {
    let rows;
    try { rows = listMessages(this.chatId); } catch { return []; }
    const recent = rows.slice(-HISTORY_TURNS);
    const out: Message[] = [];
    for (const m of recent) {
      if (m.role !== "user" && m.role !== "assistant") continue;
      const text = m.content.filter((b) => b.type === "text").map((b) => (b as any).text).join("").trim();
      if (text) out.push({ role: m.role, text });
    }
    const asked = out.findLast((m) => m.role === "user");
    if (asked && this.pendingCut()?.cancelled) asked.text = cancelledText(asked.text);
    return out;
  }

  // ── inbound ───────────────────────────────────────────────────────────────
  private onBinary(buf: Buffer) {
    // The only binary a call receives is the JPEG the `look` handshake awaited —
    // the host holds that slot, so it is the one that resolves it.
    this.host.onFrame(buf);
  }

  private async onText(str: string) {
    let msg;
    try { msg = liveClientMsgSchema.parse(JSON.parse(str)); } catch { return; }
    switch (msg.t) {
      case "user_text":
        // A permission/elicitation may be awaiting the user. A client showing it answers
        // it there, so a sentence landing here was said before the ask reached the
        // client, which then drops the ask as an older turn's: nobody can answer it. It
        // is refused, and the sentence is a turn like any other.
        this.host.cancelPendingPermissions();
        this.host.cancelPendingElicitations();
        return void this.runTurn(msg.text, msg.frames ?? [], msg.lang, msg.turn, msg.wordsAt, msg.speaker, msg.aside, msg.typed);
      case "cancel":
        emitFact("call", { interrupted: 1 });
        // A client showing an ask holds its barge-in, so a cancel means the ask never
        // reached it: the ask is refused along with the turn.
        if (this.turnActive) this.bargeSpoken = msg.spoken ?? "";
        else if (msg.spoken != null) this.cutSavedReply(msg.spoken);
        return this.interrupt();
      case "control":
        if (msg.action === "camera_on") { this.cameraOn = true; emitFact("call", { camera_used: true }); }
        else if (msg.action === "camera_off") this.cameraOn = false;
        else if (msg.action === "screen_on") { this.screenOn = true; emitFact("call", { screen_used: true }); }
        else if (msg.action === "screen_off") this.screenOn = false;
        else if (msg.action === "end") this.dispose();
        return;
      case "frame_response":
        this.host.onFrameResponse(msg.reqId, msg.failed);
        return;
      case "tool_bridge_result": {
        const r = this.bridgePending.get(msg.reqId);
        if (r) { this.bridgePending.delete(msg.reqId); r(msg.output); }
        return;
      }
      case "bind": return this.host.applyBind(msg.agentId, msg.cwd, msg.resumeSessionId);
      case "permission_response": {
        this.host.answerPermission(msg.reqId, msg.optionId); // settle() clears the map + notifies the client
        return;
      }
      case "elicitation_response": {
        this.host.answerElicitation(msg.reqId, msg.action, msg.content as Record<string, unknown> | undefined);
        return;
      }
      case "set_model": { this.host.setModel(msg.modelId).catch((e) => log.error("live", "set_model:", e)); return; }
      case "set_mode": { this.host.setMode(msg.modeId).catch((e) => log.error("live", "set_mode:", e)); return; }
      case "set_option": { this.host.setOption(msg.optionId, msg.valueId).catch((e) => log.error("live", "set_option:", e)); return; }
    }
  }

  /** Ask the client to run an OS action (clipboard / open_url / the device) and
   *  await its result. On non-desktop clients the reply is instant ("not available"). */
  private bridge(op: "clipboard_read" | "clipboard_write" | "open_url" | "flow_device", arg?: string, timeoutMs = BRIDGE_TIMEOUT_MS): Promise<string> {
    return new Promise((resolve) => {
      const reqId = randomUUID();
      const timer = setTimeout(() => {
        if (this.bridgePending.delete(reqId)) resolve("That action timed out.");
      }, timeoutMs);
      this.bridgePending.set(reqId, (out) => { clearTimeout(timer); resolve(out); });
      this.send({ t: "tool_bridge", reqId, op, arg, turn: this.replyTurn });
    });
  }

  // ── turn ────────────────────────────────────────────────────────────────
  private async runTurn(text: string, frames: TurnFrame[] = [], lang?: LanguageCode, turn?: number, wordsAt?: number[], speaker?: string, aside?: boolean, typed?: boolean) {
    if (!text.trim() || this.closed) return;
    // A new utterance during an in-flight turn (barge-in) must NOT be dropped:
    // queue it (append text, keep the freshest frames) and the finally below drains
    // it as one turn.
    if (this.turnActive) {
      const had = this.queued?.frames ?? [];
      const live = (fs: TurnFrame[]) => fs.filter((f) => f.source !== "attachment");
      const attached = (fs: TurnFrame[]) => fs.filter((f) => f.source === "attachment");
      this.queued = {
        text: this.queued ? `${this.queued.text} ${text}` : text,
        // The freshest live view, and every image the user attached along the way.
        frames: [...attached(had), ...attached(frames), ...(live(frames).length ? live(frames) : live(had))],
        lang,
        turn,
        // Two utterances' onsets count from two different starts: a joined turn keeps none.
        wordsAt: this.queued ? undefined : wordsAt,
        speaker: !this.queued || this.queued.speaker === speaker ? speaker : undefined,
        // A joined turn is mostly not what was sent on, so it is not flagged.
        aside: !this.queued && aside,
        typed: !this.queued && typed,
      };
      return;
    }
    this.turnActive = true;
    this.replyTurn = turn;
    this.lastReply = null;
    const ac = new AbortController();
    this.ac = ac;
    // A reconnect flushes a queued utterance right behind the bind: wait for the
    // history and the bound agent, or the turn is saved into the history it then
    // loads twice, and goes to the built-in brain instead of the agent.
    await this.startup.catch(() => {});

    let said = aside ? sentAside(text) : text;
    const blocks: MessageBlock[] = [];
    const hosted = new Set<string>();
    const foldCtx = newFoldCtx();
    const turnStats = { timer: new TurnTimer(), failed: false };
    const emit = this.blockEmit(blocks, ac.signal, foldCtx, turnStats);

    if (this.chatId) {
      await addMessage(this.chatId, "user", [{ type: "text", text, ...(wordsAt && { wordsAt }), ...(speaker && { speaker }), ...(typed && { typed }) }], true /* live */).catch((e) => log.error("live", "persist user turn:", e));
      // Auto-title the conversation from the first thing the user says.
      if (!this.titled) { this.titled = true; await renameChat(this.chatId, text.replace(/\s+/g, " ").trim().slice(0, 48) || "Live conversation").catch(() => {}); }
    }
    // Clean, agent-agnostic speech for the whole turn:
    //  • wrapEmitWithNarration — a short voiced status line for long tool runs (opt-out).
    //  • createCommentaryGate — speak the opening line + final answer only; mid-work
    //    commentary between tools goes to the (unspoken) reasoning channel. `flush()`
    //    after the turn resolves voices the buffered final answer.
    // Both wrap the SAME emit, so every agent (and the built-in brain) behaves alike.
    const narrated = narrationEnabled(getSetting("narrateProgress")) ? wrapEmitWithNarration(emit, ac.signal) : emit;
    const gate = createCommentaryGate(narrated, ac.signal);
    this.toolEmit = gate.emit;
    try {
      await this.host.ready.catch(() => {}); // wait out the ACP handshake on the first turn
      await this.cutSaved;
      if (this.chatId && getSetting(`agentCut:${this.chatId}`)) await setSetting(`agentCut:${this.chatId}`, "");
      // A typed "/name" loads that OpenLive skill for the turn, unless the bound
      // agent has a command by that name: the agent's own commands come first.
      this.refreshTools();
      const word = typed && !aside ? /^\/(\S+)/.exec(text.trim())?.[1] : undefined;
      const skill = word && !this.lastMeta?.commands.some((c) => c.name === word)
        ? await slashSkill(text, this.tools, { ...this.toolSession, signal: ac.signal, context: null })
        : null;
      if (skill) said = `${skill.content}\n\n${skill.rest || `Use the ${skill.name} skill.`}`;
      // The agent's own turn goes through the host, so /realtime drives it the same
      // way. `hide` drops its report of OpenLive's own tools, which show once.
      if (await this.host.runTurn({ text: withReplyLanguage(said, lang), frames, ...(!aside && !skill && text.trimStart().startsWith("/") && { command: text }) }, this.host.hide(gate.emit, hosted), ac.signal)) {
        await gate.flush();
      } else if (this.host.id) {
        // A coding agent is bound but not running (no folder yet, or its start
        // failed). NEVER answer with the built-in brain as if it were the agent —
        // that silently swaps who the user is talking to. Say what's wrong instead.
        const label = agentLabel(this.host.id);
        await emit({
          type: "error",
          message: this.host.cwd
            ? `${label} isn't connected yet. Give it a moment, or switch agents and back to retry.`
            : `${label} needs a project folder before it can start. Pick one from the folder menu in the top bar, then ask again.`,
          code: this.host.cwd ? "agent_start_failed" : "agent_no_folder",
        });
      } else {
        await this.runner.runTurn(said, frames, gate.emit, ac.signal, lang);
        await gate.flush();
      }
    } catch (e) {
      if (!ac.signal.aborted) {
        log.error("live", "turn:", e);
        // A thrown turn was invisible before — the call went quiet with no clue.
        try { await emit({ type: "error", message: `That didn't go through: ${String((e as Error)?.message ?? e)}`, code: classifyError(e) }); } catch { /* emit is best-effort */ }
      }
    } finally {
      // Turn is over (normally, barge-in, watchdog cut, or error) — never leave an
      // agent permission ask dangling; answer it cancelled (ACP MUST). No-op unless
      // one was actually pending.
      this.host.endTurn();
      this.toolEmit = () => {};
      // A stopped turn's calls to OpenLive's own tools are refused if they land late.
      if (ac.signal.aborted) this.host.refuseLate(hosted);
      const byRunner = !this.host.running && !this.host.id;
      // On barge-in, persist only what was actually SPOKEN.
      if (ac.signal.aborted && this.bargeSpoken != null) {
        truncateSpokenText(blocks, this.bargeSpoken);
        if (byRunner) this.runner.truncateReply(this.bargeSpoken);
        this.cutAgentReply(this.bargeSpoken, true);
      }
      this.bargeSpoken = null;
      scrubControlTokens(blocks);
      // Snapshot live terminal output into its tool calls + settle unfinished
      // statuses (pending/in_progress → canceled) before the turn is persisted.
      finalizeToolBlocks(blocks, foldCtx);
      const own = blocks.filter((b) => b.type === "acp_tool");
      const ownFailed = own.filter((b) => b.call.status === "failed").length;
      const answered = !ac.signal.aborted && !turnStats.failed;
      emitFact("call", {
        ...this.brain(), ...this.modelFamily(), turns: 1, lang: lang ?? "en", agent_start_ms: this.agentStartMs, ...(this.resumed && { resumed: this.resumed }),
        ...(own.length && { agent_tools: own.length }), ...(ownFailed && { agent_tools_failed: ownFailed }),
        ...turnStats.timer.timings(answered),
      });
      if (answered) reportReply("call");
      if (this.chatId && blocks.length) {
        const saved = await addMessage(this.chatId, "assistant", blocks, true /* live */).catch((e) => { log.error("live", "persist assistant turn:", e); return null; });
        if (saved && !ac.signal.aborted) this.lastReply = { id: saved.id, blocks, byRunner };
      }
      this.send({ t: "sse", event: { type: "done" }, turn: this.replyTurn });
      if (this.ac === ac) { this.ac = null; this.turnActive = false; }
      const q = this.queued; this.queued = null;
      if (q && !this.closed) void this.runTurn(q.text, q.frames, q.lang, q.turn, q.wordsAt, q.speaker, q.aside, q.typed); // drain a barge-in utterance (with its frames)
    }
  }

  /** An Emit that both forwards SSE to the client and records ordered blocks.
   *  The signal gate drops late events after a barge-in aborts the spoken turn. */
  private blockEmit(blocks: MessageBlock[], signal: AbortSignal, ctx: FoldCtx, stats: { timer: TurnTimer; failed: boolean }): Emit {
    return async (e: SseEvent) => {
      if (signal.aborted || this.closed) return; // barge-in → drop late events
      if (e.type === "text_delta") stats.timer.firstText();
      else if (e.type === "error") { stats.failed = true; reportTurnError("call", this.brain(), e); }
      foldBlock(blocks, e, ctx);
      this.send({ t: "sse", event: e, turn: this.replyTurn });
    };
  }

  /** A barge-in after the reply was saved: cut it, and the model's memory of it,
   *  back to what the voice had said. */
  private cutSavedReply(spoken: string) {
    const r = this.lastReply;
    this.lastReply = null;
    if (!r) return;
    truncateSpokenText(r.blocks, spoken);
    if (r.byRunner) this.runner.truncateReply(spoken);
    else this.cutAgentReply(spoken);
    try { updateMessageContent(r.id, r.blocks); } catch (e) { log.error("live", "cut saved reply:", e); }
  }

  /** A coding agent keeps its own memory of the reply, and a resumed session
   *  brings it back: the cut waits in the settings until a turn has told it. A
   *  cancelled request waits there for the built-in brain's next call too. */
  private cutAgentReply(spoken: string, cancelled = false) {
    this.host.cut(spoken, cancelled);
    if (this.chatId) this.cutSaved = setSetting(`agentCut:${this.chatId}`, JSON.stringify({ spoken, cancelled })).catch(() => {});
  }

  /** The cut no turn has told the brain yet. A bare string is one saved before cancels were kept. */
  private pendingCut(): { spoken: string; cancelled?: boolean } | null {
    const raw = this.chatId ? getSetting(`agentCut:${this.chatId}`) : "";
    if (!raw) return null;
    const cut = JSON.parse(raw) as string | { spoken: string; cancelled?: boolean };
    return typeof cut === "string" ? { spoken: cut } : cut;
  }

  private interrupt() {
    this.ac?.abort();
    this.host.endTurn();
  }

  /** Persist + surface prior turns recovered from an agent's session/load. Only
   *  when the chat is empty (external-origin resume) — an OpenLive-origin chat
   *  already renders its own transcript, so the replayed copy is dropped (the load
   *  just re-primes the agent's context). */
  private async ingestReplay(messages: ReplayMessage[]): Promise<void> {
    if (!this.chatId || this.closed || !messages.length) return;
    // `expectReplay` was decided at bind time, BEFORE any user turn could persist —
    // so a user who spoke before the load finished can't flip this to "OpenLive-origin"
    // and drop the recovered transcript (the old listMessages check raced that turn).
    if (!this.expectReplay) return;
    this.expectReplay = false; // ingest once
    try {
      for (const m of messages) {
        const content = m.role === "user" ? stripInjectedContext(m.content) : m.content;
        if (content.length) await addMessage(this.chatId, m.role, content); // sequential: keep order
      }
      this.send({ t: "reload_history" });
    } catch (e) { log.error("live", "replay ingest:", e); }
  }

  /** The agent failed to start: say so on the wire and count it, naming the fix. */
  private reportStartError(e: unknown): void {
    const id = this.host.id;
    const text = String((e as Error)?.message ?? e);
    const event = { type: "error" as const, message: startReasonOf(e) ? text : `Couldn't start ${id}: ${text}`, code: classifyError(e, "agent_start_failed") };
    this.send({ t: "sse", event });
    reportTurnError("call", this.brain(), { ...event, startReason: startReasonOf(e) });
  }

  // ── permission / elicitation / frames ─────────────────────────────────────
  /** Relay an agent permission ask to the client and await the chosen option id. */
  private askPermission(question: string, options: PermissionAskOption[], toolCallId?: string): Promise<string> {
    return this.host.askPermission(question, options, toolCallId);
  }

  /** Relay an agent elicitation (login URL / input form) to the client. */
  private askElicitation(req: ElicitationAsk): Promise<ElicitationAnswer> {
    return this.host.askElicitation(req);
  }

  // ── agent binding ─────────────────────────────────────────────────────────
  // applyBind / applyBindInner, the epoch guard, createBoundAgent, serveMcp, the
  // `acpSession:${chatId}` persistence, hideHosted/deadHosted, askPermission,
  // askElicitation and requestFrame's frame handshake now live in `AcpHost`
  // (./acp-host.js), so /realtime binds, hosts and asks through the same code.

  /** Who answers this call, as telemetry names it; nothing when settings cannot say. */
  private brain(): BrainIdent {
    if (this.host.id) return brainOf("acp", this.host.id);
    try { return brainOf("api", resolveLive().provider.id); }
    catch { return {}; }
  }

  /** The family of the model answering this call; nothing when it cannot be told. */
  private modelFamily() {
    if (this.host.id) return modelFact("", this.lastMeta);
    try { return modelFact(resolveLive().model); }
    catch { return {}; }
  }

  /** The call's tools, each with its chip. Built again when the folder changes, as a folder brings its own skills. */
  private callTools(): ToolSet {
    const set = registry.tools(CHAT, this.toolSession);
    const chip = (t: Tool) => (OWN_UI.has(t.name) ? t : this.chipped(t));
    return new ToolSet(set.list.map(chip), set.onDemand?.list.map(chip));
  }

  /** The call's tools built again for a turn, kept only when a switch changed what they offer (see FlowLiveSession.refreshTools). */
  private refreshTools(): void {
    const next = this.callTools();
    if (offerOf(next) === offerOf(this.tools)) return;
    this.tools = next;
    this.runner.tools = next;
    this.host.toolsChanged(next.list.map((t) => t.name));
  }

  /** A tool with its chip in the running turn, the same whichever brain calls it. */
  private chipped(t: Tool): Tool {
    return {
      ...t,
      execute: async (args, ctx) => {
        const id = randomUUID();
        const summary = toolSummary(t.name, args);
        await this.toolEmit({ type: "tool_start", id, tool: t.name, ...(summary && { summary }) });
        try { return await t.execute(args, ctx); } finally { await this.toolEmit({ type: "tool_done", id }); }
      },
    };
  }

  // ── `look` handshake ────────────────────────────────────────────────────
  /** Grab one fresh hi-res frame (serialized inside the host: one slot only). */
  private requestFrame(): Promise<Frame | null> {
    return this.host.requestFrame();
  }

  // ── send / teardown ───────────────────────────────────────────────────────
  private send(m: LiveServerMsg) {
    if (this.closed || this.ws.readyState !== this.ws.OPEN) return;
    this.ws.send(JSON.stringify(m));
  }

  private dispose() {
    if (this.closed) return;
    this.closed = true;
    liveSockets.delete(this.hear);
    this.warmAc?.abort();
    this.ac?.abort();
    this.host.dispose();
    for (const r of this.bridgePending.values()) r("The session ended.");
    this.bridgePending.clear();
    try { this.ws.close(); } catch { /* already closing */ }
  }
}
