import { randomUUID } from "node:crypto";
import { AGENT_REGISTRY, agentLabel, isAgentId, LIVE_TAG, type AgentId, type AgentMetaWire, type GuidanceItem, type LiveServerMsg, type SteeringCapability } from "@openlive/shared";
import { listMessages, setChatContext, setSetting } from "@openlive/db";
import type { Message } from "@openlive/harness";
import type { Approve, Emit, Session } from "../capabilities/types.js";
import type { ToolSet, ToolTally } from "../capabilities/dispatch.js";
import { MCP_SERVER_NAME, serveMcp } from "../capabilities/mcp.js";
import { hostedBy } from "../agents/mcp-config.js";
import { createBoundAgent, setBoundAgent, boundAgent, agentCwd, PERMISSION_CANCELLED, type Agent, type ElicitationAnswer, type ElicitationAsk, type PermissionAskOption, type ReplayMessage } from "../agents/index.js";
import type { TurnInput } from "../agents/types.js";
import { askOutcome, permissionFact, type PermissionOutcome } from "../telemetry/facts.js";
import { emitFact } from "../telemetry/emit.js";
import { isSpeakable, type Ledger } from "./ledger.js";

// ACP hosting for one conversation, extracted from `LiveSession` so `/live` and
// `/realtime` drive a coding agent through the SAME code. Nothing here knows about
// Gemini, audio or transcripts; it owns the four things that are about speaking
// ACP rather than about being a voice call:
//
//   1. BINDING — applyBind's epoch guard, the agent lifecycle, the folder rebuild,
//      and the `acpSession:<chatId>` persistence History's resume depends on.
//   2. MCP HOSTING — the session's own tools served to the agent on loopback. ONE
//      server: the one `serveMcp` already provides, never a second.
//   3. ASKING — permission and elicitation prompts, with one settle path and the
//      120 s timeouts, so a hung answer can never wedge a turn.
//   4. STEERING — ordered `GuidanceItem[]` applied per agent, because ACP has no
//      standard steering and Stage 0 measured which row each agent supports.
//
// ONE deliberate behaviour change rides along, as the build order requires:
// `serveMcp` used to bind its abort signal to `this.ac` — a SINGLE voice turn. A
// coding task outlives voice turns (the user speaks, the socket sleeps and wakes,
// the agent is still working), so the hosted tools were refused outside a live
// turn and a `user_guidance` poll aborted the instant the turn ended. They are
// bound to `activeTaskSignal()` instead: alive for the whole coding task, aborted
// only when that task is cancelled.

/** A fresh client frame (the `look` handshake's result), owned by the session that
 *  holds the socket — the host itself never grabs one. */
export type Frame = { data: string; mime: string };

/** How an agent's own session came up, as ACP reports it: `none` when no earlier
 *  one was asked for, `fell_back` when one was and a fresh session started. */
export type AgentResumeKind = "none" | "resumed" | "loaded" | "fell_back";

/** The one permission vocabulary a call offers; `kind` maps a spoken yes/no. */
export const ALLOW_OR_DENY: PermissionAskOption[] = [
  { id: "allow", label: "Allow", kind: "allow_once" },
  { id: "deny", label: "Deny", kind: "reject_once" },
];

const PERM_TIMEOUT_MS = 120_000;
const ELICIT_TIMEOUT_MS = 120_000;
const FRAME_TIMEOUT_MS = 4_000;

/** The `acpSession:<chatId>` setting: the agent's OWN session id for this chat,
 *  kept both as the resume key and on the chat row so History can dedup a call
 *  against its on-disk agent session — exactly what `claude --resume` reopens. */
export const acpSessionKey = (chatId: string): string => (chatId ? `acpSession:${chatId}` : "");

/**
 * `hideHosted` — an agent reports its use of the call's OWN tools as tool calls of
 * its own. Dropped, so they render once, as the built-in brain's do; the ids are
 * kept in `hidden` so a late update for one is dropped too.
 */
export function hideHosted(emit: Emit, hidden: Set<string>): Emit {
  return (e) => {
    if (e.type === "acp_tool_call" && hostedBy(e.call.title, MCP_SERVER_NAME)) hidden.add(e.call.id);
    const id = e.type === "acp_tool_call" ? e.call.id : e.type === "acp_tool_update" ? e.delta.id : "";
    return id && hidden.has(id) ? undefined : emit(e);
  };
}

// ── the <voice_summary> extractor ────────────────────────────────────────────

/** Code that is never speech: fences, inline code, braces, and shell-ish lines.
 *  Paths and commands are covered by the ledger's own `isSpeakable`, so a status
 *  answer and a summary stay one vocabulary instead of two. */
const CODEISH = /(```|~~~|`|\{|\}|=>|\b(?:function|const|let|var|import|export|return|class|async|await)\b)/;

/** Sentence split that keeps terminal punctuation, including the Bengali danda
 *  and the fullwidth stops, because the user code-switches mid-sentence. */
export function sentences(text: string): string[] {
  return String(text ?? "")
    .replace(/\r/g, " ")
    .split(/(?<=[.!?।॥．。])\s+|\n+/)
    .map((s) => s.replace(/[ \t]{2,}/g, " ").trim())
    .filter(Boolean);
}

/**
 * `<voice_summary>` — at most three sentences of what the agent DID, safe to speak.
 *
 * A coding agent's reply is mostly code fences, paths and commands, and the whole
 * point of a voice call is that none of it is read aloud. So: drop fenced blocks
 * first, then keep the sentences that carry no code and no path, then cap at
 * three. When none survive, fall back to the LAST THREE raw sentences — the end of
 * a reply is where the outcome is stated, and a summary that says nothing is worse
 * than one that names a file.
 *
 * Returns "" when there is nothing to say, so a caller can skip the tag entirely.
 */
export function voiceSummary(text: string): string {
  // Fenced blocks are code by construction — never speak them, not even a fragment.
  const unfenced = String(text ?? "").replace(/```[\s\S]*?```/g, " ").replace(/~~~[\s\S]*?~~~/g, " ");
  const all = sentences(unfenced);
  if (!all.length) return "";
  const readable = all.filter((s) => !CODEISH.test(s) && isSpeakable(s));
  // Fallback is the tail, not the head: the end of a long report is the outcome.
  const picked = (readable.length ? readable : all.slice(-3)).slice(0, 3);
  return picked.map((s) => s.replace(/`/g, "").replace(/\s+/g, " ").trim()).filter(Boolean).join(" ");
}

/** The tag itself, or "" when there is nothing speakable to say. */
export const voiceSummaryBlock = (text: string): string => {
  const summary = voiceSummary(text);
  return summary ? `<voice_summary>${summary}</voice_summary>` : "";
};

// ── the continuation prompt ─────────────────────────────────────────────────

/**
 * The prompt the `cancel_resume` row sends after `session/cancel`.
 *
 * That cancel is the one mechanism every agent has, and it stops the agent at a
 * step boundary — so it can lose the in-flight step. The prompt therefore has to
 * re-establish what was already DONE from the ledger, or the agent re-plans from
 * scratch and redoes work the user was told was finished. Its shape is fixed:
 *
 *   [CONTINUED]
 *   <the ledger's status digest: current step, what just completed, ages>
 *   THE TASK AS GIVEN: <the spec the task started with>
 *   CORRECTION (the user's latest words — apply it, do not repeat what is above):
 *   1. <the first correction, exactly as said>
 *   2. <the second>
 *   Continue the SAME task. Do not restart it, do not greet again, and do not ask what to do.
 *
 * Corrections are NUMBERED and never concatenated: "do X" then "actually not X"
 * joined by a space reads as one instruction, and the order is the whole point.
 */
export function continuationPrompt(digest: string, guidance: readonly GuidanceItem[], spec?: string): string {
  const done = digest.trim();
  const lines = [
    "[CONTINUED]",
    "The previous run was interrupted on purpose to apply a correction from the user. What was already done is real work: do not redo it or re-report it.",
    done || "ALREADY DONE: nothing recorded yet.",
    spec && spec.trim() ? `THE TASK AS GIVEN: ${spec.trim()}` : "REMAINING: carry on from exactly where you stopped.",
    "CORRECTION (the user's latest words — apply it, and do not repeat what is above):",
  ];
  if (guidance.length) for (const [i, g] of guidance.entries()) lines.push(`${i + 1}. ${g.text.trim()}`);
  else lines.push("(none recorded — carry on and complete the task as given.)");
  lines.push("Continue the SAME task. Do not restart it, do not greet again, and do not ask what to do.");
  return lines.join("\n");
}

/** Which steering row an agent supports. Absent means the documented safe default. */
export function steeringOf(agentId: AgentId | string | null | undefined): SteeringCapability {
  if (!agentId || !isAgentId(agentId)) return "cancel_resume";
  return AGENT_REGISTRY[agentId].acp.steering ?? "cancel_resume";
}

/** A coding task the host is running. */
export interface ActiveTask {
  taskId: string;
  spec: string;
  /** The agent driving it, so a steer dispatches on the right mechanism. */
  agentId: AgentId | null;
  /** The TASK's cancellation — not a voice turn's. */
  signal: AbortSignal;
  startedAt: number;
}

/** How a steer was applied, so the caller can narrate it and set its facts. */
export interface SteerOutcome {
  mechanism: SteeringCapability;
  /** Every correction this steer delivered, in the order the user said them. */
  delivered: GuidanceItem[];
  /** True when the running turn had to be cancelled (the in-flight step is lost). */
  cancelled: boolean;
  /** The continuation prompt sent after a cancel, when that row was taken. */
  prompt?: string;
}

/** Everything the host needs from the session that owns it. `send` must not throw. */
export interface AcpHostPorts {
  chatId: string;
  closed(): boolean;
  send(m: LiveServerMsg): void;
  /** The session's bridges (clipboard, share, emit, elicit, …), read per call. */
  session(): Session;
  /** The call's tools, read LIVE: a folder switch rebuilds them. */
  tools(): ToolSet;
  /** The call's approval policy — the same one its built-in brain runs. */
  approve: Approve;
  tally?: ToolTally;
  /** The conversation's ledger: the only source of steering text and resume state. */
  ledger: Ledger;
  /** Is a voice turn running? An ask outside one is a finished turn's. */
  turnActive(): boolean;
  /** The running voice turn's signal, which the hosted tools fall back to when no
   *  explicit coding task is running (exactly what /live bound them to before). */
  turnSignal(): AbortSignal | null;
  /** The number of the utterance the running turn answers, echoed on its events. */
  replyTurn(): number | undefined;
  /** Text history to seed the agent with, so a restart does not forget the call. */
  seed(): Message[];
  /** A cut the next turn must tell the agent about, where one is pending. */
  pendingCut(): { spoken: string; cancelled?: boolean } | null;
  /** The agent's models/modes/commands, already sent to the client. */
  onMeta(meta: AgentMetaWire): void;
  /** Persist a transcript recovered from an agent's `session/load`. */
  onReplay(messages: ReplayMessage[]): Promise<void>;
  /** This chat was empty when the resume began → persist the recovered turns. */
  onReplayExpected(wanted: boolean): void;
  /** The folder changed, so the session's tools (and its runner's) are rebuilt. */
  onFolderMoved(): void;
  onStarted(ms: number): void;
  onReady(): void;
  onStartError(e: unknown): void;
  /** How the agent's session came up, for the first turn's fact. */
  onResumed?(how: AgentResumeKind): void;
}

/**
 * Owns ACP for one conversation: binding, the hosted MCP server, asks, and
 * ordered steering. `LiveSession` delegates here; `/realtime` will too.
 */
export class AcpHost {
  private agent: Agent | null = null;
  private boundId: AgentId | null = null;
  private boundCwd = ""; // the agent's project folder (a change is a rebuild)
  /** Guards applyBind against re-entrant/overlapping binds. */
  private bindEpoch = 0;
  /** The most recent applyBind, awaited by boot() and by a queued turn. */
  private lastBind: Promise<void> = Promise.resolve();
  private agentReady: Promise<void> | null = null;
  private agentAc: AbortController | null = null;
  /** The running coding TASK's controller: what the hosted MCP tools live and die with. */
  private taskAc: AbortController | null = null;
  private task: ActiveTask | null = null;
  private mcp: ReturnType<typeof serveMcp> | null = null;
  private lastMeta: AgentMetaWire | null = null;
  private agentStartMs = 0;
  private resumed: AgentResumeKind | undefined;
  /** Agent permission asks awaiting the user, by reqId. */
  private readonly permPending = new Map<string, (optionId: string, outcome?: PermissionOutcome) => void>();
  /** Elicitations awaiting the user, by reqId and by the agent's elicitationId. */
  private readonly elicitPending = new Map<string, (a: ElicitationAnswer) => void>();
  private readonly elicitById = new Map<string, (a: ElicitationAnswer) => void>();
  /** The agent's ids for calls to the call's own tools in a STOPPED turn, refused
   *  should one land late, in the next turn. */
  private readonly deadHosted = new Set<string>();
  /** The `look` handshake: the one in-flight request, and whether its JPEG is due. */
  private lookPending: { reqId: string; resolve: (f: Frame | null) => void } | null = null;
  private awaitingLookFrame = false;
  /** Serializes concurrent `look` frame grabs (the handshake has one slot). */
  private frameChain: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(private readonly ports: AcpHostPorts) {}

  // ── what the session shows ───────────────────────────────────────────────

  get id(): AgentId | null { return this.boundId; }
  get cwd(): string { return this.boundCwd; }
  get running(): boolean { return !!this.agent; }
  get startMs(): number { return this.agentStartMs; }
  get meta(): AgentMetaWire | null { return this.lastMeta; }
  /** How the agent's session came up, carried on the first turn's fact. */
  get resumeKind(): AgentResumeKind | undefined { return this.resumed; }
  /** Resolves once the ACP handshake completes (the first turn waits on it). */
  get ready(): Promise<void> { return this.agentReady ?? Promise.resolve(); }
  /** The most recent bind, for a turn that must not race the handshake. */
  get settling(): Promise<void> { return this.lastBind; }
  /** Has any bind arrived yet? Boot restores only when none has. */
  get bindStarted(): boolean { return this.bindEpoch > 0; }

  /**
   * The signal the hosted MCP tools run under.
   *
   * A coding TASK's signal when one is running — which is the fix the build order
   * calls out, because `user_guidance` and `report_progress` have to outlive the
   * utterance that started the work, and a barge-in must not abort them.
   *
   * With no explicit task, it falls back to the CURRENT VOICE TURN's signal, which
   * is exactly what /live bound the tools to before this extraction, so a call
   * inside a turn behaves identically. With neither, the signal is aborted and the
   * call is refused: the built-in brain never makes one outside a turn, and neither
   * should an agent.
   */
  activeTaskSignal(): AbortSignal {
    if (this.taskAc) return this.taskAc.signal;
    return this.ports.turnSignal() ?? AbortSignal.abort();
  }

  /** Is a coding task running (as opposed to a single voice turn)? */
  get taskRunning(): boolean { return !!this.task && !this.task.signal.aborted; }

  /** The bind state the client's chips are held to. */
  boundState(): { agentId: AgentId | null; cwd: string; agentActive: boolean } {
    return { agentId: this.boundId, cwd: this.boundCwd, agentActive: !!this.agent };
  }

  // ── binding ──────────────────────────────────────────────────────────────

  /**
   * Boot's half of the bind: resume the persisted choice — but ONLY if the
   * client's own bind hasn't already arrived. The client re-sends its choice the
   * moment the socket opens, and on a NEW chat that message can beat this restore:
   * the restore then read stale/empty DB state, superseded the client's bind via
   * the epoch guard, and quietly reversed it — agent + folder shown in the UI,
   * nothing bound on the server.
   */
  async restore(): Promise<void> {
    if (this.bindEpoch === 0) await this.applyBind(this.ports.chatId ? boundAgent(this.ports.chatId) : null);
    else await this.lastBind; // let the in-flight client bind finish deciding
  }

  /**
   * Bind (or unbind) this conversation to a coding agent, optionally with a project
   * folder. Rebuilds + reconnects the ACP agent when the agent OR folder changes;
   * a no-op when neither did (an agent's cwd is fixed at spawn, so a folder switch
   * means a restart).
   */
  async applyBind(id: AgentId | null, cwd?: string, resumeSessionId?: string): Promise<void> {
    // Re-entrancy guard: applyBind awaits several writes, and binds can overlap (a
    // fast agent/folder switch, or a resume racing a reconnect). Stamp an epoch and
    // bail the moment a newer bind supersedes this one — otherwise two runs each
    // spawn an agent and the older one leaks an orphaned ACP child-process tree.
    const epoch = ++this.bindEpoch;
    const run = this.applyBindInner(id, cwd, resumeSessionId, epoch);
    this.lastBind = run.catch(() => {});
    await run;
    // Authoritative echo — whatever this bind attempt ended up with (including the
    // superseding-bind and no-folder paths that return early above), tell the client
    // what the session is ACTUALLY using so its chips can't drift from reality.
    if (epoch === this.bindEpoch && !this.ports.closed()) this.ports.send({ t: "bound_state", ...this.boundState() });
  }

  private async applyBindInner(id: AgentId | null, cwd: string | undefined, resumeSessionId: string | undefined, epoch: number): Promise<void> {
    const chatId = this.ports.chatId;
    // These two MUST land before agentCwd()/createBoundAgent read them back.
    if (cwd !== undefined && chatId) await setSetting(`agentCwd:${chatId}`, cwd);
    if (epoch !== this.bindEpoch) return; // superseded while awaiting
    // Commit point: only a live bind writes below. The bind is stored BEFORE the
    // session is cleared, so a bind that supersedes this one sees the new agent and
    // clears too, never finds the old agent with its session already gone.
    const agentChanged = !!chatId && id !== boundAgent(chatId); // a different agent can't load the old one's session
    if (chatId) await setBoundAgent(chatId, id);
    if (epoch !== this.bindEpoch) return;
    // Resuming one of the agent's OWN prior sessions (from History): stamp its ACP
    // session id so createBoundAgent loadSession-s it instead of starting fresh.
    if (chatId && (resumeSessionId || agentChanged)) await setSetting(acpSessionKey(chatId), resumeSessionId ?? "");
    if (epoch !== this.bindEpoch) return;
    // Canonical, per-chat→global — the SAME resolution the agent spawns in, so the
    // rebuild guard below and History grouping stay consistent.
    const effectiveCwd = chatId ? agentCwd(chatId) : "";
    // Stamp the session's agent + workspace so the History sidebar can file it
    // under agent → workspace → session.
    if (chatId) await setChatContext(chatId, id, effectiveCwd, agentChanged);
    if (epoch !== this.bindEpoch) return;
    if (id === this.boundId && effectiveCwd === this.boundCwd && this.agent) {
      // Same bind, agent already up. The client may have cleared its model/mode
      // chips (start() reuses the prewarmed socket and nulls them) — re-send the
      // last meta so they come back without a rebuild.
      if (this.lastMeta) this.ports.send({ t: "agent_meta", ...this.lastMeta });
      return;
    }
    const moved = effectiveCwd !== this.boundCwd;
    this.agentAc?.abort();
    void this.agent?.dispose();
    this.agent = null; this.agentReady = null; this.lastMeta = null;
    this.boundId = id; this.boundCwd = effectiveCwd; this.agentStartMs = 0; this.resumed = undefined;
    if (moved) this.ports.onFolderMoved();
    if (!id || this.closed || this.ports.closed() || !chatId) return;
    // A coding agent needs a real folder. Don't spawn (then instantly kill) one with
    // no cwd — that surfaced a spurious "pick a folder" error and churned processes
    // on the first bind. Wait for a bind that supplies a folder (the lobby gates Start on it).
    if (!effectiveCwd) return;
    // Decide replay-persistence NOW, before any user turn can persist a message — so a
    // user who speaks before the session/load finishes doesn't make ingestReplay think
    // the chat is OpenLive-origin and drop the recovered transcript.
    const replay = !!resumeSessionId && listMessages(chatId).length === 0;
    this.ports.onReplayExpected(replay);
    const mcp = await this.ensureMcp();
    if (epoch !== this.bindEpoch || this.closed) return;
    const agent = createBoundAgent(chatId, (q, o, toolCallId) => this.askPermission(q, o, toolCallId), {
      onMeta: (meta) => { this.lastMeta = meta; if (!this.ports.closed()) this.ports.send({ t: "agent_meta", ...meta }); },
      // Recovered transcript from a session/load — persist + tell the client to reload.
      onReplay: (msgs) => this.ports.onReplay(msgs),
      askElicitation: (req) => this.askElicitation(req),
      completeElicitation: (elicitationId) => this.elicitById.get(elicitationId)?.({ action: "accept" }),
      mcp: { wire: mcp.wire, tools: this.ports.tools().list.map((t) => t.name) },
      replay,
      onResumed: (how) => { this.resumed = how; this.ports.onResumed?.(how); },
    });
    if (!agent) return;
    this.agent = agent;
    const prior = this.ports.seed();
    if (prior.length) agent.seed(prior);
    const cut = this.ports.pendingCut();
    if (cut) agent.cut?.(cut.spoken, cut.cancelled);
    const ac = new AbortController(); this.agentAc = ac;
    const startedAt = performance.now();
    this.agentReady = agent.start(ac.signal)
      .then(() => {
        const ms = Math.round(performance.now() - startedAt);
        this.agentStartMs = ms;
        this.ports.onStarted(ms);
        if (!this.ports.closed()) this.ports.onReady();
      })
      .catch((e) => {
        if (this.ports.closed()) return;
        this.ports.onStartError(e);
      });
  }

  /**
   * ONE MCP server for this session, created lazily and reused — `serveMcp`, the
   * same one the built-in brain's tools come from. There is no second tool table.
   */
  private ensureMcp(): ReturnType<typeof serveMcp> {
    const host = this;
    return (this.mcp ??= serveMcp({
      // Read per request, so a new folder's skills are served once it is bound.
      get tools() { return host.ports.tools(); },
      // A call arriving outside a task is refused, as the built-in brain never makes
      // one, and so is one a STOPPED turn made that lands in the next.
      ctx: (agentCallId) => ({
        ...host.ports.session(),
        context: null,
        signal: agentCallId && host.deadHosted.has(agentCallId) ? AbortSignal.abort() : host.activeTaskSignal(),
      }),
      approve: host.ports.approve,
      tally: host.ports.tally,
    }));
  }

  /** OpenLive's own tools changed mid-session: tell the agent and the MCP listener. */
  toolsChanged(names: string[]): void {
    this.agent?.toolsChanged?.(names);
    void this.mcp?.then((m) => m.toolsChanged()).catch(() => {});
  }

  // ── tasks and steering ──────────────────────────────────────────────────

  /**
   * Start a coding task. `signal` is the TASK's cancellation: a barge-in on one
   * voice turn must not kill work the user already confirmed, and the hosted MCP
   * tools stay live for as long as this signal does.
   */
  startTask(taskId: string, spec: string, signal: AbortSignal): ActiveTask {
    // A task starting supersedes whatever ran before: the new signal is the one the
    // hosted tools answer to, so a late poll reports THIS task.
    this.endTask();
    const ac = new AbortController();
    this.taskAc = ac;
    if (signal.aborted) ac.abort();
    else signal.addEventListener("abort", () => ac.abort(), { once: true });
    this.task = { taskId, spec, agentId: this.boundId, signal: ac.signal, startedAt: Date.now() };
    return this.task;
  }

  /** The running task, if any. */
  currentTask(): ActiveTask | null { return this.task; }

  /** The task is over — finished, failed or cancelled. */
  endTask(): void {
    this.taskAc?.abort();
    this.taskAc = null;
    this.task = null;
  }

  /**
   * Apply a mid-task correction by whichever mechanism this agent supports
   * (Stage 0, judged on disk state rather than on what the agent says):
   *
   * | `AgentDef.acp.steering` | what `steerTask` does                                    |
   * | `meta_steering`         | `_session/steering`, when the agent advertises it          |
   * | `mcp_guidance`          | queues it; the agent drains `user_guidance` (hermes)      |
   * | `concurrent_prompt`     | not a steering row — treated as the cancel fallback        |
   * | `cancel_resume`/absent  | `session/cancel` + a ledger-derived continuation prompt   |
   *
   * The guidance is queued in the ledger FIRST and always, whatever the transport:
   * that record is what the MCP drain reads and what survives a restart. Two
   * corrections are never concatenated into one string.
   */
  async steerTask(taskId: string, guidance: GuidanceItem): Promise<SteerOutcome> {
    const ledger = this.ports.ledger;
    const queued = ledger.enqueueGuidance({ ...guidance, taskId });
    const delivered = [queued];
    const mechanism = steeringOf(this.task?.agentId ?? this.boundId);
    switch (mechanism) {
      case "meta_steering": {
        if (!this.agent) return { mechanism, delivered, cancelled: false };
        // Only legal when the agent advertises it; a refusal falls through to the
        // row that always works rather than dropping the correction.
        const ok = await this.nativeSteer(queued).catch(() => false);
        if (ok) {
          ledger.markGuidance(queued.guidanceId, "delivered_native");
          return { mechanism, delivered, cancelled: false };
        }
        return this.cancelResume(taskId, mechanism, delivered);
      }
      case "mcp_guidance": {
        // Nothing to send: the agent drains the queue itself through the hosted
        // `user_guidance` tool, which reads `drainPendingGuidance`. Marking it now
        // would hide it from that poll.
        return { mechanism, delivered, cancelled: false };
      }
      default:
        // `cancel_resume`, and `concurrent_prompt` — Stage 0 measured the latter as
        // queue-only (it does not steer), so it is NOT used as a steering mechanism.
        return this.cancelResume(taskId, mechanism, delivered);
    }
  }

  /** The `_session/steering` attempt. Resolves true only when the agent took it. */
  private async nativeSteer(item: GuidanceItem): Promise<boolean> {
    // Duck-typed: not every agent (nor every wrapper) exposes it, and it is only
    // legal where the agent advertised the capability.
    const agent = this.agent as unknown as { steer?: (text: string) => Promise<boolean> };
    if (typeof agent?.steer !== "function") return false;
    return await agent.steer(item.text);
  }

  /** Boundary-cancel: stop the running turn, then hand the agent a continuation
   *  prompt built from the ledger so it resumes instead of re-planning. */
  private async cancelResume(taskId: string, mechanism: SteeringCapability, delivered: GuidanceItem[]): Promise<SteerOutcome> {
    const ledger = this.ports.ledger;
    // Every correction still pending for this task, oldest first: a steer arriving
    // while another is unapplied must not silently drop the first.
    const pending = ledger.drainPendingGuidance(taskId);
    const applied = pending.length ? pending : delivered;
    // The digest is the ledger's own view — never the model's memory of progress.
    const prompt = continuationPrompt(ledger.buildDigest("status"), applied, this.task?.spec);
    if (!this.agent) return { mechanism, delivered: applied, cancelled: false, prompt };
    for (const g of applied) ledger.markGuidance(g.guidanceId, "applied_after_cancel");
    // ACP's own cancel: the agent resolves its turn with stopReason "cancelled".
    this.agentAc?.abort();
    // Re-prompt on the SAME session, so the agent keeps its own memory of the work
    // done. Fire-and-forget: a voice turn must never block on a coding task.
    const ac = new AbortController();
    this.agentAc = ac;
    void this.agent.runTurn({ text: prompt, frames: [] }, () => {}, ac.signal).catch(() => {});
    return { mechanism, delivered: applied, cancelled: true, prompt };
  }

  /** A correction the transport itself confirmed delivered (the MCP tool ack). */
  markDelivered(guidanceId: string, status: "delivered_native" | "delivered_mcp"): void {
    this.ports.ledger.markGuidance(guidanceId, status);
  }

  /**
   * What the hosted `user_guidance` tool returns to the agent (wired in Stage 5).
   *
   * Ordered by the ledger's own seq — the order the user said them — and marked
   * `delivered_mcp` only as they are handed over, so a poll that never happens
   * leaves the correction pending rather than silently claiming it landed.
   * Returns `NO_NEW_GUIDANCE` when there is none, which is what the Stage 0 probe
   * measured hermes against.
   */
  drainGuidance(taskId: string): { items: GuidanceItem[]; text: string } {
    const items = this.ports.ledger.drainPendingGuidance(taskId);
    for (const g of items) this.ports.ledger.markGuidance(g.guidanceId, "delivered_mcp");
    // One numbered line per correction. NEVER joined into a sentence: "do X" and
    // "actually not X" must stay two instructions in the order they were spoken.
    const text = items.length
      ? items.map((g, i) => `${i + 1}. ${g.text.trim()}`).join("\n")
      : "NO_NEW_GUIDANCE";
    return { items, text };
  }

  // ── asking ──────────────────────────────────────────────────────────────

  /** Relay an agent permission ask to the client (spoken + chips + inline on the
   *  tool card when toolCallId is known) and await the chosen option id; times
   *  out to "deny" so a hung decision never wedges a turn. */
  askPermission(question: string, options: PermissionAskOption[], toolCallId?: string): Promise<string> {
    return new Promise((resolve) => {
      if (this.closed || this.ports.closed()) { permissionFact("call", "cancelled"); return resolve("deny"); }
      // A cut or finished turn's late ask: its number is stale, so the client drops
      // it, and left pending here it would take the next utterance as its answer.
      if (!this.ports.turnActive()) { permissionFact("call", "cancelled"); return resolve(PERMISSION_CANCELLED); }
      const reqId = randomUUID();
      const expiresAt = Date.now() + PERM_TIMEOUT_MS; // client renders the countdown + speaks a reminder
      // Single settle path for every outcome (answered / auto-deny / cancelled): it
      // removes the pending entry AND tells the client the ask is resolved, so the
      // chip is dismissed and a later utterance isn't mis-read as a yes/no answer.
      const settle = (optionId: string, outcome?: PermissionOutcome) => {
        if (!this.permPending.delete(reqId)) return; // already settled
        clearTimeout(timer);
        permissionFact("call", outcome ?? askOutcome(options, optionId));
        this.ports.send({ t: "permission_resolved", reqId });
        resolve(optionId);
      };
      const timer = setTimeout(() => settle("deny", "timeout"), PERM_TIMEOUT_MS);
      this.permPending.set(reqId, settle);
      this.ports.send({ t: "permission", reqId, question, options, expiresAt, toolCallId, turn: this.ports.replyTurn() });
    });
  }

  /** Relay an agent elicitation (login URL / input form) to the client and await
   *  the user's action. Same lifecycle discipline as askPermission: one settle
   *  path, 120s timeout → cancel, dismissed client-side via elicitation_resolved.
   *  URL elicitations also register under the agent's elicitationId so an
   *  agent-side completion (OAuth landed) auto-accepts the card. */
  askElicitation(req: ElicitationAsk): Promise<ElicitationAnswer> {
    return new Promise((resolve) => {
      const inTurn = this.ports.turnActive();
      // Outside a turn only a request-scoped ask is live: ACP's scope for auth and
      // setup before any session exists. A session-scoped one then is a finished
      // turn's, and held it would take the next sentence as its answer.
      if (this.closed || this.ports.closed() || (!inTurn && !req.requestScoped)) return resolve({ action: "cancel" });
      emitFact("call", { elicitations: 1 });
      const reqId = randomUUID();
      const expiresAt = Date.now() + ELICIT_TIMEOUT_MS;
      const settle = (a: ElicitationAnswer) => {
        if (!this.elicitPending.delete(reqId)) return; // already settled
        if (req.elicitationId) this.elicitById.delete(req.elicitationId);
        clearTimeout(timer);
        this.ports.send({ t: "elicitation_resolved", reqId });
        resolve(a);
      };
      const timer = setTimeout(() => settle({ action: "cancel" }), ELICIT_TIMEOUT_MS);
      this.elicitPending.set(reqId, settle);
      if (req.elicitationId) this.elicitById.set(req.elicitationId, settle);
      // Unnumbered outside a turn, so the client shows it whichever turn it is on.
      this.ports.send({ t: "elicitation", reqId, mode: req.mode, message: req.message, url: req.url, schema: req.schema, expiresAt, turn: inTurn ? this.ports.replyTurn() : undefined });
    });
  }

  /** The client answered a permission ask. */
  answerPermission(reqId: string, optionId: string): void {
    this.permPending.get(reqId)?.(optionId); // settle() clears the map + notifies the client
  }

  /** The client answered an elicitation. */
  answerElicitation(reqId: string, action: "accept" | "decline" | "cancel", content?: Record<string, unknown>): void {
    this.elicitPending.get(reqId)?.({ action, content });
  }

  /** Answer any in-flight agent permission ask as cancelled (ACP MUST when a turn is
   *  cancelled — barge-in OR a watchdog cut). Idempotent: settle() removes each from
   *  the map (so iterate a copy) and no-ops if already settled. */
  cancelPendingPermissions(): void {
    for (const settle of [...this.permPending.values()]) settle(PERMISSION_CANCELLED);
  }

  /** Settle any in-flight elicitations as cancelled (turn end / teardown). */
  cancelPendingElicitations(): void {
    for (const settle of [...this.elicitPending.values()]) settle({ action: "cancel" });
  }

  /** A turn is over (normally, barge-in, watchdog cut or error) — never leave an
   *  agent ask dangling. */
  endTurn(): void {
    this.cancelPendingPermissions();
    this.cancelPendingElicitations();
  }

  /** A STOPPED turn's calls to the call's own tools are refused should they land
   *  in the next turn. */
  refuseLate(ids: Iterable<string>): void {
    for (const id of ids) this.deadHosted.add(id);
  }

  /** An Emit that drops an agent's report of its own use of the call's tools, and
   *  records their ids, so a late update is dropped with the call. */
  hide(emit: Emit, hidden: Set<string>): Emit {
    return hideHosted(emit, hidden);
  }

  // ── the `look` frame handshake ──────────────────────────────────────────
  /** Grab one fresh hi-res frame. SERIALIZED: the client↔server handshake has a
   *  single in-flight slot (binary frames carry no reqId), so two concurrent `look`
   *  calls (the turn-runner fans tool calls out with Promise.all) would otherwise
   *  overwrite each other's slot — the first promise would never resolve and the
   *  whole session would wedge (turnActive stuck true). Chaining makes each look
   *  wait its turn; every one resolves (frame or null on timeout). */
  requestFrame(): Promise<Frame | null> {
    const run = () => new Promise<Frame | null>((resolve) => {
      const reqId = randomUUID();
      const timer = setTimeout(() => {
        if (this.lookPending?.reqId === reqId) { this.lookPending = null; this.awaitingLookFrame = false; resolve(null); }
      }, FRAME_TIMEOUT_MS);
      this.lookPending = { reqId, resolve: (f) => { clearTimeout(timer); resolve(f); } };
      this.awaitingLookFrame = false;
      this.ports.send({ t: "need_frame", reqId });
    });
    const p = this.frameChain.then(run, run);
    this.frameChain = p.catch(() => {});
    return p;
  }

  /** The client says a frame is (or is not) coming. The JPEG itself then arrives as
   *  the next binary frame, which `onFrame` reads. */
  onFrameResponse(reqId: string, failed?: boolean): void {
    if (this.lookPending?.reqId !== reqId) return;
    if (failed) { const p = this.lookPending; this.lookPending = null; this.awaitingLookFrame = false; p.resolve(null); }
    else this.awaitingLookFrame = true;
  }

  /** A binary frame from the client. True when it was the frame `look` awaited. */
  onFrame(buf: Buffer): boolean {
    if (buf[0] !== LIVE_TAG.FRAME_IN) return false;
    if (!this.awaitingLookFrame || !this.lookPending) return false;
    const p = this.lookPending;
    this.lookPending = null; this.awaitingLookFrame = false;
    p.resolve({ data: Buffer.from(buf.subarray(1)).toString("base64"), mime: "image/jpeg" });
    return true;
  }

  // ── model / mode / option / cut ─────────────────────────────────────────

  /** Drive one turn of the bound agent. False when there is no agent to drive, so
   *  the session can say what is wrong instead of silently using another brain. */
  async runTurn(input: TurnInput, emit: Emit, signal: AbortSignal): Promise<boolean> {
    if (!this.agent) return false;
    await this.agent.runTurn(input, emit, signal);
    return true;
  }

  async setModel(modelId: string): Promise<void> { await this.agent?.setModel?.(modelId); }
  async setMode(modeId: string): Promise<void> { await this.agent?.setMode?.(modeId); }
  async setOption(optionId: string, valueId: string): Promise<void> { await this.agent?.setOption?.(optionId, valueId); }

  /** The user cut the agent's reply; a resumed session hears it on its next turn. */
  cut(spoken: string, cancelled = false): void { this.agent?.cut?.(spoken, cancelled); }

  // ── teardown ────────────────────────────────────────────────────────────

  /** The session's socket closed: refuse every ask, stop the task and the agent. */
  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    this.endTask();
    this.agentAc?.abort();
    void this.agent?.dispose();
    void this.mcp?.then((m) => m.close()).catch(() => {});
    for (const settle of [...this.permPending.values()]) settle("deny", "cancelled");
    this.cancelPendingElicitations();
    this.lookPending?.resolve(null);
  }
}

/** Which agent a conversation is bound to, for a caller that has no host yet. */
export const boundAgentLabel = (chatId: string): string => agentLabel(boundAgent(chatId));
