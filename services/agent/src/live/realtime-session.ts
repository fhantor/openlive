// File: services/agent/src/live/realtime-session.ts
// WHY: the agent-side session of a NATIVE Gemini Live call. `/live` is a text proxy; `/realtime`
// is the audio one, so its session owns exactly what a native-audio call needs and nothing of
// what the renderer already owns (VAD, playback, AEC):
//
//   * the BINARY transport — 16 kHz PCM16 up, `[u32le generation]`+24 kHz PCM16 down, where the
//     generation counter IS the barge-in contract (01-CONTRACTS.md rule 1);
//   * the SIX Live tools, every one of which returns from memory in well under 50 ms. Live
//     function calling is SEQUENTIAL: a handler that awaited the coding agent would freeze the
//     whole conversation, so no handler here ever awaits one — a dispatch is recorded and
//     started with `void`, and the answer is already written by then;
//   * the confirmation gate's session half — the model NOMINATES (`confirm_instruction`) and this
//     file only lets that nomination through once `draft-gate.ts` has validated the person's own
//     transcript. Two keys, never one;
//   * the four pushed events (`task_finished`, `needs_input`, `task_failed`, `steer_applied`),
//     delivered only when Gemini is idle, waking a sleeping socket when needed. Progress is
//     otherwise PULLED through `get_status`;
//   * persistence: clean user/assistant transcript and the coding agent's tool cards through
//     `addMessage`. `[LEDGER_COLD_START]` / `[STATE_SYNC]` are seeded into the model and are
//     NEVER persisted (01-CONTRACTS.md rule 5).
//
// This file knows nothing about ACP internals and nothing about the browser: ACP is `AcpHost`
// (Stage 4), the socket is `GeminiBidiClient` (Stage 3), the truth is `Ledger` (Stage 2), and the
// gate is `DraftGate` (Stage 2).

import { createHash, randomUUID } from "node:crypto";
import type { WebSocket } from "ws";
import {
  GEMINI_TOOL_NAMES,
  isAgentId,
  withReplyLanguage,
  type ActiveDraftView,
  type GuidanceItem,
  type LanguageCode,
  type LiveServerMsg,
  type MessageBlock,
  type RealtimeClientMsg,
  type RealtimePhase,
  type RealtimeServerMsg,
  type TaskDigestView,
} from "@openlive/shared";
import { addMessage, createChat, listMessages } from "@openlive/db";
import { layout, resolveHome } from "@openlive/shared/home";
import type { Message } from "@openlive/harness";
import { AcpHost, voiceSummary, type AcpHostPorts } from "./acp-host.js";
import { DraftGate, type DispatchTicket } from "./draft-gate.js";
import {
  createGeminiLiveClient,
  type BidiPhase,
  type GeminiBidiClient,
  type GeminiBidiConfig,
  type GeminiContentTurn,
  type GeminiFunctionCall,
  type GeminiFunctionResponse,
} from "./gemini-bidi.js";
import { Ledger, speakable } from "./ledger.js";
import { foldBlock } from "../block-emit.js";
import { ToolSet } from "../capabilities/dispatch.js";
import { allowAll } from "../capabilities/approval.js";
import type { Approve, Emit, Session as ToolSession, Tool } from "../capabilities/types.js";
import type { ReplayMessage } from "../agents/index.js";
import { log } from "../log.js";

// ── the wire constants (01-CONTRACTS.md, B in 00-BUILD-ORDER.md) ────────────

/** The one pathname `ws.ts` routes here. */
export const REALTIME_PATH = "/realtime";
/** Client → server: raw PCM16LE mono 16000 Hz, 60–100 ms per frame, NO header. */
export const REALTIME_IN_RATE = 16_000;
/** Server → client: raw PCM16LE mono 24000 Hz, behind a 4-byte generation. */
export const REALTIME_OUT_RATE = 24_000;
/** A Live tool handler must return from memory inside this budget. It is a bound, not a target:
 *  every handler below is plain synchronous object building, so the real number is microseconds. */
export const TOOL_BUDGET_MS = 50;
/** Pushed events are stated to the model behind this marker, never persisted. */
export const EVENT_PREFIX = "[ORCHESTRATOR EVENT]";
/** A transcript row is bounded like any other message. */
const MAX_TRANSCRIPT_CHARS = 4_000;
/** Recent stored turns seeded into a resumed agent's context. */
const SEED_TURNS = 20;

/** The one binary frame shape of this socket: `[4-byte uint32LE generation][PCM16LE 24 kHz]`. */
export function audioFrame(gen: number, pcm16: Buffer): Buffer {
  const generation = Buffer.allocUnsafe(4);
  generation.writeUInt32LE(gen >>> 0, 0);
  return Buffer.concat([generation, pcm16]);
}

/** Text that belongs to the MODEL's context and never to the person's chat history. A digest
 *  that reached the speech channel must not become a stored turn, so the whole line is dropped
 *  rather than echoed as a fragment. The header carries a payload inside its brackets
 *  (`[STATE_SYNC resume=ok checkpoint_seq=142]`), so the match is on the opening token. */
const INTERNAL_TAG = /\[(?:LEDGER_COLD_START|STATE_SYNC|SESSION_INIT|CONTINUED|ORCHESTRATOR EVENT)(?:\]|\s)/;

/** Strip internal-state lines from anything about to be persisted or said. */
export function cleanForChat(text: string): string {
  return String(text ?? "")
    .split(/\r?\n/)
    .filter((line) => !INTERNAL_TAG.test(line))
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join("\n")
    .trim();
}

/** Blocks with their internal-state text removed, everything else kept (tool cards, diffs). */
export function cleanBlocks(blocks: readonly MessageBlock[]): MessageBlock[] {
  const out: MessageBlock[] = [];
  for (const b of blocks) {
    if (b.type === "text" || b.type === "reasoning") {
      const text = cleanForChat(b.text);
      if (text) out.push({ ...b, text });
      continue;
    }
    out.push(b);
  }
  return out;
}

/**
 * Rule 4 of 01-CONTRACTS.md: computed ONCE at propose time from
 * `(chatId, normalizedSpec, parentTaskId, draftId)` and written to the ledger as `task_dispatched`
 * BEFORE the ACP call, so a reconnect can never produce a second dispatch for the same key.
 * Not at dispatch time — computing it late is what lets a replay recompute a different key.
 */
export function idempotencyKeyOf(input: {
  chatId: string;
  englishSpec: string;
  parentTaskId: string | null;
  draftId: string;
}): string {
  const spec = String(input.englishSpec ?? "").toLowerCase().replace(/\s+/g, " ").trim();
  return createHash("sha256")
    .update(`${input.chatId}\u0000${spec}\u0000${input.parentTaskId ?? ""}\u0000${input.draftId}`)
    .digest("hex")
    .slice(0, 32);
}

/** The orchestrator protocol. Configuration is immutable while the socket is open, so this is
 *  fixed text: the rules that keep a spoken "sent it" from existing where no dispatch happened. */
export const ORCHESTRATOR_PROTOCOL = [
  "You are the VOICE of OpenLive's realtime orchestrator. The person speaks to you natively and you",
  "answer with your own voice. You never run code, never edit files and never claim work you did not do.",
  "Speak Bengali by default and mirror the person's language. Keep answers to one or two short sentences.",
  "",
  "An instruction reaches the coding agent only through all three steps, in order:",
  "1. propose_instruction — stage the draft (a Bengali summary plus the exact English imperative).",
  "   Nothing is sent at this point. Never say otherwise.",
  "2. request_confirmation — call it exactly as you read the summary aloud and ask whether to send.",
  "   End that read-back on an OPEN question, never on an affirmative like ঠিক আছে.",
  "3. confirm_instruction — nominate the confirmation only after the person clearly says yes / পাঠাও.",
  "   The orchestrator independently checks their own transcript: a yes with a condition, a question,",
  "   a backchannel, an echo of your own read-back, or a 'yes' that answers your question all fail.",
  "   If it returns dispatched:true the task really started. Anything else means it was NOT sent —",
  "   say so honestly and ask again. Never invent a dispatch.",
  "",
  "Progress must be PULLED, never pushed: call get_status before answering ANY question about what is",
  "happening, then say the current step and the last completed ones in plain speech. Never read file",
  "paths, commands, code, JSON or ledger text aloud.",
  "cancel_task only when the person clearly asks to stop. answer_agent only while a permission or",
  "elicitation request is pending, and only with the request_id you were given.",
  "",
  `Lines beginning ${EVENT_PREFIX} are trustworthy state changes. State them briefly; do not embellish.`,
].join("\n");

/** What `createGeminiLiveClient` takes: the key is resolved from the repo's own resolver. */
export type RealtimeClientConfig = Omit<GeminiBidiConfig, "apiKey" | "model"> & { model?: string; voiceName?: string };

/** Test/diagnostic seams. Production passes nothing: every real dependency is the committed one. */
export interface RealtimeDeps {
  createClient?: (cfg: RealtimeClientConfig) => Promise<GeminiBidiClient>;
  host?: AcpHost;
  ledger?: Ledger;
  gate?: DraftGate;
  now?: () => number;
}

/** A request the coding agent is waiting on. `answer_agent` is only valid against one of these. */
interface PendingAsk {
  kind: "permission" | "elicitation";
  options: Array<{ id: string; label: string; kind?: string }>;
}

/** The four events that are pushed. Everything else about progress is pulled. */
type PushedEvent = "task_finished" | "needs_input" | "task_failed" | "steer_applied";

interface QueuedEvent {
  kind: PushedEvent;
  text: string;
}

/**
 * One `/realtime` voice call: the native-audio socket, the ledger, the confirmation gate and the
 * coding agent, tied together. Construct it from `ws.ts` and `start()` it; nothing else calls in.
 */
export class RealtimeSession {
  /** The live session of each chat. One socket per chat, and a coding task can outlive the socket
   *  that started it (Stage 9's reconnect case), so a second socket for the same chat must find it. */
  private static readonly sockets = new Map<string, RealtimeSession>();
  /** A chat whose previous socket died mid-task: the task is still running under that session. */
  private static readonly continuing = new Map<string, RealtimeSession>();

  private readonly ws: WebSocket;
  private readonly chatId: string;
  private readonly lang?: LanguageCode;
  private readonly device: boolean;
  private readonly deps: RealtimeDeps;
  private readonly now: () => number;

  private readonly ledger: Ledger;
  private readonly gate: DraftGate;
  private readonly host: AcpHost;
  private readonly toolSession: ToolSession;
  private readonly approve: Approve;
  private readonly tools: ToolSet;

  private gemini: GeminiBidiClient | null = null;
  private phase: RealtimePhase = "connecting";
  /** The barge-in contract: every outbound audio frame is tagged with it. */
  private gen = 0;
  private closed = false;
  private startup: Promise<void> = Promise.resolve();

  /** idempotencyKey computed at propose time, by draftId. */
  private readonly draftKeys = new Map<string, string>();
  /** A model nomination seen before key 2 agreed. Never dispatches on its own. */
  private nominated: { draftId: string; version: number } | null = null;

  /** The task's own cancellation — a barge-in on one utterance must never kill it. */
  private taskAc: AbortController | null = null;
  /** How each task it started ended, so a task_status is honest before the ledger folds it. */
  private readonly taskOutcome = new Map<string, TaskDigestView["status"]>();
  /** Tasks already asked to stop: `cancel_task` is idempotent, never a second cancel. */
  private readonly cancelled = new Set<string>();
  /** True while the socket is watching a task its previous socket started. */
  private waitingOnOrphan = false;

  /** Transcript accumulation: Gemini sends incremental transcription chunks. */
  private inputBuf = "";
  private assistantBuf = "";
  private transcriptSeq = 0;
  /** What the renderer said about the person's speech onset, for the gate's echo check. */
  private lastOnsetMs = 0;
  private lastPlaybackEndMs: number | undefined;
  private lastEchoSafe = true;
  /** True once the renderer reported speech, until it reports the utterance ended.
   *  One `audioStreamEnd` per utterance closes the Gemini turn; without it the model
   *  hears everything and answers nothing (measured 2026-11-20). */
  private spokeSinceIdle = false;
  /** PCM bytes received since the last audioStreamEnd; a non-zero count means there is
   *  an utterance to close even if no speaking:true transition was ever reported. */
  private micBytesSinceEnd = 0;

  /** Permission/elicitation asks the coding agent is waiting on, by wire id. */
  private readonly pendingAsks = new Map<string, PendingAsk>();
  /** The four pushed events, waiting for Gemini to be idle. */
  private pendingEvents: QueuedEvent[] = [];
  private delivering = false;
  /** The last checkpoint handle written to the ledger, so each one is written once. */
  private lastCheckpointHandle = "";
  /** How many sockets this session has had: the first is cold, every later one resumes. */
  private connects = 0;

  // ── the SIX Live tools ───────────────────────────────────────────────────
  // One handler per declaration, keyed by the shared name list so a missing tool is a compile
  // error rather than a silent "unknown tool" at 2 a.m. Every handler is SYNCHRONOUS: building
  // its answer from memory and returning. That is the whole non-awaiting discipline, in the type.
  private readonly handlers: Record<
    (typeof GEMINI_TOOL_NAMES)[number],
    (args: Record<string, unknown>) => Record<string, unknown>
  > = {
    propose_instruction: (a) => this.toolPropose(a),
    request_confirmation: (a) => this.toolRequestConfirmation(a),
    confirm_instruction: (a) => this.toolConfirmInstruction(a),
    get_status: (a) => this.toolGetStatus(a),
    cancel_task: (a) => this.toolCancelTask(a),
    answer_agent: (a) => this.toolAnswerAgent(a),
  };

  constructor(ws: WebSocket, chatId: string, lang?: LanguageCode, device = false, deps: RealtimeDeps = {}) {
    this.ws = ws;
    this.chatId = chatId;
    this.lang = lang;
    this.device = device;
    this.deps = deps;
    this.now = deps.now ?? (() => Date.now());

    this.ledger = deps.ledger ?? new Ledger(chatId, layout(resolveHome()).data, this.now);
    this.gate = deps.gate ?? new DraftGate({ now: this.now });

    // What a realtime call can reach. The renderer owns audio and playback, so there is no camera,
    // clipboard or device port to serve here: the hosted MCP server publishes only the tools below.
    this.toolSession = { workspace: () => this.host.cwd };
    this.approve = allowAll;
    this.tools = this.buildTools();
    this.host = deps.host ?? new AcpHost(this.ports());

    ws.on("message", (data: Buffer, isBinary: boolean) => {
      try {
        if (isBinary) this.onBinary(data);
        else this.onText(data.toString());
      } catch (e) {
        log.error("realtime", "inbound frame:", e);
      }
    });
    ws.on("close", () => this.dispose());
    ws.on("error", () => this.dispose());
  }

  // ── lifecycle ───────────────────────────────────────────────────────────

  start(): Promise<void> {
    this.startup = this.boot().catch((e) => {
      log.error("realtime", "boot:", e);
      this.setPhase("error", "boot_failed");
      this.send({ t: "error", code: "realtime_boot_failed", message: String((e as Error)?.message ?? e) });
    });
    return this.startup;
  }

  private async boot(): Promise<void> {
    this.setPhase("connecting");
    log.debug("realtime", `boot chat=${this.chatId || "(none)"} device=${this.device ? 1 : 0}`);
    if (this.chatId) {
      try {
        await createChat(this.chatId);
      } catch (e) {
        log.warn("realtime", "create chat:", e);
      }
      // A previous socket for this chat may still be finishing a coding task. It keeps running;
      // this socket watches the ledger instead of spawning a second agent on the same project.
      const continuing = RealtimeSession.continuing.get(this.chatId);
      if (continuing && continuing !== this && continuing.host.currentTask()) {
        this.waitingOnOrphan = true;
        log.warn("realtime", `chat ${this.chatId} reconnected while its task still runs`);
      }
    }
    if (!this.waitingOnOrphan) await this.host.restore().catch((e) => log.warn("realtime", "restore bind:", e));
    if (this.closed) return;
    if (this.chatId) RealtimeSession.sockets.set(this.chatId, this);

    this.send({ t: "bound_state", ...this.boundState() });
    this.send({ t: "draft_state", draft: null });
    this.send({ t: "task_status", task: null });

    const make = this.deps.createClient ?? ((cfg: RealtimeClientConfig) => createGeminiLiveClient(cfg));
    try {
      this.gemini = await make(this.clientConfig());
    } catch (e) {
      this.setPhase("error", "client_config");
      this.send({ t: "error", code: "gemini_config_failed", message: String((e as Error)?.message ?? e) });
      return;
    }
    this.wireClient(this.gemini);
    try {
      await this.gemini.connect();
      this.setPhase("live");
    } catch (e) {
      this.setPhase("error", "connect_failed");
      this.send({ t: "error", code: "gemini_connect_failed", message: String((e as Error)?.message ?? e) });
      return;
    }
    this.flushEvents();
  }

  private clientConfig(): RealtimeClientConfig {
    return {
      // `?lang=` is the session's language; the model is told it in words, as `/live` does, because
      // the Live setup message takes no language hint.
      systemInstruction: withReplyLanguage(ORCHESTRATOR_PROTOCOL, this.lang),
      // A resumable handle survives the socket, and the ledger is where it survives a restart.
      resumeHandle: this.ledger.getCheckpoint()?.handle ?? "",
      silenceDurationMs: 900,
      // Seeded ONLY on a cold connect. A digest, never a persisted chat message.
      getColdHistory: () => this.coldHistory(),
    };
  }

  /** `[LEDGER_COLD_START]` — sent to the model, never written to `addMessage`. The trailing model
   *  turn is what lets the server treat the whole block as history under
   *  `initialHistoryInClientContent` instead of answering it. */
  private coldHistory(): GeminiContentTurn[] {
    return [
      { role: "user", parts: [{ text: this.ledger.buildDigest("cold_header") }] },
      { role: "model", parts: [{ text: "Understood. I keep my answers short, in the person's language." }] },
    ];
  }

  private wireClient(client: GeminiBidiClient): void {
    client.on("audio", (pcm: Buffer) => this.sendAudio(pcm));
    client.on("interrupted", () => {
      // The server stopped generating: everything already queued for playback is stale, so the
      // generation moves and the renderer drops the frames still in flight.
      this.bumpGeneration("gemini_interrupted");
      this.send({ t: "phase", phase: "live", epoch: this.gen, reason: "interrupted" });
    });
    client.on("inputTranscription", (e: { text: string; finished: boolean }) => this.onInputTranscription(e));
    client.on("outputTranscription", (e: { text: string; finished: boolean }) => this.onOutputTranscription(e));
    client.on("turnComplete", () => {
      this.flushInput();
      this.flushAssistant();
      this.flushEvents();
    });
    client.on("generationComplete", () => this.flushEvents());
    client.on("phase", (e: { phase: BidiPhase; reason?: string }) => {
      this.setPhase(mapPhase(e.phase), e.reason);
      if (e.phase === "live") this.flushEvents();
    });
    client.on("toolCall", (e: { calls: GeminiFunctionCall[]; socketSeq: number }) => this.onToolCalls(e));
    client.on("sessionResumptionUpdate", () => this.checkpoint());
    client.on("setupComplete", () => {
      this.connects += 1;
      // A swapped socket must be told what it missed — silently, with `turnComplete:false`.
      if (this.connects > 1) this.sendResumeSync();
    });
    client.on("resumptionContextLost", (e: unknown) => log.warn("realtime", "resumption context lost:", e));
    client.on("error", (e: unknown) => {
      this.send({ t: "error", code: "gemini_error", message: String((e as Error)?.message ?? e) });
    });
    client.on("resumeFailed", (e: unknown) => log.warn("realtime", "resume failed, falling back cold:", e));
  }

  /** The resume header, through the one channel that does not make a native-audio model speak. */
  private sendResumeSync(): void {
    const from = this.ledger.getCheckpoint()?.seq ?? 0;
    const delta = this.ledger.buildDigest("resume_delta", from); // `[STATE_SYNC …]`, never persisted
    this.gemini?.sendSilentSync([{ role: "user", parts: [{ text: delta }] }]);
  }

  /** Only a `resumable:true` handle reaches here (the client filters the mid-work ones itself). */
  private checkpoint(): void {
    const client = this.gemini;
    if (!client) return;
    const handle = client.currentResumeHandle();
    if (!handle || handle === this.lastCheckpointHandle) return;
    this.lastCheckpointHandle = handle;
    this.ledger.recordCheckpoint(handle, client.currentCheckpointSeq());
  }

  // ── inbound ────────────────────────────────────────────────────────────

  /** Client → server binary is raw 16 kHz PCM16LE. The client coalesces it to 60–100 ms; this
   *  forwards whole chunks and never re-frames them itself (the rollback rule). */
  private onBinary(buf: Buffer): void {
    if (this.closed || !buf.length) return;
    this.micBytesSinceEnd += buf.length;
    this.gemini?.pushMicPcm16(buf);
  }

  private onText(str: string): void {
    let raw: Record<string, unknown>;
    try {
      raw = JSON.parse(str) as Record<string, unknown>;
    } catch {
      return;
    }
    const msg = raw as unknown as RealtimeClientMsg;
    switch (msg.t) {
      case "barge_in": {
        // The person started talking over the model. The renderer has already flushed playback;
        // moving the generation is what stops the frames still in flight from leaking back.
        this.bumpGeneration("client_barge_in");
        this.send({ t: "barge_in_ack", epoch: this.gen });
        return;
      }
      case "mute": {
        this.gemini?.setMuted(Boolean(msg.muted));
        return;
      }
      case "vad_state": {
        this.lastOnsetMs = Number.isFinite(msg.onsetMs) ? Number(msg.onsetMs) : 0;
        this.lastPlaybackEndMs = typeof msg.playbackEndMs === "number" ? msg.playbackEndMs : undefined;
        this.lastEchoSafe = msg.echoSafe !== false;
        // The renderer's VAD says the utterance is over. Tell Gemini so, or the turn never
        // closes: measured 2026-11-20, a continuous mic stream with automatic activity
        // detection reports ACTIVITY_START and then waits for an end that never arrives, so
        // the model never answers — no transcript, no tool call, no audio at all.
        // `audioStreamEnd` is the documented client signal for "that was the whole utterance".
        if (msg.speaking) {
          this.spokeSinceIdle = true;
        } else if (this.spokeSinceIdle || this.micBytesSinceEnd > 0) {
          // Close the utterance whenever audio actually arrived since the last close —
          // not only when a speaking:true transition was seen. A renderer that streams
          // PCM and reports only the end of speech must still get its turn answered.
          this.spokeSinceIdle = false;
          this.micBytesSinceEnd = 0;
          this.gemini?.sendAudioStreamEnd();
        }
        return;
      }
      case "draft_action": {
        this.onDraftAction(msg);
        return;
      }
      case "bind": {
        // The wire says string|null; only a known agent id may reach the bind.
        const agentId = isAgentId(msg.agentId) ? msg.agentId : null;
        void this.host
          .applyBind(agentId, msg.cwd, msg.resumeSessionId)
          .catch((e) => log.warn("realtime", "applyBind:", e));
        return;
      }
      case "permission_response": {
        this.pendingAsks.delete(msg.id);
        this.host.answerPermission(msg.id, msg.optionId);
        return;
      }
      case "elicitation_response": {
        this.pendingAsks.delete(msg.id);
        this.host.answerElicitation(msg.id, msg.action, msg.content as Record<string, unknown> | undefined);
        return;
      }
      case "control": {
        if (msg.action === "end") this.dispose();
        else if (msg.action === "sleep") void this.gemini?.sleep("client_control").catch(() => {});
        else if (msg.action === "wake") void this.gemini?.wake().catch(() => {});
        return;
      }
      default:
        return;
    }
  }

  // ── outbound ───────────────────────────────────────────────────────────

  private setPhase(phase: RealtimePhase, reason?: string): void {
    if (this.phase === phase && !reason) return;
    this.phase = phase;
    this.send({ t: "phase", phase, epoch: this.gen, ...(reason ? { reason } : {}) });
  }

  /** The barge-in contract: a monotonic generation the renderer gates playback on. */
  private bumpGeneration(reason: string): void {
    this.gen = (this.gen + 1) >>> 0;
    log.debug("realtime", `generation ${this.gen} (${reason})`);
  }

  /** `[4-byte uint32LE generation][PCM16LE 24 kHz]` — the only binary this socket sends. Audio
   *  belongs to the socket that owns the model: unlike JSON, it is never re-routed to a newly
   *  connected one (that call is a different session with its own generation counter). */
  private sendAudio(pcm: Buffer): void {
    if (this.closed || !pcm.length || !this.open()) return;
    try {
      this.ws.send(audioFrame(this.gen, pcm));
    } catch (e) {
      log.warn("realtime", "audio send:", e);
    }
  }

  /** Every JSON message. A socket that already died while its coding task runs routes to the
   *  chat's CURRENT socket instead of nowhere, so a reconnect still sees the progress. */
  private send(m: RealtimeServerMsg): void {
    const target = this.socketForSend();
    if (!target || !target.open()) return;
    try {
      target.ws.send(JSON.stringify(m));
    } catch (e) {
      log.warn("realtime", "send:", e);
    }
  }

  private open(): boolean {
    return this.ws.readyState === this.ws.OPEN;
  }

  private socketForSend(): RealtimeSession | null {
    if (!this.closed) return this;
    return this.chatId ? RealtimeSession.sockets.get(this.chatId) ?? null : null;
  }

  /** LiveServerMsg (the ACP vocabulary) → RealtimeServerMsg (this socket's vocabulary). */
  private forwardLive(m: LiveServerMsg): void {
    switch (m.t) {
      case "sse":
        this.send({ t: "sse", event: m.event as unknown as Record<string, unknown> });
        return;
      case "permission": {
        const options = m.options.map((o) => ({ id: o.id, label: o.label, ...(o.kind ? { kind: o.kind } : {}) }));
        this.pendingAsks.set(m.reqId, { kind: "permission", options });
        this.send({ t: "permission", id: m.reqId, question: m.question, options: options.map((o) => ({ id: o.id, label: o.label })) });
        // Only now is `answer_agent` legal, and only now does the person need to be asked.
        this.pushTaskEvent("needs_input", `${m.question} — request_id=${m.reqId}`);
        return;
      }
      case "permission_resolved":
        this.pendingAsks.delete(m.reqId);
        return; // the renderer dismisses its chip; no further wire message of ours
      case "elicitation": {
        this.pendingAsks.set(m.reqId, { kind: "elicitation", options: [] });
        this.send({ t: "elicitation", id: m.reqId, mode: m.mode, message: m.message, url: m.url, schema: m.schema });
        this.pushTaskEvent("needs_input", `${m.message} — request_id=${m.reqId}`);
        return;
      }
      case "elicitation_resolved":
        this.pendingAsks.delete(m.reqId);
        return;
      case "bound_state":
        this.send({ t: "bound_state", agentId: m.agentId, cwd: m.cwd });
        return;
      case "agent_meta": {
        // The meta payload is open-ended; `t` is this socket's discriminator, not the agent's.
        const { t: _t, ...meta } = m;
        this.send({ t: "agent_meta", ...meta });
        return;
      }
      case "error":
        this.send({ t: "error", code: m.code ?? "agent_error", message: m.message });
        return;
      default:
        // tool_bridge / need_frame / reminder / flow belong to `/live`'s renderer bridge. A
        // realtime socket has none of those ports, so silence is the honest answer.
        return;
    }
  }

  private boundState(): { agentId: string | null; cwd?: string } {
    const state = this.host.boundState();
    return { agentId: state.agentId, ...(state.cwd ? { cwd: state.cwd } : {}) };
  }

  // ── transcription + the gate's session half ────────────────────────────

  private onInputTranscription(e: { text: string; finished?: boolean }): void {
    if (this.closed) return;
    this.inputBuf += e.text;
    // Gemini does NOT reliably set `finished:true` — measured 2026-11-20 on this key:
    // every inputTranscription frame arrived with finished=false, and the turn then
    // ended (toolCall / turnComplete). Gating the flush on `finished===true` alone
    // therefore swallowed EVERY user transcript: no caption, no persistence, and —
    // worse — key 2 never ran, so a spoken "হ্যাঁ, পাঠাও" could never validate a
    // draft. Flush on a real turn boundary as well (turnComplete/generationComplete
    // call flushInput), which is what the ACP turn actually is.
    if (e.finished === true) this.flushInput();
  }

  private flushInput(): void {
    const text = cleanForChat(this.inputBuf);
    this.inputBuf = "";
    if (!text) return;
    const seq = ++this.transcriptSeq;
    this.send({ t: "transcript", role: "user", text, final: true, seq });
    void this.persist("user", text);
    this.checkConfirmation(text, seq);
  }

  private onOutputTranscription(e: { text: string }): void {
    if (this.closed) return;
    this.assistantBuf += e.text;
  }

  /** The read-back, the answer, the model's own questions: persisted only when a turn ends so a
   *  row is never a partial sentence. */
  private flushAssistant(): void {
    const text = cleanForChat(this.assistantBuf);
    this.assistantBuf = "";
    if (!text) return;
    const seq = ++this.transcriptSeq;
    this.send({ t: "transcript", role: "assistant", text, final: true, seq });
    void this.persist("assistant", text);
  }

  /**
   * KEY 2. The person's own transcript against the armed read-back. This function can move the
   * draft to `verifying`; it can NEVER dispatch by itself — `confirm_instruction` (key 1) or the
   * DraftCard (key 3) still has to arrive, and only `dispatchDraft`'s CAS lets one of them commit.
   */
  private checkConfirmation(text: string, seq: number): void {
    const armed = this.gate.view();
    if (!armed || armed.state === "proposing" || armed.state === "expired") return; // a "yes" answering a question
    if (!this.lastEchoSafe) return; // the renderer says the mic may be hearing our own read-back
    const { verdict, draft } = this.gate.onUserTranscript(text, seq, this.lastOnsetMs, this.lastPlaybackEndMs);
    if (draft) this.broadcastDraft();
    if (!verdict.ok) {
      log.debug("realtime", `confirmation rejected: ${verdict.reason}`);
      return;
    }
    if (draft?.state !== "verifying") return;
    // Key 2 agreed. If the model already nominated this exact version, both keys are in.
    const nom = this.nominated;
    if (nom && nom.draftId === draft.draftId && nom.version === draft.version) {
      this.nominated = null;
      this.tryDispatch(draft.draftId, draft.version, "voice");
    }
  }

  // ── the six tools (all synchronous, all from memory) ────────────────────

  private onToolCalls(e: { calls: GeminiFunctionCall[]; socketSeq: number }): void {
    // The user's utterance for this turn is complete by the time the model acts on it —
    // Gemini ends the input transcript at the tool call, and (measured) never sets
    // finished:true. Flushing here is what lets key 2 see the words, and what makes the
    // user's own turn persist at all.
    this.flushInput();
    const responses: GeminiFunctionResponse[] = [];
    const started = this.now();
    for (const call of e.calls) responses.push(this.runTool(call));
    const spent = this.now() - started;
    // A sequential function-calling model is frozen for this long. It must stay a rounding error;
    // anything near the budget means a handler started awaiting something.
    if (spent > TOOL_BUDGET_MS) log.warn("realtime", `tool handlers took ${spent}ms — a Live handler must not await`);
    this.gemini?.sendToolResponses(responses, e.socketSeq);
  }

  private runTool(call: GeminiFunctionCall): GeminiFunctionResponse {
    const handler = this.handlers[call.name as (typeof GEMINI_TOOL_NAMES)[number]];
    if (typeof handler !== "function") {
      return { id: call.id, name: call.name, response: { ok: false, error: `Unknown tool "${call.name}".` } };
    }
    let response: Record<string, unknown>;
    try {
      response = handler(call.args ?? {});
    } catch (e) {
      response = { ok: false, error: String((e as Error)?.message ?? e) };
    }
    return { id: call.id, name: call.name, response };
  }

  /** Stage a draft. Explicitly does NOT dispatch — the model has no dispatch tool at all. */
  private toolPropose(args: Record<string, unknown>): Record<string, unknown> {
    const summaryBn = String(args.summary_bn ?? "").trim();
    const englishSpec = String(args.english_spec ?? "").trim();
    if (!englishSpec) return { ok: false, error: "english_spec is required." };
    const steering = args.target === "steer_active_task";
    const parentTaskId = steering ? this.host.currentTask()?.taskId ?? null : null;
    if (steering && !parentTaskId) {
      // A steer corrects work that is still running; with no task there is nothing to steer.
      return { ok: false, reason: "no_active_task", note: "Nothing is running. Stage a new task with target new_task." };
    }
    const intentType: ActiveDraftView["intentType"] = steering
      ? args.steer_kind === "steer_pivot" ? "steer_pivot" : "steer_add"
      : "new_task";
    // One active draft per chat: a re-propose supersedes (version bumps) instead of arming a second.
    const existing = this.gate.view();
    const draftId = existing && existing.state !== "expired" ? existing.draftId : `draft_${randomUUID()}`;
    const draft = this.gate.propose({
      draftId,
      summaryBn: summaryBn || englishSpec,
      englishSpec,
      intentType,
      openQuestions: Array.isArray(args.open_questions) ? args.open_questions.map((q) => String(q)).filter(Boolean) : [],
      parentTaskId,
    });
    // Rule 4: computed HERE, at propose time.
    this.draftKeys.set(draft.draftId, this.keyOf(draft.draftId, englishSpec, parentTaskId));
    this.nominated = null;
    this.broadcastDraft();
    return {
      ok: true,
      draft_id: draft.draftId,
      version: draft.version,
      state: draft.state,
      dispatched: false,
      note: "Staged only — nothing has been sent. Read the summary back and call request_confirmation.",
    };
  }

  /** Speak the read-back and arm the gate. Key 2 can only match after this. */
  private toolRequestConfirmation(args: Record<string, unknown>): Record<string, unknown> {
    const draftId = String(args.draft_id ?? "");
    const version = Number(args.version ?? 0);
    const readback = String(args.readback_bn ?? "").trim();
    const armed = this.gate.requestConfirmation(draftId, version, readback, this.now());
    if (!armed) {
      return { ok: false, error: "No such draft version.", note: "Call propose_instruction first (or re-propose a newer version)." };
    }
    this.broadcastDraft();
    return { ok: true, draft_id: draftId, version: armed.version, state: armed.state, armed: true };
  }

  /**
   * KEY 1 — a NOMINATION and nothing else. It is only let through once key 2 has already put the
   * draft in `verifying`; from `confirming` it is remembered and refused, which is what makes a
   * model that "decides" the person agreed unable to dispatch anything.
   */
  private toolConfirmInstruction(args: Record<string, unknown>): Record<string, unknown> {
    const draftId = String(args.draft_id ?? "");
    const version = Number(args.version ?? 0);
    const view = this.gate.view();
    if (!view || view.draftId !== draftId || view.version !== version) {
      return { ok: false, dispatched: false, error: "No such draft version.", note: "Nothing was sent." };
    }
    const blocked = this.dispatchBlocked();
    if (blocked) return { ok: false, dispatched: false, reason: blocked, note: "Nothing was sent." };
    if (view.state === "confirming") {
      this.nominated = { draftId, version };
      return {
        ok: false,
        dispatched: false,
        reason: "awaiting_spoken_confirmation",
        note: "Nomination recorded. The person's own yes is still required — do NOT claim it was sent.",
      };
    }
    if (view.state !== "verifying") {
      return { ok: false, dispatched: false, reason: view.state, note: "Nothing was sent." };
    }
    if (!this.gate.confirmInstruction(draftId, version)) {
      return { ok: false, dispatched: false, reason: "not_armed", note: "Nothing was sent." };
    }
    const ticket = this.gate.dispatchDraft(draftId, version, "voice");
    if (!ticket) return { ok: true, dispatched: false, reason: "already_dispatched", note: "It was already sent; do not send it again." };
    const out = this.dispatchTicket(ticket);
    this.broadcastDraft();
    this.broadcastTask();
    return out.duplicate
      ? { ok: true, dispatched: false, reason: "already_dispatched", note: "It was already sent; do not send it again." }
      : { ok: true, dispatched: true, task_id: out.taskId, note: "The coding agent really started." };
  }

  /** PULLED progress. Synchronous, ledger-only: it must never touch the coding agent. */
  private toolGetStatus(args: Record<string, unknown>): Record<string, unknown> {
    const digest = this.ledger.buildDigest("status");
    const taskId = this.host.currentTask()?.taskId ?? this.ledger.latestTaskId();
    const view = taskId ? this.ledger.taskView(taskId) : null;
    const detailed = args.scope === "detailed";
    return {
      ok: true,
      digest,
      ...(view ? { status: this.taskOutcome.get(view.taskId) ?? view.status } : { status: "none" }),
      ...(detailed && view ? { task: view, guidance: this.ledger.listGuidance(view.taskId).slice(-5) } : {}),
      note: "Sourced from the ledger just now. Say the current step and the last completed ones; invent nothing.",
    };
  }

  /** Idempotent by construction: cancelling twice is one cancel, and no task is not a failure. */
  private toolCancelTask(args: Record<string, unknown>): Record<string, unknown> {
    const wanted = typeof args.task_id === "string" ? args.task_id.trim() : "";
    const task = this.host.currentTask();
    if (!task) {
      if (wanted && !this.ledger.taskView(wanted)) return { ok: false, reason: "no_such_task", note: "Nothing was cancelled." };
      return { ok: true, cancelled: false, note: "No task is running." };
    }
    if (wanted && wanted !== task.taskId) {
      return { ok: false, reason: "not_running", note: `Nothing with that id is running; the active task is ${task.taskId}.` };
    }
    if (this.cancelled.has(task.taskId)) return { ok: true, cancelled: true, task_id: task.taskId, note: "Already stopping." };
    this.cancelled.add(task.taskId);
    this.taskOutcome.set(task.taskId, "cancelled");
    this.taskAc?.abort();
    this.host.endTask();
    this.taskAc = null;
    this.broadcastTask();
    return { ok: true, cancelled: true, task_id: task.taskId };
  }

  /** Valid ONLY while the coding agent is waiting on a permission or elicitation ask. */
  private toolAnswerAgent(args: Record<string, unknown>): Record<string, unknown> {
    const requestId = String(args.request_id ?? "").trim();
    const pending = this.pendingAsks.get(requestId);
    if (!pending) {
      return {
        ok: false,
        reason: "no_pending_request",
        note: "Only valid while a permission or elicitation request is pending; nothing was answered.",
      };
    }
    const decisionRaw = String(args.decision ?? "");
    const decision = decisionRaw === "approve" || decisionRaw === "deny" || decisionRaw === "reply" ? decisionRaw : "deny";
    this.pendingAsks.delete(requestId);
    if (pending.kind === "permission") {
      const wantApproval = decision === "approve";
      const pick =
        pending.options.find((o) => (wantApproval ? o.kind === "allow_once" : o.kind === "reject_once")) ??
        pending.options[wantApproval ? 0 : pending.options.length - 1];
      const optionId = pick?.id ?? (wantApproval ? "allow" : "deny");
      this.host.answerPermission(requestId, optionId);
      return { ok: true, answered: optionId };
    }
    const reply = decision === "reply" ? String(args.reply_text ?? "") : undefined;
    this.host.answerElicitation(
      requestId,
      decision === "deny" ? "decline" : "accept",
      reply !== undefined ? { text: reply } : undefined,
    );
    return { ok: true, answered: decision };
  }

  // ── dispatch ───────────────────────────────────────────────────────────

  private keyOf(draftId: string, englishSpec: string, parentTaskId: string | null): string {
    return idempotencyKeyOf({ chatId: this.chatId, englishSpec, parentTaskId, draftId });
  }

  /** Why a dispatch may not happen right now, or null. Checked BEFORE a ticket is consumed. */
  private dispatchBlocked(): string | null {
    if (this.closed) return "socket_closed";
    if (this.waitingOnOrphan) return "previous_task_running";
    if (!this.host.id) return "no_agent_bound";
    return null;
  }

  /** Key 3 (the DraftCard) and the key-1-after-key-2 path both funnel through here. */
  private tryDispatch(draftId: string, version: number, source: "voice" | "ui"): Record<string, unknown> {
    const blocked = this.dispatchBlocked();
    if (blocked) {
      return { dispatched: false, reason: blocked, note: blocked === "no_agent_bound" ? "No coding agent is bound, so nothing was sent." : "Nothing was sent." };
    }
    const ticket = this.gate.dispatchDraft(draftId, version, source);
    if (!ticket) return { dispatched: false, reason: "already_dispatched" };
    const out = this.dispatchTicket(ticket);
    this.broadcastDraft();
    this.broadcastTask();
    return { dispatched: !out.duplicate, task_id: out.taskId };
  }

  /**
   * The commit. Order is the whole contract:
   *   1. compute the key (already done at propose time);
   *   2. `recordDispatch` — the CAS on disk, BEFORE the agent hears the spec (rule 4);
   *   3. `host.startTask`, then `void host.runTurn(...)` — started, never awaited.
   * A replay of the same key returns null at step 2 and the ACP call never happens.
   */
  private dispatchTicket(ticket: DispatchTicket): { taskId: string; key: string; duplicate: boolean } {
    const key = this.draftKeys.get(ticket.draftId) ?? this.keyOf(ticket.draftId, ticket.englishSpec, ticket.parentTaskId);
    if (ticket.intentType !== "new_task") return this.steerTicket(ticket, key);
    const taskId = `task_${randomUUID()}`;
    const rec = this.ledger.recordDispatch(taskId, key, ticket.englishSpec, this.host.id ?? "builtin");
    if (!rec) {
      // The key is already committed: a reconnect (or a duplicate click) replayed this dispatch.
      log.warn("realtime", "dispatch replay refused — idempotency key already committed");
      return { taskId: this.findTaskByKey(key) ?? "", key, duplicate: true };
    }
    const ac = new AbortController();
    this.taskAc = ac;
    this.taskOutcome.set(taskId, "running");
    this.host.startTask(taskId, ticket.englishSpec, ac.signal);
    this.broadcastTask();
    this.runTask(taskId, ticket.englishSpec, ac);
    return { taskId, key, duplicate: false };
  }

  /** The coding task itself. Everything here is fire-and-forget from the voice socket's view. */
  private runTask(taskId: string, spec: string, ac: AbortController): void {
    const blocks: MessageBlock[] = [];
    let said = "";
    const emit: Emit = (e) => {
      if (e.type === "text_delta") said += e.text;
      foldBlock(blocks, e);
      if (PROGRESS_EVENTS.has(e.type)) this.ledger.recordAcpEvent(taskId, e);
      this.send({ t: "sse", event: e as unknown as Record<string, unknown> });
      this.broadcastTask();
    };
    void this.host
      .runTurn({ text: spec, frames: [] }, emit, ac.signal)
      .then((ran) => this.finishTask(taskId, blocks, said, ran ? "task_finished" : "task_failed"))
      .catch((e) => {
        log.warn("realtime", "task:", e);
        this.finishTask(taskId, blocks, said, "task_failed");
      });
  }

  private finishTask(taskId: string, blocks: MessageBlock[], said: string, kind: "task_finished" | "task_failed"): void {
    const cancelled = this.cancelled.has(taskId);
    this.ledger.recordAcpEvent(taskId, { type: kind === "task_failed" ? "error" : "done", text: kind });
    if (this.host.currentTask()?.taskId === taskId) this.host.endTask();
    if (this.taskAc && !this.taskAc.signal.aborted) this.taskAc = null;
    this.taskOutcome.set(taskId, cancelled ? "cancelled" : kind === "task_failed" ? "failed" : "done");
    void this.persistTaskBlocks(blocks);
    this.broadcastTask();
    if (cancelled) return; // the person asked it to stop; nothing to announce
    const summary = voiceSummary(said);
    this.pushTaskEvent(kind, `${taskId} ${kind === "task_failed" ? "failed" : "finished"}${summary ? `: ${summary}` : ""}`);
    // A task the previous socket started has now finished: let that socket's agent bind and clean up.
    if (RealtimeSession.continuing.get(this.chatId) === this) {
      RealtimeSession.continuing.delete(this.chatId);
      this.host.dispose();
      const current = this.chatId ? RealtimeSession.sockets.get(this.chatId) : undefined;
      if (current && current !== this) current.resumeAfterOrphan();
    }
  }

  /** A correction to a running task: `GuidanceItem` order is preserved by the ledger, and the
   *  transport is chosen per agent inside `AcpHost`. Never awaited by a Live handler. */
  private steerTicket(ticket: DispatchTicket, key: string): { taskId: string; key: string; duplicate: boolean } {
    const taskId = ticket.parentTaskId ?? this.host.currentTask()?.taskId ?? this.ledger.latestTaskId() ?? "";
    if (!taskId) return { taskId: "", key, duplicate: false };
    // A steer's durable record IS the guidance row; the same key twice is the same correction.
    if (this.ledger.listGuidance(taskId).some((g) => g.idempotencyKey === key)) {
      log.warn("realtime", "steer replay refused — the correction is already in the ledger");
      return { taskId, key, duplicate: true };
    }
    const item: GuidanceItem = {
      guidanceId: `guid_${randomUUID()}`,
      seq: 0,
      taskId,
      idempotencyKey: key,
      kind: ticket.intentType === "new_task" ? "steer_add" : ticket.intentType,
      text: ticket.englishSpec,
      sourceSummaryBn: ticket.summaryBn,
      status: "pending",
      createdAt: this.now(),
    };
    void this.host
      .steerTask(taskId, item)
      .then((outcome) => {
        this.broadcastTask();
        this.pushTaskEvent("steer_applied", `${ticket.summaryBn || ticket.englishSpec} (${outcome.mechanism})`);
      })
      .catch((e) => log.warn("realtime", "steer:", e));
    return { taskId, key, duplicate: false };
  }

  private findTaskByKey(key: string): string | null {
    const latest = this.ledger.latestTaskId();
    if (latest && this.ledger.hasDispatchedKey(key)) return latest;
    return null;
  }

  /** Key 3 from the DraftCard. Still requires key 2's version match — never a bypass. */
  private onDraftAction(msg: Extract<RealtimeClientMsg, { t: "draft_action" }>): void {
    const action = msg.action;
    if (action === "send") {
      const blocked = this.dispatchBlocked();
      if (blocked) {
        this.send({ t: "error", code: "dispatch_blocked", message: blocked === "no_agent_bound" ? "No coding agent is bound." : "Not now." });
        this.broadcastDraft();
        return;
      }
    }
    const out = this.gate.onDraftAction(action, msg.draftId, msg.version);
    if (out.dispatched) this.dispatchTicket(out.dispatched);
    this.broadcastDraft();
    this.broadcastTask();
  }

  // ── the four pushed events ─────────────────────────────────────────────

  /**
   * Queue one of the four events. Progress is pulled (`get_status`); these four are the only
   * pushes, and they are delivered only when Gemini is idle — a native-audio model speaking over
   * its own silence is exactly what the user is trying to escape.
   */
  private pushTaskEvent(kind: PushedEvent, text: string): void {
    // A socket that already died has no model to speak to; the ledger and task_status carry it.
    if (this.closed) return;
    const clean = speakable(text).slice(0, 240);
    this.pendingEvents.push({ kind, text: clean });
    this.flushEvents();
  }

  private flushEvents(): void {
    if (this.closed || this.delivering || !this.pendingEvents.length || !this.gemini) return;
    const phase = this.gemini.currentPhase();
    if (phase !== "live" && phase !== "sleeping") return;
    if (this.gemini.isTurnBusy()) return;
    const batch = this.pendingEvents.splice(0, this.pendingEvents.length);
    this.delivering = true;
    void (async () => {
      try {
        // A sleeping socket is woken FIRST, and only now: a dropped push is better than a wake
        // during a generation.
        if (this.gemini?.currentPhase() === "sleeping") await this.gemini.wake();
        if (!this.gemini || this.closed || this.gemini.currentPhase() !== "live") {
          this.pendingEvents.unshift(...batch);
          return;
        }
        for (const e of batch) this.gemini.sendRealtimeText(this.delivery(e));
      } catch (e) {
        log.warn("realtime", "deliver event:", e);
        this.pendingEvents.unshift(...batch);
      } finally {
        this.delivering = false;
        if (this.pendingEvents.length) queueMicrotask(() => this.flushEvents());
      }
    })();
  }

  private delivery(e: QueuedEvent): string {
    const label =
      e.kind === "task_finished" ? "the task finished"
      : e.kind === "task_failed" ? "the task failed"
      : e.kind === "needs_input" ? "the coding agent needs an answer"
      : "the correction was applied";
    return `${EVENT_PREFIX} ${label}: ${e.text}. Tell the person in one short sentence, then stop.`;
  }

  // ── persistence (clean transcript and tool cards only) ────────────────

  private async persist(role: "user" | "assistant", text: string): Promise<void> {
    if (!this.chatId) return;
    const clean = cleanForChat(text).slice(0, MAX_TRANSCRIPT_CHARS);
    if (!clean) return;
    try {
      await addMessage(this.chatId, role, [{ type: "text", text: clean }], true);
    } catch (e) {
      log.error("realtime", "persist transcript:", e);
    }
  }

  /** The coding agent's tool cards, once the task settles: the one place a realtime chat shows
   *  what was actually done. Cleaned of internal-state text, saved even if the socket is gone. */
  private async persistTaskBlocks(blocks: readonly MessageBlock[]): Promise<void> {
    if (!this.chatId) return;
    const clean = cleanBlocks(blocks);
    if (!clean.length) return;
    try {
      await addMessage(this.chatId, "assistant", clean, true);
    } catch (e) {
      log.error("realtime", "persist task blocks:", e);
    }
  }

  private async ingestReplay(messages: ReplayMessage[]): Promise<void> {
    if (!this.chatId || !messages.length) return;
    try {
      // An OpenLive-origin chat already renders its own transcript; only a resumed external
      // session's turns are new to it.
      if (listMessages(this.chatId).length) return;
      for (const m of messages) {
        const clean = cleanBlocks(m.content);
        if (clean.length) await addMessage(this.chatId, m.role, clean, true);
      }
    } catch (e) {
      log.error("realtime", "replay ingest:", e);
    }
  }

  private rehydrate(): Message[] {
    if (!this.chatId) return [];
    let rows: ReturnType<typeof listMessages>;
    try {
      rows = listMessages(this.chatId);
    } catch {
      return [];
    }
    const out: Message[] = [];
    for (const m of rows.slice(-SEED_TURNS)) {
      if (m.role !== "user" && m.role !== "assistant") continue;
      const text = cleanForChat(m.content.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join(" "));
      if (text) out.push({ role: m.role, text });
    }
    return out;
  }

  // ── ACP ports ─────────────────────────────────────────────────────────

  private ports(): AcpHostPorts {
    return {
      chatId: this.chatId,
      closed: () => this.closed,
      send: (m) => this.forwardLive(m),
      session: () => this.toolSession,
      // The ONE MCP server serves these two tools; the loopback host and the ACP binding are the
      // same `AcpHost` `/live` uses, so `/realtime` adds no second server.
      tools: () => this.tools,
      approve: this.approve,
      ledger: this.ledger,
      // A coding task outlives the utterances around it; the asks belong to the task.
      turnActive: () => !!this.host.currentTask(),
      turnSignal: () => this.taskAc?.signal ?? null,
      replyTurn: () => undefined,
      seed: () => this.rehydrate(),
      pendingCut: () => null,
      onMeta: (meta) => this.send({ t: "agent_meta", ...meta }),
      onReplay: (msgs) => this.ingestReplay(msgs),
      onReplayExpected: () => {},
      onFolderMoved: () => {},
      onStarted: () => {},
      onReady: () => {},
      onStartError: (e) => {
        this.send({ t: "error", code: "agent_start_failed", message: String((e as Error)?.message ?? e) });
      },
    };
  }

  /**
   * The two tools the coding agent reaches over the hosted MCP server. `user_guidance` is the
   * `mcp_guidance` steering row `hermes acp` honoured in Stage 0; `report_progress` lets the agent
   * write its own step into the ledger, so `get_status` stays grounded in fact rather than in the
   * model's memory. Both run under `AcpHost.activeTaskSignal()`, i.e. for the whole task.
   */
  private buildTools(): ToolSet {
    const userGuidance: Tool<unknown, unknown> = {
      name: "user_guidance",
      description:
        "Drain the person's queued mid-task corrections, oldest first. Call this at every step boundary and apply them in the order returned. Returns NO_NEW_GUIDANCE when there is none.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
        additionalProperties: false,
      },
      readOnly: true,
      execute: async () => {
        const taskId = this.host.currentTask()?.taskId ?? this.ledger.latestTaskId() ?? "";
        const { text } = taskId ? this.host.drainGuidance(taskId) : { text: "NO_NEW_GUIDANCE" };
        return { content: [{ type: "text", text }], details: { text } };
      },
    };
    const reportProgress: Tool<{ status?: string; detail?: string }, unknown> = {
      name: "report_progress",
      description:
        "Record the step you are on, and what you just completed, so the person can ask for a truthful status. Call it when a step starts or finishes; it never stops your work.",
      parameters: {
        type: "object",
        properties: {
          status: { type: "string", description: "Short, plain-language description of the step that is now current." },
          detail: { type: "string", description: "Optional: what the previous step actually achieved." },
        },
        required: ["status"],
        additionalProperties: false,
      },
      readOnly: true,
      execute: async (args) => {
        const taskId = this.host.currentTask()?.taskId ?? this.ledger.latestTaskId() ?? "";
        const status = String(args.status ?? "").trim();
        if (taskId && status) this.ledger.recordAcpEvent(taskId, { type: "status", text: status });
        const detail = String(args.detail ?? "").trim();
        if (taskId && detail) this.ledger.recordAcpEvent(taskId, { type: "tool_done", tool: detail });
        this.broadcastTask();
        return { content: [{ type: "text", text: "Recorded." }], details: { ok: true, taskId } };
      },
    };
    return new ToolSet([userGuidance, reportProgress]);
  }

  // ── the task that outlives the socket ────────────────────────────────

  /** The previous socket's mid-task session has finished: bind this socket's agent now. */
  resumeAfterOrphan(): void {
    this.waitingOnOrphan = false;
    if (this.closed) return;
    void this.host.restore().catch((e) => log.warn("realtime", "restore after task:", e));
  }

  private broadcastDraft(): void {
    this.send({ t: "draft_state", draft: this.gate.view() });
  }

  private broadcastTask(): void {
    const taskId = this.host.currentTask()?.taskId ?? this.ledger.latestTaskId();
    const view = taskId ? this.ledger.taskView(taskId) : null;
    const task = view ? { ...view, status: this.taskOutcome.get(view.taskId) ?? view.status } : null;
    this.send({ t: "task_status", task });
  }

  // ── teardown ─────────────────────────────────────────────────────────

  private dispose(): void {
    if (this.closed) return;
    this.closed = true;
    // A socket that died mid-confirmation re-asks instead of resuming a half-done dispatch; a
    // draft that already dispatched is latched in the gate, so a reconnect can never re-send it.
    this.gate.onDisconnect();
    this.pendingAsks.clear();
    this.pendingEvents.length = 0;
    this.nominated = null;
    const client = this.gemini;
    this.gemini = null;
    if (client) void client.close().catch(() => {});
    if (this.chatId && RealtimeSession.sockets.get(this.chatId) === this) RealtimeSession.sockets.delete(this.chatId);
    if (this.host.currentTask()) {
      // Do NOT abort a coding task because the voice socket dropped: that is exactly the Stage 9
      // failure ("the agent keeps running"). The host stays alive, its progress keeps landing in
      // the ledger, and it is disposed the moment the task ends. A reconnecting socket sees this
      // chat in `continuing` and watches instead of spawning a second agent on the same project.
      if (this.chatId) RealtimeSession.continuing.set(this.chatId, this);
      log.warn("realtime", `socket closed mid-task ${this.host.currentTask()!.taskId}: continuing`);
    } else {
      this.host.dispose();
    }
    try {
      this.ws.close();
    } catch {
      /* already closing */
    }
  }

  /** Diagnostics and tests only — never on the wire. */
  snapshot(): {
    phase: RealtimePhase;
    epoch: number;
    draft: ActiveDraftView | null;
    pendingEvents: number;
    pendingAsks: number;
    tasks: Record<string, string>;
    closed: boolean;
    waitingOnOrphan: boolean;
  } {
    return {
      phase: this.phase,
      epoch: this.gen,
      draft: this.gate.view(),
      pendingEvents: this.pendingEvents.length,
      pendingAsks: this.pendingAsks.size,
      tasks: Object.fromEntries(this.taskOutcome),
      closed: this.closed,
      waitingOnOrphan: this.waitingOnOrphan,
    };
  }
}

/** The subset of ACP events the ledger folds into progress; `text_delta` noise stays out of it. */
const PROGRESS_EVENTS = new Set(["acp_tool_call", "acp_tool_update", "tool_start", "tool_done", "todos", "status", "done", "error"]);

/** The Gemini client's phases, as the wire vocabulary. */
function mapPhase(phase: BidiPhase): RealtimePhase {
  switch (phase) {
    case "live":
      return "live";
    case "sleeping":
      return "sleeping";
    case "connecting":
      return "connecting";
    case "rotating":
    case "reconnecting":
      return "reconnecting";
    case "error":
      return "error";
    default:
      return "idle";
  }
}
