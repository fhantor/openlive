// File: services/agent/src/live/gemini-bidi.ts
// WHY: Pure Gemini Live BidiGenerateContent WebSocket client (Node 24 + ws), decoupled from the
// OpenLive session/DB so fault-injection tests run standalone. One class owns the socket, the
// faults a Live call actually suffers (1006 transport drop, stale 1007 handle, 1008 inactivity,
// goAway) and the recovery for each.

import { EventEmitter } from "node:events";
import WebSocket, { type RawData } from "ws";

export const GEMINI_LIVE_WS_BASE =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

/** The only native-audio model verified on this key (`responseModalities:["AUDIO"]`). */
export const DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.8-live";
/** The voice the documented setup message uses; a caller may override it per call. */
export const DEFAULT_GEMINI_LIVE_VOICE = "Aoede";

export type GeminiTurnRole = "user" | "model";

export interface GeminiContentTurn {
  role: GeminiTurnRole;
  parts: Array<{ text: string }>;
}

export interface GeminiFunctionCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface GeminiFunctionResponse {
  id: string;
  name: string;
  response: Record<string, unknown>;
  willContinue?: boolean;
  scheduling?: "SILENT" | "WHEN_IDLE" | "INTERRUPT";
}

export type BidiPhase =
  | "idle"
  | "connecting"
  | "live"
  | "sleeping"
  | "rotating"
  | "reconnecting"
  | "closed"
  | "error";

export interface GeminiBidiConfig {
  apiKey: string;
  model?: string;
  voiceName?: string;
  silenceDurationMs?: number;
  systemInstruction: string;
  /** Last stored resumable handle; "" (or absent) starts cold. */
  resumeHandle?: string;
  /** Prompt-token count right before the last disconnect — the cold-reset canary baseline. */
  preDisconnectPromptTokens?: number;
  /** The bounded cold-start header, seeded once per cold (re)connect. */
  getColdHistory?: () => GeminiContentTurn[] | Promise<GeminiContentTurn[]>;
  reconnectAttempts?: number;
  /** Override the BidiGenerateContent endpoint. The fault tests point this at a local mock. */
  wsUrl?: string;
  /** How long to wait for `setupComplete` before giving up on a socket (default 15 s). */
  setupTimeoutMs?: number;
}

/** What the repo's one key path resolves to, before it reaches the socket. */
export interface ResolvedGeminiLive {
  apiKey: string;
  model: string;
  voiceName?: string;
}

/** A selected model is only reused here when it is a *Live* model; the live picker may hold a chat id. */
const isLiveModel = (model: string): boolean => /gemini-[\w.-]*-live$/i.test(model.trim());

/**
 * Resolve the key the same way every other live path in the agent does: `resolveLive()` in
 * providers.ts (the resolver turn-runner.ts uses), then the provider's declared env keys as the
 * documented fallback. Never invent a second key store, and never log the value.
 * Imported lazily so the socket class itself stays free of settings/DB (the fault tests load it alone).
 */
export async function resolveGeminiLive(opts: { model?: string; voiceName?: string } = {}): Promise<ResolvedGeminiLive> {
  let selectedModel = "";
  let selectedKey = "";
  try {
    const { resolveLive } = await import("../providers.js");
    const { provider, model, apiKey } = resolveLive();
    selectedModel = model;
    // Only the Google provider's key is a Gemini key; another provider's would be rejected by the API.
    if (provider.id === "google" && apiKey) selectedKey = apiKey;
  } catch {
    /* settings unreadable (probe/test environment) — the env fallback below still works */
  }
  selectedKey ||= process.env.GEMINI_API_KEY?.trim() ?? "";
  selectedKey ||= process.env.GOOGLE_API_KEY?.trim() ?? "";
  if (!selectedKey) {
    throw new Error(
      "No Gemini Live API key. Pick Google (Gemini) as the Live provider in Settings, or set GEMINI_API_KEY / GOOGLE_API_KEY.",
    );
  }
  const model = opts.model?.trim() || (isLiveModel(selectedModel) ? selectedModel.trim() : DEFAULT_GEMINI_LIVE_MODEL);
  return { apiKey: selectedKey, model, ...(opts.voiceName ? { voiceName: opts.voiceName } : {}) };
}

/** Build the client from the repo's settings, so the caller never handles the raw key. */
export async function createGeminiLiveClient(
  cfg: Omit<GeminiBidiConfig, "apiKey" | "model"> & { model?: string; voiceName?: string },
): Promise<GeminiBidiClient> {
  const { voiceName, ...rest } = cfg;
  const resolved = await resolveGeminiLive({ model: cfg.model, voiceName });
  return new GeminiBidiClient({
    ...rest,
    apiKey: resolved.apiKey,
    model: resolved.model,
    voiceName: voiceName ?? resolved.voiceName,
  });
}

// WHY: Uppercase Schema types and NON_BLOCKING behavior match v1beta BidiGenerateContent FunctionDeclaration proto JSON rules.
// The model has no dispatch tool on purpose: a model that can lie about "sent it" cannot exist without one.
export const ORCHESTRATOR_FUNCTION_DECLARATIONS = [
  {
    name: "propose_instruction",
    description:
      "Stage or update the active coding instruction draft after listening to the user. Never executes code. Always use this before asking confirmation.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Structured draft extracted from user speech.",
      properties: {
        summary_bn: {
          type: "STRING" as const,
          description: "Concise, natural Bengali summary of the instruction to read back to the user.",
        },
        english_spec: {
          type: "STRING" as const,
          description: "Exact, unambiguous English imperative specification for the coding agent.",
        },
        target: {
          type: "STRING" as const,
          enum: ["new_task", "steer_active_task"],
          description: "Whether this starts a new coding task or modifies the currently running coding task.",
        },
        steer_kind: {
          type: "STRING" as const,
          enum: ["steer_add", "steer_pivot"],
          description: "Required when target is steer_active_task: steer_add appends constraints, steer_pivot replaces direction.",
        },
        open_questions: {
          type: "ARRAY" as const,
          description: "Unresolved ambiguities that must be clarified with the user before requesting confirmation.",
          items: { type: "STRING" as const },
        },
      },
      required: ["summary_bn", "english_spec", "target"],
    },
  },
  {
    name: "request_confirmation",
    description:
      "Transition the staged draft into confirming state right as you read the summary aloud and ask the user whether to send it.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Draft version being read back for user confirmation.",
      properties: {
        draft_id: {
          type: "STRING" as const,
          description: "Active draft identifier returned by propose_instruction.",
        },
        version: {
          type: "INTEGER" as const,
          description: "Exact draft version number being read back.",
        },
        readback_bn: {
          type: "STRING" as const,
          description: "The Bengali read-back spoken to the user, ending with an open question such as 'এটা কি পাঠাবো?' (not an affirmative).",
        },
      },
      required: ["draft_id", "version", "readback_bn"],
    },
  },
  {
    name: "confirm_instruction",
    description:
      "Request dispatch of the confirmed draft after the user explicitly says yes/pathao. The orchestrator independently verifies the user's transcript.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Confirmed draft coordinates.",
      properties: {
        draft_id: {
          type: "STRING" as const,
          description: "Draft identifier to dispatch.",
        },
        version: {
          type: "INTEGER" as const,
          description: "Exact draft version number the user confirmed.",
        },
      },
      required: ["draft_id", "version"],
    },
  },
  {
    name: "get_status",
    description:
      "Fetch live, ground-truth progress from the orchestrator ledger and active coding agent (current tool, completed steps, todo progress). Always call before answering status questions.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Progress detail level.",
      properties: {
        scope: {
          type: "STRING" as const,
          enum: ["summary", "detailed"],
          description: "summary returns current stage and recent tools; detailed adds full todo list and recent guidance items.",
        },
      },
      required: [],
    },
  },
  {
    name: "cancel_task",
    description:
      "Abort the currently running coding-agent task when the user explicitly asks to stop or abort work.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Cancellation details.",
      properties: {
        task_id: {
          type: "STRING" as const,
          description: "Running task ID if known; omit to target the active task.",
        },
        reason: {
          type: "STRING" as const,
          description: "Short reason stated by the user.",
        },
      },
      required: ["reason"],
    },
  },
  {
    name: "answer_agent",
    description:
      "Forward the user's explicit answer to a pending coding-agent permission request or elicitation question.",
    behavior: "NON_BLOCKING" as const,
    parameters: {
      type: "OBJECT" as const,
      description: "Decision for the pending agent prompt.",
      properties: {
        request_id: {
          type: "STRING" as const,
          description: "Pending permission or elicitation request ID.",
        },
        decision: {
          type: "STRING" as const,
          enum: ["approve", "deny", "reply"],
          description: "approve or deny for permission prompts; reply for free-form elicitation questions.",
        },
        reply_text: {
          type: "STRING" as const,
          description: "Answer text when decision is reply.",
        },
      },
      required: ["request_id", "decision"],
    },
  },
] as const;

// WHY: 16 kHz PCM16LE = 32 bytes/ms; 80 ms (2560 B) sits cleanly in the 60-100 ms window and avoids tiny-chunk disconnects.
export const MIC_COALESCE_BYTES = 2560;
export const MIC_FLUSH_MS = 80;
/** 1.5 s @ 16 kHz PCM16LE — the ring the renderer keeps while sleeping is about the same size. */
export const MAX_PREROLL_BYTES = 48_000;

/** "30s" / "12.5s" (proto3 Duration JSON) to milliseconds; anything else is 0. */
export function parseProtoDurationMs(raw: unknown): number {
  if (typeof raw !== "string") return 0;
  const m = /^([\d.]+)s$/i.exec(raw.trim());
  if (!m) return 0;
  const seconds = Number(m[1]);
  return Number.isFinite(seconds) ? Math.max(0, Math.round(seconds * 1000)) : 0;
}

export class GeminiBidiClient extends EventEmitter {
  private ws: WebSocket | null = null;
  /** Bumped on every socket swap, so a frame from the previous socket can never act on this one. */
  private socketSeq = 0;
  private phase: BidiPhase = "idle";
  private resumeHandle: string;
  /** The seq of the last `resumable:true` update: the checkpoint a resume replays past. */
  private lastResumableSeq = 0;
  /** Server frames seen on this client, used as the checkpoint when the server sends no index. */
  private serverFrameSeq = 0;
  private coldConnect = false;
  private awaitingColdHistory = false;
  private wasResumedConnect = false;
  private canaryChecked = false;
  /** Set when resumption context was proven lost: the next connect must be cold and reseed. */
  private needsColdReseed = false;
  private preDisconnectPromptTokens: number;
  private lastPromptTokenCount = 0;

  private turnOpen = false;
  private generating = false;
  private inFlightToolIds = new Map<string, number>(); // callId -> socketSeq
  private muted = false;

  private micChunks: Buffer[] = [];
  private micBytes = 0;
  private micFlushTimer: ReturnType<typeof setTimeout> | null = null;
  private preRollChunks: Buffer[] = [];
  private preRollBytes = 0;

  private goAwayTimer: ReturnType<typeof setTimeout> | null = null;
  private rotationPending = false;
  private silentSyncQueue: GeminiContentTurn[][] = [];
  private intentionalClose = false;
  private reconnectAttempts = 0;

  constructor(private readonly cfg: GeminiBidiConfig) {
    super();
    this.resumeHandle = cfg.resumeHandle ?? "";
    this.preDisconnectPromptTokens = cfg.preDisconnectPromptTokens ?? 0;
  }

  currentPhase(): BidiPhase {
    return this.phase;
  }

  currentResumeHandle(): string {
    return this.resumeHandle;
  }

  /** The seq of the last stored resumable checkpoint; the ledger persists this next to the handle. */
  currentCheckpointSeq(): number {
    return this.lastResumableSeq;
  }

  isTurnBusy(): boolean {
    return this.turnOpen || this.generating;
  }

  private setPhase(next: BidiPhase, reason?: string): void {
    if (this.phase === next && !reason) return;
    this.phase = next;
    this.emit("phase", { phase: next, reason });
  }

  /** Emit a fatal error without Node's "unhandled 'error' event throws" footgun. */
  private fail(reason: string, err: Error): void {
    this.setPhase("error", reason);
    if (this.listenerCount("error") > 0) this.emit("error", err);
  }

  async connect(opts?: { forceCold?: boolean; preRoll?: Buffer[] }): Promise<void> {
    try {
      await this.connectOnce(opts);
    } catch (err) {
      // Any failed resume — a 1007 the server sent, or a socket that never opened — leaves the handle in
      // doubt. Losing it must be survivable, not fatal: clear it and take exactly one cold attempt.
      if (this.resumeHandle === "" || this.phase === "closed") throw err;
      this.resumeHandle = "";
      const error = err instanceof Error ? err : new Error(String(err));
      if (this.listenerCount("resumeFailed") > 0) this.emit("resumeFailed", error);
      await this.connectOnce({ forceCold: true });
    }
  }

  private async connectOnce(opts?: { forceCold?: boolean; preRoll?: Buffer[] }): Promise<void> {
    if (opts?.preRoll?.length) {
      for (const chunk of opts.preRoll) this.pushPreRoll(chunk);
    }
    this.intentionalClose = false;
    // A canary-proven context loss forces the cold path even when a handle is still stored.
    const wantsCold = Boolean(opts?.forceCold) || this.needsColdReseed;
    const useHandle = !wantsCold && this.resumeHandle.length > 0;
    this.coldConnect = !useHandle;
    this.needsColdReseed = false;
    this.wasResumedConnect = useHandle;
    this.canaryChecked = false;
    this.awaitingColdHistory = this.coldConnect;

    if (this.phase !== "live") this.setPhase("connecting");

    const modelId = this.cfg.model ?? DEFAULT_GEMINI_LIVE_MODEL;
    const base = this.cfg.wsUrl ?? GEMINI_LIVE_WS_BASE;
    const url = `${base}?key=${encodeURIComponent(this.cfg.apiKey)}`;
    const seq = ++this.socketSeq;
    const sock = new WebSocket(url);
    const setupTimeoutMs = this.cfg.setupTimeoutMs ?? 15_000;

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const teardown = () => {
        if (this.ws === sock) this.ws = null;
        this.awaitingColdHistory = false;
      };
      const setupTimeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        teardown();
        try {
          sock.terminate();
        } catch {
          /* ignore */
        }
        reject(new Error(`Gemini setupComplete timeout (${setupTimeoutMs}ms)`));
      }, setupTimeoutMs);

      sock.once("open", () => {
        this.ws = sock;
        this.sendSetupFrame(sock, modelId, useHandle ? this.resumeHandle : "");
      });

      sock.on("message", (data: RawData) => {
        if (seq !== this.socketSeq) return;
        let frame: Record<string, unknown>;
        try {
          frame = JSON.parse(rawDataToUtf8(data)) as Record<string, unknown>;
        } catch {
          return;
        }

        if (!settled && "setupComplete" in frame) {
          settled = true;
          clearTimeout(setupTimeout);
          this.reconnectAttempts = 0;
          this.turnOpen = false;
          this.generating = false;
          this.rotationPending = false;
          this.clearGoAwayTimer();
          this.emit("setupComplete", frame.setupComplete);

          // Cold seeding must finish before any realtimeInput: the server treats these turns as
          // history (initialHistoryInClientContent) and an audio frame racing them reorders state.
          void (async () => {
            try {
              if (this.coldConnect) {
                let turns: GeminiContentTurn[] = [];
                try {
                  turns = this.cfg.getColdHistory ? await this.cfg.getColdHistory() : [];
                } catch {
                  turns = []; // the default [SESSION_INIT] seed below is the survivable fallback
                }
                await this.sendColdHistory(turns);
              }
              this.awaitingColdHistory = false;
              this.setPhase("live");
              this.drainPreRoll();
              this.drainSilentSyncQueue();
              resolve();
            } catch (err) {
              teardown();
              try {
                sock.terminate();
              } catch {
                /* ignore */
              }
              reject(err instanceof Error ? err : new Error(String(err)));
            }
          })();
          return;
        }

        this.handleServerFrame(frame, seq);
      });

      sock.once("error", (err) => {
        if (!settled) {
          settled = true;
          clearTimeout(setupTimeout);
          teardown();
          reject(err);
        }
      });

      sock.once("close", (code: number, reasonBuf: Buffer) => {
        clearTimeout(setupTimeout);
        const reason = reasonBuf?.toString("utf8") ?? "";
        if (!settled) {
          settled = true;
          teardown();
          reject(new Error(`Gemini closed during setup (${code}): ${reason}`));
          return;
        }
        if (seq !== this.socketSeq) return;
        this.ws = null;
        this.clearGoAwayTimer();
        this.onSocketClosed(code, reason);
      });
    });
  }

  private buildSetupPayload(modelId: string, handle: string): Record<string, unknown> {
    // WHY: gemini-3.8-live rejects thinkingConfig; omit it unless a model explicitly ends with -extended-thinking.
    const setup: Record<string, unknown> = {
      model: modelId.startsWith("models/") ? modelId : `models/${modelId}`,
      generationConfig: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: this.cfg.voiceName ?? DEFAULT_GEMINI_LIVE_VOICE,
            },
          },
        },
      },
      systemInstruction: {
        parts: [{ text: this.cfg.systemInstruction }],
      },
      tools: [
        {
          functionDeclarations: ORCHESTRATOR_FUNCTION_DECLARATIONS,
        },
      ],
      realtimeInputConfig: {
        automaticActivityDetection: {
          disabled: false,
          startOfSpeechSensitivity: "START_SENSITIVITY_HIGH",
          endOfSpeechSensitivity: "END_SENSITIVITY_LOW",
          prefixPaddingMs: 200,
          silenceDurationMs: this.cfg.silenceDurationMs ?? 900,
        },
        activityHandling: "START_OF_ACTIVITY_INTERRUPTS",
        turnCoverage: "TURN_INCLUDES_ONLY_ACTIVITY",
      },
      inputAudioTranscription: {},
      outputAudioTranscription: {},
      contextWindowCompression: {
        slidingWindow: {},
      },
      sessionResumption: handle ? { handle } : {},
      ...(this.coldConnect
        ? {
            historyConfig: {
              initialHistoryInClientContent: true,
            },
          }
        : {}),
    };
    return { setup };
  }

  private sendSetupFrame(sock: WebSocket, modelId: string, handle: string): void {
    sock.send(JSON.stringify(this.buildSetupPayload(modelId, handle)));
  }

  // WHY: Under initialHistoryInClientContent:true, turnComplete:true ending on role:"model" seeds cold state without speaking.
  async sendColdHistory(turns: GeminiContentTurn[]): Promise<void> {
    const sock = this.ws;
    if (!sock || sock.readyState !== WebSocket.OPEN) {
      throw new Error("Cannot sendColdHistory: socket not open");
    }
    const safeTurns: GeminiContentTurn[] = turns.length
      ? turns
      : [
          {
            role: "user",
            parts: [{ text: "[SESSION_INIT] Ready for Bengali/English voice orchestration." }],
          },
          {
            role: "model",
            parts: [{ text: "Understood. Ready silently." }],
          },
        ];
    const payload = JSON.stringify({
      clientContent: {
        turns: safeTurns,
        turnComplete: true,
      },
    });
    // Await the flush: the caller may release mic audio the moment this resolves.
    await new Promise<void>((resolve, reject) => {
      sock.send(payload, (err) => (err ? reject(err) : resolve()));
    });
    this.awaitingColdHistory = false;
  }

  // WHY: clientContent interrupts any active generation regardless of turnComplete; queue until turnComplete & idle.
  sendSilentSync(turns: GeminiContentTurn[]): void {
    if (!turns.length) return;
    if (
      !this.ws ||
      this.ws.readyState !== WebSocket.OPEN ||
      this.awaitingColdHistory ||
      this.turnOpen ||
      this.generating
    ) {
      this.silentSyncQueue.push(turns);
      return;
    }
    this.ws.send(
      JSON.stringify({
        clientContent: {
          turns,
          turnComplete: false,
        },
      }),
    );
  }

  private drainSilentSyncQueue(): void {
    if (
      !this.silentSyncQueue.length ||
      !this.ws ||
      this.ws.readyState !== WebSocket.OPEN ||
      this.awaitingColdHistory ||
      this.turnOpen ||
      this.generating
    ) {
      return;
    }
    const merged = this.silentSyncQueue.splice(0, this.silentSyncQueue.length).flat();
    this.ws.send(
      JSON.stringify({
        clientContent: {
          turns: merged,
          turnComplete: false,
        },
      }),
    );
  }

  pushMicPcm16(chunk: Buffer): void {
    if (!chunk.length || this.muted) return;
    const sock = this.ws;
    // Anything that is not a live socket is pre-roll, not lost audio: the renderer keeps its own
    // 1.5 s ring, and a reconnect drains whatever is buffered here before the next frame.
    if (this.phase !== "live" || this.awaitingColdHistory || !sock || sock.readyState !== WebSocket.OPEN) {
      this.pushPreRoll(chunk);
      return;
    }

    this.micChunks.push(chunk);
    this.micBytes += chunk.length;
    if (this.micBytes >= MIC_COALESCE_BYTES) {
      this.flushMicBuffer();
      if (this.micBytes > 0) this.armMicFlushTimer();
      return;
    }
    this.armMicFlushTimer();
  }

  private armMicFlushTimer(): void {
    if (this.micFlushTimer) return;
    this.micFlushTimer = setTimeout(() => {
      this.micFlushTimer = null;
      // A timer tick is a latency bound, not a reason to emit a runt frame: below the 60 ms floor the
      // audio waits for the next chunk, and every boundary (sleep, rotation, mute, close) forces it out.
      this.flushMicBuffer();
    }, MIC_FLUSH_MS);
  }

  /** Take whatever is buffered, or null; also cancels the pending timer. */
  private takeMicPayload(): Buffer | null {
    if (this.micFlushTimer) {
      clearTimeout(this.micFlushTimer);
      this.micFlushTimer = null;
    }
    if (!this.micBytes) return null;
    const payload = Buffer.concat(this.micChunks, this.micBytes);
    this.micChunks = [];
    this.micBytes = 0;
    return payload;
  }

  /**
   * Emit whole 80 ms (60-100 ms window) frames. A sub-frame remainder is held for the next chunk
   * unless `force` is set — a socket swap must not silently drop the tail of what was said.
   */
  private flushMicBuffer(force = false): void {
    const payload = this.takeMicPayload();
    if (!payload) return;
    const sock = this.ws;
    if (!sock || sock.readyState !== WebSocket.OPEN || this.awaitingColdHistory || this.phase !== "live") {
      // Never drop audio across a swap: it becomes pre-roll for the socket being opened.
      this.pushPreRoll(payload);
      return;
    }

    const whole = Math.floor(payload.length / MIC_COALESCE_BYTES) * MIC_COALESCE_BYTES;
    if (whole === 0 && !force) {
      this.micChunks.push(payload);
      this.micBytes += payload.length;
      return;
    }
    let off = 0;
    for (; off + MIC_COALESCE_BYTES <= whole; off += MIC_COALESCE_BYTES) {
      this.sendAudioFrame(sock, payload.subarray(off, off + MIC_COALESCE_BYTES));
    }
    if (off < payload.length) {
      const rest = payload.subarray(off);
      if (force) this.sendAudioFrame(sock, rest);
      else {
        this.micChunks.push(rest);
        this.micBytes += rest.length;
      }
    }
  }

  private sendAudioFrame(sock: WebSocket, pcm16: Buffer): void {
    if (!pcm16.length) return;
    sock.send(
      JSON.stringify({
        realtimeInput: {
          audio: {
            data: pcm16.toString("base64"),
            mimeType: "audio/pcm;rate=16000",
          },
        },
      }),
    );
  }

  private pushPreRoll(chunk: Buffer): void {
    this.preRollChunks.push(chunk);
    this.preRollBytes += chunk.length;
    while (this.preRollBytes > MAX_PREROLL_BYTES && this.preRollChunks.length > 0) {
      const dropped = this.preRollChunks.shift()!;
      this.preRollBytes -= dropped.length;
    }
  }

  private drainPreRoll(): void {
    if (!this.preRollBytes || this.muted) {
      this.preRollChunks = [];
      this.preRollBytes = 0;
      return;
    }
    const all = Buffer.concat(this.preRollChunks, this.preRollBytes);
    this.preRollChunks = [];
    this.preRollBytes = 0;
    for (let off = 0; off < all.length; off += MIC_COALESCE_BYTES) {
      this.pushMicPcm16(all.subarray(off, Math.min(off + MIC_COALESCE_BYTES, all.length)));
    }
    // A drained ring is already-old audio: send its tail rather than holding a sub-frame of it.
    this.flushMicBuffer(true);
  }

  setMuted(muted: boolean): void {
    const next = Boolean(muted);
    if (next === this.muted) return;
    this.muted = next;
    if (next) {
      // Muting drops what was said before the mute took effect rather than sending it late.
      this.takeMicPayload();
      this.sendAudioStreamEnd();
    }
  }

  sendAudioStreamEnd(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
  }

  sendRealtimeText(text: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.awaitingColdHistory) return;
    this.ws.send(JSON.stringify({ realtimeInput: { text } }));
  }

  sendToolResponses(responses: GeminiFunctionResponse[], callSocketSeq?: number): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    if (callSocketSeq !== undefined && callSocketSeq !== this.socketSeq) return false;
    const valid: GeminiFunctionResponse[] = [];
    for (const r of responses) {
      const recordedSeq = this.inFlightToolIds.get(r.id);
      if (recordedSeq !== undefined) {
        this.inFlightToolIds.delete(r.id);
        if (recordedSeq !== this.socketSeq) continue; // the call belonged to a socket we no longer have
      }
      valid.push(r);
    }
    if (!valid.length) return false;
    this.ws.send(JSON.stringify({ toolResponse: { functionResponses: valid } }));
    this.maybeFinishQuietRotation();
    return true;
  }

  /** Stop the socket but keep the handle and the checkpoint: `wake()` resumes where this left off. */
  async sleep(reason = "idle_sleep"): Promise<void> {
    if (this.phase === "sleeping" || this.phase === "closed") return;
    this.intentionalClose = true;
    // A sleeping call starts a fresh turn: the audio buffered around this instant is stale.
    this.takeMicPayload();
    this.sendAudioStreamEnd();
    this.clearGoAwayTimer();
    this.rotationPending = false;
    this.preDisconnectPromptTokens = this.lastPromptTokenCount || this.preDisconnectPromptTokens;
    const old = this.ws;
    this.ws = null;
    this.socketSeq++;
    this.turnOpen = false;
    this.generating = false;
    this.inFlightToolIds.clear();
    if (old && (old.readyState === WebSocket.OPEN || old.readyState === WebSocket.CONNECTING)) {
      try {
        old.close(1000, reason);
      } catch {
        /* ignore */
      }
    }
    this.setPhase("sleeping", reason);
  }

  async wake(preRoll?: Buffer[]): Promise<void> {
    if (this.phase === "live" || this.phase === "connecting") {
      if (preRoll?.length) {
        for (const c of preRoll) this.pushMicPcm16(c);
      }
      return;
    }
    this.setPhase("reconnecting", "wake");
    await this.connect({ preRoll });
  }

  async close(): Promise<void> {
    this.intentionalClose = true;
    this.clearGoAwayTimer();
    this.rotationPending = false;
    this.takeMicPayload();
    this.preRollChunks = [];
    this.preRollBytes = 0;
    this.inFlightToolIds.clear();
    const old = this.ws;
    this.ws = null;
    this.socketSeq++;
    if (old) {
      try {
        old.close(1000, "closed");
      } catch {
        /* ignore */
      }
    }
    this.setPhase("closed");
  }

  private handleServerFrame(frame: Record<string, unknown>, seq: number): void {
    this.serverFrameSeq++;

    const usage = frame.usageMetadata as { promptTokenCount?: number } | undefined;
    if (usage && typeof usage.promptTokenCount === "number") {
      const current = usage.promptTokenCount;
      if (this.wasResumedConnect && !this.canaryChecked) {
        this.canaryChecked = true;
        // WHY: Ephemeral/resumption silent context drops keep setupComplete OK while promptTokenCount
        // collapses back near the bare-prompt baseline. Emitting an event is not enough: the handle
        // itself is now a lie, so drop it and force the next connect through the cold-header path.
        if (
          this.preDisconnectPromptTokens > 800 &&
          current < Math.max(300, this.preDisconnectPromptTokens * 0.45)
        ) {
          this.resumeHandle = "";
          this.needsColdReseed = true;
          this.emit("resumptionContextLost", {
            before: this.preDisconnectPromptTokens,
            after: current,
            checkpointSeq: this.lastResumableSeq,
          });
        }
      }
      this.lastPromptTokenCount = current;
      this.emit("usageMetadata", usage);
    }

    const resumption = frame.sessionResumptionUpdate as
      | { newHandle?: string; resumable?: boolean; lastConsumedClientMessageIndex?: number }
      | undefined;
    if (resumption) {
      // Only a `resumable:true` handle is valid: during generation or a tool call the server reports
      // `resumable:false`, and that checkpoint points at a state BEFORE the in-flight work.
      if (resumption.resumable === true && resumption.newHandle) {
        this.resumeHandle = resumption.newHandle;
        const index = resumption.lastConsumedClientMessageIndex;
        this.lastResumableSeq =
          typeof index === "number" && Number.isFinite(index) ? index : this.serverFrameSeq;
      }
      this.emit("sessionResumptionUpdate", resumption);
    }

    const goAway = frame.goAway as { timeLeft?: string } | undefined;
    if (goAway) {
      this.emit("goAway", goAway);
      this.scheduleQuietRotation(goAway.timeLeft);
    }

    const toolCall = frame.toolCall as
      | { functionCalls?: Array<{ id?: string; name?: string; args?: Record<string, unknown> }> }
      | undefined;
    if (toolCall?.functionCalls?.length) {
      const calls: GeminiFunctionCall[] = [];
      for (const raw of toolCall.functionCalls) {
        if (!raw.name) continue;
        const id = raw.id ?? `call_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        this.inFlightToolIds.set(id, seq);
        calls.push({ id, name: raw.name, args: raw.args ?? {} });
      }
      if (calls.length) this.emit("toolCall", { calls, socketSeq: seq });
    }

    const cancel = frame.toolCallCancellation as { ids?: string[] } | undefined;
    if (cancel?.ids?.length) {
      for (const id of cancel.ids) this.inFlightToolIds.delete(id);
      this.emit("toolCallCancellation", cancel.ids);
      this.maybeFinishQuietRotation();
    }

    const content = frame.serverContent as Record<string, unknown> | undefined;
    if (!content) return;

    if (content.interrupted === true) {
      this.generating = false;
      this.emit("interrupted");
    }

    const inTx = content.inputTranscription as { text?: string; finished?: boolean } | undefined;
    if (inTx && typeof inTx.text === "string" && inTx.text.length > 0) {
      this.emit("inputTranscription", { text: inTx.text, finished: Boolean(inTx.finished) });
    }

    const modelTurn = content.modelTurn as
      | { parts?: Array<{ inlineData?: { mimeType?: string; data?: string } }> }
      | undefined;
    if (modelTurn?.parts?.length) {
      this.turnOpen = true;
      this.generating = true;
      for (const part of modelTurn.parts) {
        const inline = part.inlineData;
        if (inline?.data && String(inline.mimeType ?? "").startsWith("audio/pcm")) {
          this.emit("audio", Buffer.from(inline.data, "base64"));
        }
      }
    }

    const outTx = content.outputTranscription as { text?: string; finished?: boolean } | undefined;
    if (outTx && typeof outTx.text === "string" && outTx.text.length > 0) {
      this.emit("outputTranscription", { text: outTx.text, finished: Boolean(outTx.finished) });
    }

    if (content.generationComplete === true) {
      this.generating = false;
      this.emit("generationComplete");
    }

    if (content.turnComplete === true) {
      this.turnOpen = false;
      this.generating = false;
      this.emit("turnComplete");
      this.drainSilentSyncQueue();
      this.maybeFinishQuietRotation();
    }
  }

  private scheduleQuietRotation(timeLeftRaw?: string): void {
    if (this.rotationPending) return;
    this.rotationPending = true;
    const ms = parseProtoDurationMs(timeLeftRaw);
    const budgetMs = ms > 0 ? Math.max(1500, Math.min(45_000, ms - 4000)) : 12_000;
    this.goAwayTimer = setTimeout(() => {
      this.goAwayTimer = null;
      // The deadline has arrived, but the order is "rotate at a quiet boundary": if a generation or
      // a tool call is still in flight the intent stays armed and turnComplete/toolResponse finishes it.
      if (this.isTurnBusy() || this.inFlightToolIds.size > 0) return;
      void this.performRotation();
    }, budgetMs);
    this.maybeFinishQuietRotation();
  }

  private maybeFinishQuietRotation(): void {
    if (!this.rotationPending) return;
    if (!this.turnOpen && !this.generating && this.inFlightToolIds.size === 0) {
      this.clearGoAwayTimer();
      void this.performRotation();
    }
  }

  private clearGoAwayTimer(): void {
    if (this.goAwayTimer) {
      clearTimeout(this.goAwayTimer);
      this.goAwayTimer = null;
    }
  }

  private async performRotation(): Promise<void> {
    if (this.phase === "closed" || this.phase === "rotating") return;
    // Never swap under a live generation: the new socket would not know what the old one was saying.
    if (this.isTurnBusy() || this.inFlightToolIds.size > 0) return;
    this.rotationPending = false;
    this.intentionalClose = true;
    // Flush while the socket is still the live one: after setPhase("rotating") the tail would be
    // routed to pre-roll for the *next* socket instead of out on this one.
    this.flushMicBuffer(true);
    this.sendAudioStreamEnd();
    this.setPhase("rotating", "goAway");
    this.preDisconnectPromptTokens = this.lastPromptTokenCount || this.preDisconnectPromptTokens;
    const old = this.ws;
    this.ws = null;
    this.socketSeq++;
    this.inFlightToolIds.clear();
    if (old) {
      try {
        old.close(1000, "goAway_rotate");
      } catch {
        /* ignore */
      }
    }
    try {
      await this.connect();
    } catch (err) {
      this.fail("rotation_failed", err instanceof Error ? err : new Error(String(err)));
    }
  }

  private onSocketClosed(code: number, reason: string): void {
    if (this.intentionalClose) return;
    this.preDisconnectPromptTokens = this.lastPromptTokenCount || this.preDisconnectPromptTokens;
    this.turnOpen = false;
    this.generating = false;
    this.inFlightToolIds.clear();

    // 1008: the server's inactivity close. A real Live call must survive it, so sleep() (keeping the
    // handle/checkpoint for wake) rather than failing the call the way the current plugin does.
    const authFatal = /api key|permission|auth|quota|billing/i.test(reason);
    if (code === 1008 && !authFatal) {
      void this.sleep("1008_inactivity");
      return;
    }
    if (code === 1008 || authFatal) {
      this.fail(`Gemini fatal close (${code}): ${reason}`, new Error(`Gemini fatal close (${code}): ${reason}`));
      return;
    }

    // 1007 after a successful setup: the stored resumption handle is stale. Clear it once (the current
    // plugin retries the same handle four times) and the reconnect below is cold.
    if (code === 1007 && this.resumeHandle) this.resumeHandle = "";

    void this.reconnectLoop(`close_${code}`);
  }

  private async reconnectLoop(reason: string): Promise<void> {
    const maxAttempts = Math.max(1, this.cfg.reconnectAttempts ?? 4);
    this.setPhase("reconnecting", reason);
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if (this.intentionalClose || this.phase === "closed") return;
      this.reconnectAttempts = attempt;
      try {
        await this.connect({ forceCold: this.needsColdReseed });
        return;
      } catch (err) {
        // A failed resume is a stale handle, never a fatal call: fall back cold on the next attempt.
        this.resumeHandle = "";
        if (attempt === maxAttempts) {
          this.fail(`Reconnect failed: ${String((err as Error)?.message ?? err)}`, err instanceof Error ? err : new Error(String(err)));
          return;
        }
        await new Promise((r) => setTimeout(r, 500 * attempt));
      }
    }
  }
}

function rawDataToUtf8(data: RawData): string {
  if (Buffer.isBuffer(data)) return data.toString("utf8");
  if (Array.isArray(data)) return Buffer.concat(data).toString("utf8");
  return Buffer.from(data).toString("utf8");
}
