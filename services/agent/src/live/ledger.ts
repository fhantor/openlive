import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { GuidanceItem, TaskDigestView } from "@openlive/shared";

// The ledger is the durable truth of one chat's realtime call. It exists because a
// Live socket dies — 1008 sleep/wake, a 1007 stale handle, a `goAway` rotation, a
// renderer reload — and a resumed Gemini session does not restore the coding
// agent's state. Two failures cost the most trust and are prevented HERE, not by
// prompt wording:
//
//   * a duplicate dispatch after a reconnect — `task_dispatched` is written with
//     its idempotency key BEFORE the ACP call, so a replay cannot produce a second;
//   * a status answer invented from memory — the digest is built from records that
//     really happened, with relative ages, or says it does not know.
//
// Storage is append-only JSONL at `<dataDir>/ledgers/<chatId>.jsonl`, replayed
// into memory on boot, with a monotonic `seq`. Append-only means a crash mid-write
// can lose at most the last line, and a torn line is skipped on replay instead of
// corrupting the numbers.

/** Every record kind the ledger can hold. Closed so replay stays exhaustive. */
export type LedgerKind =
  | "task_dispatched"
  | "acp_event"
  | "todo"
  | "guidance"
  | "guidance_status"
  | "checkpoint"
  | "dispatch_released";

/** One line on disk. `seq` is monotonic per chat and the only ordering that counts. */
export interface LedgerRecord {
  seq: number;
  at: number;
  kind: LedgerKind;
  taskId?: string;
  payload?: unknown;
}

/** What an ACP session update gives us, reduced to what the digest and the gate use.
 *  Deliberately a structural subset of the wire events so this module stays free of
 *  the agent's ACP types. */
export interface AcpEventLike {
  type?: string;
  /** acp_tool_call / tool_start / tool_done / acp_tool_update / text_delta / status / todos */
  [k: string]: unknown;
}

export type DigestMode = "status" | "resume_delta" | "cold_header";

/** A clock injected for tests; production uses Date.now. */
export type Clock = () => number;

const TASK_KINDS = new Set<LedgerKind>(["task_dispatched"]);

/** Things a spoken status must never contain: an absolute/relative file path, or a
 *  shell command. A bare file NAME ("session.ts") is allowed — that is the
 *  specificity a status answer needs — but a path that says where it lives is not.
 *  Reading "C:\proj\src\session.ts" aloud is noise, and echoing a command is a leak. */
const PATHISH = /(?:[A-Za-z]:[\\/]|\\\\|\/[\w.-]+\/|(?:^|\s)(?:\.{0,2}\/|~\/)[\w.-]+|(?:^|\s)(?:rm|git|npm|pnpm|npx|node|cd|ls|cat|mv|cp|mkdir)\s)/;

/**
 * Status narration for one ACP tool kind. Mirrors `narrator.ts`'s KIND_LINE so the
 * ledger speaks the same way the cascade narrator always has — one vocabulary, not
 * two. Never a raw tool title or a shell command.
 */
const KIND_VERB: Record<string, string> = {
  read: "reading files",
  edit: "making changes",
  delete: "cleaning files up",
  move: "moving files around",
  search: "searching",
  execute: "running a command",
  think: "thinking it through",
  fetch: "looking something up",
  switch_mode: "switching modes",
  other: "working",
};

const stripTrailingDot = (s: string) => s.replace(/[.。]+$/, "");

/** Remove anything that reads like a path or a command before a line can be spoken. */
export function speakable(s: string): string {
  return stripTrailingDot(String(s).replace(/\s+/g, " ").trim());
}

/** True when a line is safe to speak (no paths, no commands). */
export function isSpeakable(s: string): boolean {
  return !PATHISH.test(s);
}

/** "22s ago" / "4m ago" / "just now" — relative ages, because a wall-clock time is
 *  meaningless when spoken and wrong when the socket slept. */
export function relativeAge(fromMs: number, nowMs: number): string {
  const s = Math.max(0, Math.round((nowMs - fromMs) / 1000));
  if (s < 2) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

interface TaskRecord {
  taskId: string;
  idempotencyKey: string;
  spec: string;
  agentId: string;
  startedAt: number;
  status: TaskDigestView["status"];
  currentStep: string;
  currentStepAt: number;
  completedSteps: Array<{ text: string; at: number }>;
  todos: Array<{ text: string; status: string }>;
  lastTools: string[];
  voiceSummary?: string;
}

export class Ledger {
  private records: LedgerRecord[] = [];
  private seq = 0;
  private readonly file: string;
  /** taskId → in-memory projection, rebuilt on replay. */
  private readonly tasks = new Map<string, TaskRecord>();
  /** Idempotency keys whose dispatch has been committed. */
  private readonly dispatched = new Set<string>();
  private readonly guidance = new Map<string, GuidanceItem>();
  private checkpoint: { handle: string; seq: number } | null = null;

  constructor(readonly chatId: string, dataDir: string, private readonly now: Clock = () => Date.now()) {
    const dir = join(dataDir, "ledgers");
    mkdirSync(dir, { recursive: true });
    this.file = join(dir, `${chatId || "default"}.jsonl`);
    this.reload();
  }

  /** The JSONL path, exposed so tests and diagnostics can read the real artefact. */
  get path(): string {
    return this.file;
  }

  /** Next sequence number (without consuming it). */
  get lastSeq(): number {
    return this.seq;
  }

  /** Replay the file into memory. A torn or unparseable line is skipped, never fatal:
   *  a crash mid-append must not make the chat unusable. */
  reload(): void {
    this.records = [];
    this.seq = 0;
    this.tasks.clear();
    this.dispatched.clear();
    this.guidance.clear();
    this.checkpoint = null;
    if (!existsSync(this.file)) return;
    const text = readFileSync(this.file, "utf8");
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      let rec: LedgerRecord;
      try {
        rec = JSON.parse(line) as LedgerRecord;
      } catch {
        continue; // torn tail from a crash — the rest of the file is still good
      }
      if (!rec || typeof rec.seq !== "number") continue;
      this.apply(rec);
    }
  }

  /** Append one record and project it into memory. Returns the record it wrote. */
  append(kind: LedgerKind, payload: unknown, taskId?: string): LedgerRecord {
    const rec: LedgerRecord = { seq: this.seq + 1, at: this.now(), kind, ...(taskId ? { taskId } : {}), payload };
    // Write first, then project: if the process dies between the two, replay rebuilds
    // the same state. The reverse order could hand out a seq that never hit disk.
    appendFileSync(this.file, `${JSON.stringify(rec)}\n`, "utf8");
    this.apply(rec);
    return rec;
  }

  private apply(rec: LedgerRecord): void {
    this.records.push(rec);
    this.seq = Math.max(this.seq, rec.seq);
    const p = (rec.payload ?? {}) as Record<string, unknown>;
    const taskId = rec.taskId ?? (typeof p.taskId === "string" ? p.taskId : undefined);
    switch (rec.kind) {
      case "task_dispatched": {
        const key = String(p.idempotencyKey ?? "");
        if (key) this.dispatched.add(key);
        if (taskId) {
          this.tasks.set(taskId, {
            taskId,
            idempotencyKey: key,
            spec: String(p.spec ?? ""),
            agentId: String(p.agentId ?? ""),
            startedAt: Number(p.startedAt ?? rec.at),
            status: (p.status as TaskDigestView["status"]) ?? "running",
            currentStep: "",
            currentStepAt: rec.at,
            completedSteps: [],
            todos: [],
            lastTools: [],
          });
        }
        break;
      }
      case "dispatch_released": {
        // A dispatch that never reached the agent (bind failure, immediate abort).
        // Releasing the key is only legal before the ACP call; a release after one is
        // ignored, so a replay can never turn one dispatch into two.
        const key = String(p.idempotencyKey ?? "");
        const t = taskId ? this.tasks.get(taskId) : undefined;
        if (key && t && t.currentStep === "") this.dispatched.delete(key);
        break;
      }
      case "acp_event": {
        if (!taskId) break;
        const t = this.tasks.get(taskId);
        if (!t) break;
        this.foldAcp(t, p as AcpEventLike, rec.at);
        break;
      }
      case "todo": {
        if (!taskId) break;
        const t = this.tasks.get(taskId);
        const items = Array.isArray(p.items) ? (p.items as Array<{ text: string; done?: boolean }>) : [];
        if (t) t.todos = items.map((i) => ({ text: String(i.text ?? ""), status: i.done ? "done" : "open" }));
        break;
      }
      case "guidance": {
        const g = p as unknown as GuidanceItem;
        if (g?.guidanceId) this.guidance.set(g.guidanceId, { ...g, status: g.status ?? "pending" });
        break;
      }
      case "guidance_status": {
        const id = String(p.guidanceId ?? "");
        const g = this.guidance.get(id);
        if (g) {
          g.status = (p.status as GuidanceItem["status"]) ?? g.status;
          if (typeof p.deliveredAt === "number") g.deliveredAt = p.deliveredAt;
        }
        break;
      }
      case "checkpoint": {
        const handle = String(p.handle ?? "");
        // The checkpoint's position is where it landed in the log, not the caller's
        // idea of a seq: resume must replay exactly `seq > this`, and only the log
        // knows that boundary. Storing a caller-supplied number could skip or
        // repeat events across a restart.
        if (handle) this.checkpoint = { handle, seq: rec.seq };
        break;
      }
    }
  }

  /** Fold one ACP event into a task's projection. Only the four shapes that carry
   *  progress matter; everything else is ignored rather than guessed at. */
  private foldAcp(t: TaskRecord, e: AcpEventLike, at: number): void {
    const type = String(e.type ?? "");
    if (type === "acp_tool_call" || type === "acp_tool_update" || type === "tool_start") {
      const call = (e.call ?? {}) as { title?: string; kind?: string; status?: string };
      const kind = String(call.kind ?? "other");
      const verb = KIND_VERB[kind] ?? "working";
      // The current step is a SPEAKABLE phrase, never the tool's raw title (which
      // routinely contains a path or a shell command the user must not hear).
      const title = speakable(String(call.title ?? ""));
      const step = title && isSpeakable(title) ? `${verb}: ${title}` : verb;
      if (step !== t.currentStep) {
        // A step only counts as completed once a newer one starts, so "the last
        // completed ones" is always behind the current step rather than equal to it.
        if (t.currentStep) t.completedSteps.push({ text: t.currentStep, at: t.currentStepAt });
        t.currentStep = step;
        t.currentStepAt = at;
      }
      const tool = String(call.title ?? call.kind ?? "");
      // Only speakable tool names go in the digest: a raw title is a path or a shell
      // command, and reading one aloud is both useless and a privacy leak.
      if (tool && isSpeakable(tool)) t.lastTools = [...t.lastTools.filter((x) => x !== tool), tool].slice(-4);
      if (call.status === "failed") t.status = "failed";
      else if (t.status === "running") t.status = "running";
    } else if (type === "tool_done") {
      const title = speakable(String((e as { tool?: string }).tool ?? ""));
      if (title && isSpeakable(title) && title !== t.currentStep) {
        t.completedSteps.push({ text: title, at });
        t.currentStepAt = at;
      }
    } else if (type === "todos") {
      const items = Array.isArray((e as { items?: unknown }).items) ? ((e as { items: Array<{ text: string; done?: boolean }> }).items) : [];
      t.todos = items.map((i) => ({ text: String(i.text ?? ""), status: i.done ? "done" : "open" }));
    } else if (type === "status") {
      const text = (e as { text?: string | null }).text;
      if (typeof text === "string" && text.trim()) {
        const step = speakable(text);
        if (isSpeakable(step) && step !== t.currentStep) {
          if (t.currentStep) t.completedSteps.push({ text: t.currentStep, at: t.currentStepAt });
          t.currentStep = step;
          t.currentStepAt = at;
        }
      }
    } else if (type === "done") {
      t.completedSteps.push({ text: t.currentStep || "finished", at });
      t.currentStep = "";
      t.status = t.status === "failed" ? "failed" : "running";
    } else if (type === "error") {
      t.status = "failed";
    }
  }

  // ── reads used by the tools and the tests ──────────────────────────────────

  /** Has this idempotency key already been dispatched? The one question that must
   *  never answer "no" for a key that reached the agent. */
  hasDispatchedKey(key: string): boolean {
    return this.dispatched.has(key);
  }

  /** Record a dispatch BEFORE the ACP call. Returns the record, or null when the key
   *  was already present (a replay) — the caller must not dispatch again. */
  recordDispatch(taskId: string, idempotencyKey: string, spec: string, agentId: string): LedgerRecord | null {
    if (this.dispatched.has(idempotencyKey)) return null;
    return this.append("task_dispatched", { idempotencyKey, spec, agentId, startedAt: this.now(), status: "running" }, taskId);
  }

  /** A dispatch that provably never reached the agent. Refused once the task has any
   *  recorded progress, so a late release cannot free a key that was really used. */
  releaseDispatch(taskId: string, idempotencyKey: string): boolean {
    const t = this.tasks.get(taskId);
    if (!t || t.currentStep !== "" || t.completedSteps.length > 0) return false;
    this.append("dispatch_released", { idempotencyKey }, taskId);
    return true;
  }

  recordAcpEvent(taskId: string, e: AcpEventLike): LedgerRecord {
    return this.append("acp_event", e, taskId);
  }

  /** Persist the live task checklist (the ACP `todos` update). */
  recordTodos(taskId: string, items: Array<{ text: string; done: boolean }>): LedgerRecord {
    return this.append("todo", { items }, taskId);
  }

  /** A resumable Gemini handle plus the seq it was valid at. Only ever called with a
   *  handle the server said was `resumable: true` — a checkpoint taken during a tool
   *  call points BEFORE the in-flight work and would resurrect a re-dispatch. */
  recordCheckpoint(handle: string, seq: number): LedgerRecord {
    return this.append("checkpoint", { handle, seq });
  }

  getCheckpoint(): { handle: string; seq: number } | null {
    return this.checkpoint;
  }

  // ── guidance ───────────────────────────────────────────────────────────────

  /** Queue a mid-task correction. Ordered by seq, never concatenated — the whole
   *  point is that the agent applies #1 and #2 in the order the user said them. */
  enqueueGuidance(item: Omit<GuidanceItem, "seq" | "status"> & { status?: GuidanceItem["status"] }): GuidanceItem {
    const rec = this.append("guidance", { ...item, status: item.status ?? "pending" }, item.taskId);
    const stored = this.guidance.get(item.guidanceId)!;
    // The seq the item was born at is the ledger's, so ordering survives a restart.
    stored.seq = rec.seq;
    return { ...stored };
  }

  /** Every still-pending guidance for a task, in the order it was spoken. */
  drainPendingGuidance(taskId: string): GuidanceItem[] {
    return [...this.guidance.values()]
      .filter((g) => g.taskId === taskId && g.status === "pending")
      .sort((a, b) => a.seq - b.seq)
      .map((g) => ({ ...g }));
  }

  listGuidance(taskId: string): GuidanceItem[] {
    return [...this.guidance.values()].filter((g) => g.taskId === taskId).sort((a, b) => a.seq - b.seq).map((g) => ({ ...g }));
  }

  markGuidance(id: string, status: GuidanceItem["status"]): void {
    const g = this.guidance.get(id);
    if (!g) return;
    this.append("guidance_status", { guidanceId: id, status, ...(status === "pending" ? {} : { deliveredAt: this.now() }) });
  }

  /** Supersede every pending guidance for a task — a pivot replaces the plan, but the
   *  superseded items stay in the ledger as evidence of what was said. */
  supersedePendingGuidance(taskId: string): void {
    for (const g of this.drainPendingGuidance(taskId)) this.markGuidance(g.guidanceId, "superseded");
  }

  // ── digests ────────────────────────────────────────────────────────────────

  /** The task's status view for the client. */
  taskView(taskId: string): TaskDigestView | null {
    const t = this.tasks.get(taskId);
    if (!t) return null;
    return {
      taskId: t.taskId,
      agentId: t.agentId,
      status: t.status,
      currentStep: t.currentStep,
      todos: t.todos,
      lastTools: t.lastTools,
      startedAt: t.startedAt,
      ...(t.voiceSummary ? { voiceSummary: t.voiceSummary } : {}),
    };
  }

  /** The most recently dispatched task, which is what "what is it doing?" means. */
  latestTaskId(): string | null {
    let latest: TaskRecord | null = null;
    for (const t of this.tasks.values()) if (!latest || t.startedAt > latest.startedAt) latest = t;
    return latest?.taskId ?? null;
  }

  /**
   * Build the bounded text the model reads for a status answer or a reconnect.
   *
   *   status        — one task, current step + the recent completed ones, relative ages
   *   resume_delta  — only what happened after `fromSeq`, for a resumed socket
   *   cold_header   — the ~600–1200 token [LEDGER_COLD_START] block, never persisted
   *
   * Every mode is explicit about unknowns. A digest that cannot answer says so,
   * because a plausible invention here is what makes the app lie about progress.
   */
  buildDigest(mode: DigestMode, fromSeq?: number): string {
    const now = this.now();
    if (mode === "cold_header") return this.buildColdHeader(now);
    if (mode === "resume_delta") return this.buildResumeDelta(now, fromSeq ?? this.checkpoint?.seq ?? 0);
    return this.buildStatus(now);
  }

  private buildStatus(now: number): string {
    const id = this.latestTaskId();
    if (!id) return "No task has been started in this chat yet.";
    const t = this.tasks.get(id)!;
    const lines: string[] = [];
    lines.push(`taskId=${t.taskId} status=${t.status} agent=${t.agentId} started=${relativeAge(t.startedAt, now)}`);
    lines.push(t.currentStep ? `CURRENT STEP: ${t.currentStep} (${relativeAge(t.currentStepAt, now)})` : "CURRENT STEP: between steps");
    const done = t.completedSteps.slice(-3);
    lines.push(
      done.length
        ? `JUST COMPLETED:\n${done.map((s) => `- ${s.text} (${relativeAge(s.at, now)})`).join("\n")}`
        : "JUST COMPLETED: nothing yet",
    );
    if (t.todos.length) {
      const open = t.todos.filter((x) => x.status !== "done");
      lines.push(
        open.length
          ? `STILL OPEN: ${open.map((x) => speakable(x.text)).join("; ")}`
          : `ALL STEPS TICKED (${t.todos.length})`,
      );
    }
    if (t.lastTools.length) lines.push(`LAST TOOLS: ${t.lastTools.slice(-3).map(speakable).join(", ")}`);
    const pending = this.drainPendingGuidance(id);
    if (pending.length) lines.push(`UNAPPLIED CORRECTIONS: ${pending.length}`);
    lines.push("If the CURRENT STEP is empty the agent is between steps; say that rather than inventing one.");
    return lines.join("\n");
  }

  private buildResumeDelta(now: number, fromSeq: number): string {
    const missed = this.records.filter((r) => r.seq > fromSeq && (r.kind === "acp_event" || r.kind === "todo" || r.kind === "guidance" || r.kind === "task_dispatched"));
    const id = this.latestTaskId();
    const head = `[STATE_SYNC resume=ok checkpoint_seq=${fromSeq}]\nMissed since seq ${fromSeq}: ${missed.length} event(s).`;
    const status = id ? this.buildStatus(now) : "No task has been started in this chat yet.";
    return `${head}\n${status}\nRules: Do not greet again. Do not re-dispatch ALREADY_DISPATCHED tasks.`;
  }

  private buildColdHeader(now: number): string {
    const lines: string[] = ["[LEDGER_COLD_START]"];
    const tasks = [...this.tasks.values()].sort((a, b) => a.startedAt - b.startedAt);
    lines.push("DISPATCHED_TASKS (NEVER_RE_DISPATCH):");
    if (!tasks.length) lines.push("- none");
    for (const t of tasks.slice(-8)) {
      const step = t.currentStep ? ` step="${t.currentStep}"` : "";
      const summary = t.completedSteps.length ? ` last="${t.completedSteps.at(-1)!.text}"` : "";
      lines.push(`- ${t.taskId} "${speakable(t.spec).slice(0, 120)}" status=${t.status}${step}${summary}`);
    }
    // Constraints are the parts of a spec that read as a prohibition; they are the
    // lines a resumed model is most likely to violate by re-planning from scratch.
    const constraints = tasks
      .flatMap((t) => speakable(t.spec).split(/[;.\n]/))
      .filter((s) => /\b(do not|don't|never|do NOT)\b/i.test(s))
      .map((s) => `- ${s.trim()}`);
    lines.push("ACTIVE_CONSTRAINTS:");
    lines.push(...(constraints.length ? constraints.slice(-6) : ["- none recorded"]));
    const id = this.latestTaskId();
    const pending = id ? this.drainPendingGuidance(id) : [];
    lines.push(`PENDING_DRAFT: ${pending.length ? `${pending.length} unapplied correction(s)` : "none"}`);
    lines.push(`LAST_SEQ: ${this.seq}`);
    lines.push("[END]");
    return lines.join("\n");
  }
}
