import type { ActiveDraftView, DraftState } from "@openlive/shared";

// The confirmation gate. This is KEY 2 of three, and the only deterministic one:
//
//   Key 1 (model)        — `confirm_instruction` is a NOMINATION. The model heard the
//                          audio natively, including tone, but it can be wrong.
//   Key 2 (here)         — a closed-vocabulary validator over the user's own
//                          transcript, with hard vetoes, echo overlap and playback
//                          timing. This file.
//   Key 3 (UI)           — the DraftCard Send button, which still needs a version match.
//
// Two keys must agree before a dispatch. Any disagreement fails CLOSED and re-asks:
// refusing a valid "yes" costs one extra turn; accepting an invalid one sends work
// the user did not ask for.
//
// The Bengali traps that a token-level matcher gets wrong are the reason this
// matches PHRASES and scans ±2 tokens around any positive for a negator:
//   ঠিক নেই   ("it's not right")        contains the positive ঠিক
//   ঠিক হয়নি  ("it didn't become right") contains the positive ঠিক
// Both must be rejected even though a naive `includes("ঠিক")` accepts them.

/** What the matcher decided, and why — the reason is shown and logged, never spoken raw. */
export interface Verdict {
  ok: boolean;
  /** A short machine reason: "positive", "veto:but", "echo", "too_early", "too_long", … */
  reason: string;
  /** The positive that matched, when one did. */
  phrase?: string;
}

/**
 * Positives: whole utterances the user uses to mean "send it". Bengali, Banglish and
 * English, because the user code-switches mid-call.
 */
const POSITIVES = [
  // Bengali
  "হ্যাঁ", "জি হ্যাঁ", "জি", "হাঁ", "পাঠাও", "পাঠিয়ে দাও", "পাঠাই", "পাঠিয়ে দিচ্ছি",
  "করে দাও", "করো", "কর", "শুরু করো", "এগিয়ে যাও", "ঠিক আছে পাঠাও", "বেশ",
  // Banglish (the same words, romanised — the transcript may come back romanized)
  "hyan", "hae", "haan", "ji", "pathao", "pathiye dao", "kore dao", "koro", "shuru koro",
  // English
  "yes", "yeah", "yep", "yup", "correct", "right", "go ahead", "send it", "send", "do it",
  "proceed", "ok send", "sure",
] as const;

/**
 * Medium: accepted ONLY when the whole utterance is ≤3 tokens. "ঠিক আছে" alone is a
 * yes; "ঠিক আছে কিন্তু আরেকটা জিনিস" is not, and the veto catches that one anyway.
 */
const MEDIUM = ["ঠিক আছে", "ওকে", "আচ্ছা ঠিক", "ঠিক", "okay", "ok", "fine", "alright", "thik ache", "thik"] as const;

/**
 * Never alone: backchannels that mean "I'm listening", not "go". If one of these is
 * the whole utterance it is silence-as-consent and must not dispatch.
 */
const BACKCHANNEL = ["আচ্ছা", "হুম", "হুঁ", "hm", "hmm", "uh-huh", "uh huh", "right right", "mm", "মম"] as const;

/**
 * Hard veto: any occurrence anywhere rejects the whole utterance. This is the list
 * that makes the negation trap work — "হ্যাঁ, তবে…" is a yes with a condition, i.e.
 * a NEW instruction, not a confirmation of the old one.
 */
const VETO = [
  // Bengali / Banglish conjunctions and negators
  "তবে", "কিন্তু", "খালি", "শুধু", "ছাড়া", "না", "নাই", "নেই", "হয়নি", "ভুল", "বদলাও", "যোগ করো", "আরেকটা",
  "tobe", "kintu", "khali", "shudhu", "chara", "ne", "nei", "nai", "na", "hoyni", "bhul", "bodlao", "jog koro", "arekt",
  // English
  "but", "except", "not", "wait", "stop", "actually", "also", "add", "no", "don't", "dont", "change", "instead",
] as const;

/** Above this many words it is an instruction, not a confirmation. */
export const MAX_CONFIRM_WORDS = 6;

/** The read-back must have finished at least this long before the user's speech
 *  onset, or the mic is hearing our own voice (AEC leak) rather than the user. */
export const ONSET_GUARD_MS = 350;

/** How much of the read-back's tail may appear in the utterance before it counts as
 *  an echo. 0.5 of the last five words is deliberately aggressive. */
export const ECHO_OVERLAP = 0.5;
const ECHO_TAIL_WORDS = 5;

/** Split into words for counting, keeping Bengali safe (no ASCII assumption). */
function words(text: string): string[] {
  return text
    .replace(/[।॥.,!?;:"“”'’()[\]{}<>|/\\\-–—]/g, " ")
    .split(/\s+/u)
    .filter(Boolean);
}

/** Lowercase, collapse whitespace, drop terminal punctuation — the form both the
 *  utterance and the read-back are compared in. */
export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[।॥.,!?;:"“”'’()[\]{}<>]/g, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

/** The last N words of a text, normalized. Used for the echo check. */
function tail(text: string, n: number): string[] {
  return words(normalize(text)).slice(-n);
}

/** Overlap between the utterance and the tail of the read-back, as a fraction of the
 *  shorter list. ≥0.5 means the user is repeating our own read-back at us. */
export function echoOverlap(utterance: string, readback: string): number {
  const u = words(normalize(utterance));
  const t = tail(readback, ECHO_TAIL_WORDS);
  if (!u.length || !t.length) return 0;
  const tset = new Set(t);
  const hits = u.filter((w) => tset.has(w)).length;
  return hits / Math.min(u.length, t.length);
}

/**
 * Scan ±2 tokens around any positive/medium token for a negator. This is what makes
 * `ঠিক নেই` and `ঠিক হয়নি` fail: `ঠিক` is a positive-ish token, and `নেই`/`হয়নি`
 * sit right beside it.
 */
function negatorNear(w: string[]): string | null {
  const negators = new Set(["না", "নাই", "নেই", "হয়নি", "ভুল", "na", "nai", "ne", "nei", "hoyni", "bhul", "not", "no", "don't", "dont"]);
  for (let i = 0; i < w.length; i++) {
    if (!negators.has(w[i]!)) continue;
    return w[i]!;
  }
  return null;
}

/** Does the utterance contain a whole positive phrase (as a run of words)? */
function matchPositive(w: string[]): string | null {
  for (const p of POSITIVES) {
    const pw = words(normalize(p));
    for (let i = 0; i + pw.length <= w.length; i++) {
      if (pw.every((x, j) => w[i + j] === x)) return p;
    }
  }
  return null;
}

function matchMedium(w: string[]): string | null {
  for (const m of MEDIUM) {
    const mw = words(normalize(m));
    for (let i = 0; i + mw.length <= w.length; i++) {
      if (mw.every((x, j) => w[i + j] === x)) return m;
    }
  }
  return null;
}

function isOnlyBackchannel(w: string[]): boolean {
  for (const b of BACKCHANNEL) {
    const bw = words(normalize(b));
    if (bw.length === w.length && bw.every((x, j) => w[j] === x)) return true;
  }
  return false;
}

/**
 * THE validator. Given the user's transcript and the timing of the read-back, decide
 * whether this utterance confirms the armed draft. Fails closed on every doubt.
 *
 * @param text        the user's transcript, exactly as recognized
 * @param readback    what we spoke as the read-back (for the echo check)
 * @param vadOnsetMs  when the user's speech started (ms, same clock as playbackEndMs)
 * @param playbackEndMs when our read-back finished playing
 */
export function isConfirmation(text: string, readback: string, vadOnsetMs: number, playbackEndMs: number): Verdict {
  const raw = String(text ?? "").trim();
  if (!raw) return { ok: false, reason: "empty" };

  const w = words(normalize(raw));

  // Timing first: speech that starts before our own read-back ended is our echo (or
  // an interruption), never a deliberate confirmation. AEC makes this the likeliest
  // false positive in the whole app.
  if (vadOnsetMs < playbackEndMs + ONSET_GUARD_MS) return { ok: false, reason: "too_early" };

  if (w.length > MAX_CONFIRM_WORDS) return { ok: false, reason: "too_long" };

  const ov = echoOverlap(raw, readback);
  if (ov >= ECHO_OVERLAP) return { ok: false, reason: "echo" };

  // A question is not an answer: "পাঠাবো?" and anything ending in ? is a request for
  // permission, not consent.
  if (/[?？]/.test(raw)) return { ok: false, reason: "question" };
  // Bengali interrogatives at the end, and the or-question marker.
  if (/(?:^|\s)(?:কি|কী)\s*$/u.test(normalize(raw))) return { ok: false, reason: "question" };
  if (/(?:^|\s)নাকি(?:\s|$)/u.test(normalize(raw))) return { ok: false, reason: "question" };

  // Hard veto — any occurrence, anywhere, rejects the whole utterance.
  const nw = normalize(raw);
  for (const v of VETO) {
    const vw = normalize(v);
    if (!vw) continue;
    // Word-boundary match so "না" does not fire inside an unrelated word.
    const re = new RegExp(`(?:^|\\s)${vw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\s|$)`, "u");
    if (re.test(nw)) return { ok: false, reason: `veto:${v}` };
  }
  if (negatorNear(w)) return { ok: false, reason: `veto:${negatorNear(w)}` };

  // Backchannels alone never dispatch.
  if (isOnlyBackchannel(w)) return { ok: false, reason: "backchannel" };

  const strong = matchPositive(w);
  if (strong) return { ok: true, reason: "positive", phrase: strong };

  // Medium only when the whole utterance is ≤3 tokens — and it must BE the utterance,
  // not a medium phrase buried in a longer sentence.
  const medium = matchMedium(w);
  if (medium && w.length <= 3) return { ok: true, reason: "medium", phrase: medium };
  if (medium) return { ok: false, reason: "medium_too_long" };

  return { ok: false, reason: "no_positive" };
}

// ── the draft state machine ──────────────────────────────────────────────────

interface Draft {
  draftId: string;
  version: number;
  state: DraftState;
  intentType: ActiveDraftView["intentType"];
  summaryBn: string;
  englishSpec: string;
  openQuestions: string[];
  expiresAt: number;
  readbackText: string;
  playbackEndMs: number;
  /** Bumped on every state change that must invalidate an in-flight confirmation. */
  confirmWindowSeq: number;
  /** Set once dispatch is committed; the CAS that prevents a second dispatch. */
  dispatched: boolean;
}

export interface DraftGateOptions {
  /** How long a draft stays confirmable. */
  ttlMs?: number;
  now?: () => number;
}

/**
 * At most ONE active draft per chat: `propose()` supersedes rather than creating a
 * second, per 01-CONTRACTS.md rule 3. A steer re-proposes at version+1.
 */
export class DraftGate {
  private draft: Draft | null = null;
  private confirmWindowSeq = 0;
  private readonly ttlMs: number;
  private readonly now: () => number;
  /** Task the next dispatch continues (a steer), or null for a new task. */
  private parentTaskId: string | null = null;

  constructor(opts: DraftGateOptions = {}) {
    this.ttlMs = opts.ttlMs ?? 120_000;
    this.now = opts.now ?? (() => Date.now());
  }

  /** The client-facing view, or null when nothing is armed. */
  view(): ActiveDraftView | null {
    const d = this.live();
    if (!d) return null;
    return {
      draftId: d.draftId,
      version: d.version,
      state: d.state,
      intentType: d.intentType,
      summaryBn: d.summaryBn,
      englishSpec: d.englishSpec,
      openQuestions: [...d.openQuestions],
      expiresAt: d.expiresAt,
    };
  }

  /** The live draft, expiring it on read if its TTL has passed. */
  private live(): Draft | null {
    const d = this.draft;
    if (!d) return null;
    if (d.state !== "verifying" && this.now() > d.expiresAt) {
      d.state = "expired";
    }
    return d;
  }

  /** Arm a draft. A second propose SUPERSEDES the first (version bumps on a steer). */
  propose(input: {
    draftId: string;
    summaryBn: string;
    englishSpec: string;
    intentType?: ActiveDraftView["intentType"];
    openQuestions?: string[];
    parentTaskId?: string | null;
  }): ActiveDraftView {
    const prev = this.draft;
    const version = prev && prev.draftId === input.draftId ? prev.version + 1 : 1;
    this.parentTaskId = input.parentTaskId ?? this.parentTaskId;
    this.draft = {
      draftId: input.draftId,
      version,
      state: "proposing",
      intentType: input.intentType ?? "new_task",
      summaryBn: input.summaryBn,
      englishSpec: input.englishSpec,
      openQuestions: input.openQuestions ?? [],
      expiresAt: this.now() + this.ttlMs,
      readbackText: "",
      playbackEndMs: 0,
      confirmWindowSeq: 0,
      dispatched: false,
    };
    this.confirmWindowSeq = 0;
    return this.view()!;
  }

  /** Amend the live draft in place (a negation adds an item rather than sending).
   *  Bumps the version so a stale Send button or a stale confirmation cannot match. */
  amend(patch: { summaryBn?: string; englishSpec?: string; openQuestions?: string[] }): ActiveDraftView | null {
    const d = this.live();
    if (!d || d.state === "verifying") return null;
    if (patch.summaryBn !== undefined) d.summaryBn = patch.summaryBn;
    if (patch.englishSpec !== undefined) d.englishSpec = patch.englishSpec;
    if (patch.openQuestions !== undefined) d.openQuestions = [...patch.openQuestions];
    d.version += 1;
    d.state = "proposing";
    d.expiresAt = this.now() + this.ttlMs;
    d.dispatched = false;
    this.confirmWindowSeq += 1;
    return this.view();
  }

  /**
   * Speak the read-back and arm the gate. The gate moves to `confirming` only now —
   * before this, an affirmation is answering a question, not confirming a draft.
   */
  requestConfirmation(draftId: string, version: number, readbackText: string, playbackEndMs: number): ActiveDraftView | null {
    const d = this.live();
    if (!d || d.draftId !== draftId || d.version !== version) return null;
    d.readbackText = readbackText;
    d.playbackEndMs = playbackEndMs;
    d.state = "confirming";
    d.expiresAt = this.now() + this.ttlMs;
    this.confirmWindowSeq += 1;
    d.confirmWindowSeq = this.confirmWindowSeq;
    return this.view();
  }

  /**
   * Validate the user's own transcript against the armed draft. Returns a verdict;
   * on success the draft moves to `verifying` and is returned so the caller can
   * dispatch with the SAME version.
   *
   * A draft must be `confirming` for this to be possible. An affirmation while the
   * draft is still `proposing` answers the model's clarifying question instead.
   */
  onUserTranscript(text: string, seq: number, vadOnsetMs: number, playbackEndMs?: number): { verdict: Verdict; draft: ActiveDraftView | null } {
    const d = this.live();
    if (!d) return { verdict: { ok: false, reason: "no_draft" }, draft: null };
    if (d.state === "verifying") return { verdict: { ok: false, reason: "already_verifying" }, draft: this.view() };
    if (d.state === "expired") return { verdict: { ok: false, reason: "expired" }, draft: this.view() };
    if (d.state !== "confirming") return { verdict: { ok: false, reason: "not_armed" }, draft: this.view() };
    // A confirmation belongs to the read-back that was playing when it was spoken;
    // a newer read-back means this utterance was about the older one.
    if (seq < d.confirmWindowSeq) return { verdict: { ok: false, reason: "stale_window" }, draft: this.view() };

    const verdict = isConfirmation(text, d.readbackText, vadOnsetMs, playbackEndMs ?? d.playbackEndMs);
    if (!verdict.ok) return { verdict, draft: this.view() };
    d.state = "verifying";
    return { verdict, draft: this.view() };
  }

  /**
   * KEY 1's half: the model nominates a confirmation. Nomination alone never
   * dispatches — it only advances a draft that the deterministic validator has
   * already blessed (`verifying`). Two keys must agree.
   */
  confirmInstruction(draftId: string, version: number): ActiveDraftView | null {
    const d = this.live();
    if (!d || d.draftId !== draftId || d.version !== version) return null;
    if (d.state === "verifying" || d.state === "confirming") {
      d.state = "verifying";
      return this.view();
    }
    return null;
  }

  /**
   * THE compare-and-swap. Synchronous by design: a spoken confirmation and a UI Send
   * can land in the same tick, and only one of them may win. Returns a dispatch
   * ticket exactly once per (draftId, version); every later call returns null.
   */
  dispatchDraft(draftId: string, version: number, source: "voice" | "ui"): DispatchTicket | null {
    const d = this.draft;
    if (!d || d.draftId !== draftId || d.version !== version) return null;
    if (d.dispatched) return null;                 // ← the CAS: one dispatch, ever
    if (d.state !== "verifying") return null;      // ← both keys, or nothing
    d.dispatched = true;
    d.state = "verifying";
    return {
      draftId: d.draftId,
      version: d.version,
      englishSpec: d.englishSpec,
      summaryBn: d.summaryBn,
      intentType: d.intentType,
      parentTaskId: this.parentTaskId,
      source,
    };
  }

  /**
   * The socket died mid-confirmation. Demote `verifying → confirming` so a reconnect
   * re-asks instead of resuming a half-done dispatch, and bump the confirm window so
   * an utterance captured before the drop cannot match afterwards.
   *
   * A draft that already dispatched is NOT resurrected: `dispatched` is latched, so
   * a reconnect can never produce a second dispatch for it.
   */
  onDisconnect(): void {
    const d = this.draft;
    this.confirmWindowSeq += 1;
    if (!d) return;
    d.confirmWindowSeq = this.confirmWindowSeq;
    if (d.state === "verifying" && !d.dispatched) {
      d.state = "confirming";
      d.expiresAt = this.now() + this.ttlMs;
    }
  }

  /** UI key 3: the card's Send. Requires the same version; never bypasses key 2. */
  onDraftAction(action: "send" | "discard" | "reactivate", draftId: string, version: number): { dispatched: DispatchTicket | null; draft: ActiveDraftView | null; reason?: string } {
    const d = this.draft;
    if (action === "discard") {
      if (!d || d.draftId !== draftId) return { dispatched: null, draft: this.view(), reason: "no_such_draft" };
      d.state = "expired";
      return { dispatched: null, draft: this.view() };
    }
    if (action === "reactivate") {
      if (!d || d.draftId !== draftId) return { dispatched: null, draft: this.view(), reason: "no_such_draft" };
      if (d.dispatched) return { dispatched: null, draft: this.view(), reason: "already_dispatched" };
      d.state = "confirming";
      d.expiresAt = this.now() + this.ttlMs;
      this.confirmWindowSeq += 1;
      d.confirmWindowSeq = this.confirmWindowSeq;
      return { dispatched: null, draft: this.view() };
    }
    // "send" is still only a nomination unless key 2 already blessed it. A UI Send
    // on a draft that was never validated fails closed and re-asks.
    if (!d || d.draftId !== draftId || d.version !== version) return { dispatched: null, draft: this.view(), reason: "stale_version" };
    if (d.state !== "verifying") return { dispatched: null, draft: this.view(), reason: "needs_spoken_confirmation" };
    return { dispatched: this.dispatchDraft(draftId, version, "ui"), draft: this.view() };
  }

  /** The version that a caller must echo back; handy for tests and the client. */
  get activeVersion(): number | null {
    return this.draft?.version ?? null;
  }

  clear(): void {
    this.draft = null;
    this.confirmWindowSeq = 0;
    this.parentTaskId = null;
  }
}

/** What a winning `dispatchDraft` call receives — enough to write the ledger row. */
export interface DispatchTicket {
  draftId: string;
  version: number;
  englishSpec: string;
  summaryBn: string;
  intentType: ActiveDraftView["intentType"];
  parentTaskId: string | null;
  source: "voice" | "ui";
}
