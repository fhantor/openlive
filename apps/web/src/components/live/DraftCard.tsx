"use client";

import { RotateCcw, Send, Trash2 } from "lucide-react";
import type { ActiveDraftView, DraftState, RealtimeClientMsg } from "@openlive/shared";
import { Button, Chip, type DotTone } from "@/components/ui";

// The draft the coding agent is waiting on: the Bengali summary that was read
// back, the English spec the agent will actually receive, the version and state,
// and the three things the person may do with it.
//
// Send is key 3 of the confirmation gate (docs/04-REVIEWER-NOTES.md §2). It is
// offered only once the deterministic spoken key (key 2) has advanced this draft
// to `verifying`, and the action carries the version it was drawn from — the
// agent compares that version again in a synchronous compare-and-swap, so a stale
// card can never turn into a dispatch. A draft the spoken key has not validated
// stays unsendable here, by construction: `draftActionMessage` returns null.
//
// With no draft armed it renders nothing at all, so a cascade call looks exactly
// as it did before this card existed.

/** One of the three card actions, spelled as the wire spells it. */
export type DraftAction = Extract<RealtimeClientMsg, { t: "draft_action" }>["action"];
/** The wire message a card action sends (`packages/shared/src/realtime-protocol.ts`). */
export type DraftActionMessage = Extract<RealtimeClientMsg, { t: "draft_action" }>;

const ACTIONS: readonly DraftAction[] = ["send", "discard", "reactivate"];

const LABEL: Record<DraftAction, string> = { send: "Send", discard: "Discard", reactivate: "Re-activate" };
const ARIA: Record<DraftAction, string> = { send: "Send the draft", discard: "Discard the draft", reactivate: "Re-activate the draft" };

const STATE: Record<DraftState, { label: string; dot: DotTone }> = {
  proposing: { label: "Proposing", dot: "muted" },
  confirming: { label: "Confirming", dot: "arc" },
  verifying: { label: "Verifying", dot: "success" },
  expired: { label: "Expired", dot: "danger" },
};

/**
 * Whether the card may offer `action` for `draft`. Send is key 3: it opens only
 * after the spoken key has validated this draft (state `verifying`), never on the
 * model's nomination alone. Discard closes any draft that is still live, and
 * Re-activate exists for one that expired. Pure.
 */
export function actionEnabled(action: DraftAction, draft: ActiveDraftView | null): boolean {
  if (!draft) return false;
  if (action === "send") return draft.state === "verifying";
  if (action === "reactivate") return draft.state === "expired";
  return draft.state !== "expired";
}

/** The card's own words for why Send is closed, or null when it is open. Pure. */
export function sendHint(draft: ActiveDraftView | null): string | null {
  if (!draft || draft.state === "verifying") return null;
  if (draft.state === "proposing") return "Say it, then confirm out loud — Send opens once the spoken read-back agrees.";
  if (draft.state === "confirming") return "The read-back is being spoken. Say yes to confirm — Send opens then.";
  return "This draft expired. Re-activate it to confirm it again.";
}

/**
 * The wire message for `action`, or null when the card must not send it at all.
 * The version is read from the view on screen right now, never from a remembered
 * one, so a card drawn for an older version can only ever echo that old version
 * (and the agent's gate refuses it). Pure.
 */
export function draftActionMessage(draft: ActiveDraftView | null, action: DraftAction): DraftActionMessage | null {
  if (!draft || !actionEnabled(action, draft)) return null;
  return { t: "draft_action", action, draftId: draft.draftId, version: draft.version };
}

/**
 * Whether `msg` still names the draft on screen. A click read before the wire
 * moved on (a steer amended the spec, which bumps the version) fails closed here
 * instead of dispatching the version it was drawn from. Pure.
 */
export function actionMatchesDraft(draft: ActiveDraftView | null, msg: { draftId: string; version: number }): boolean {
  return !!draft && msg.draftId === draft.draftId && msg.version === draft.version;
}

export function DraftCard({ draft, onAction }: {
  draft: ActiveDraftView | null;
  onAction: (msg: DraftActionMessage) => void;
}) {
  if (!draft) return null;
  const state = STATE[draft.state];
  const hint = sendHint(draft);
  const press = (action: DraftAction) => {
    const msg = draftActionMessage(draft, action);
    if (msg) onAction(msg);
  };
  return (
    <section aria-label="Draft instruction" aria-live="polite"
      className="fixed left-4 top-16 z-overlay flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2.5 rounded-xl border border-arc/35 bg-card p-3 shadow-card">
      <div className="flex min-w-0 items-center gap-2">
        <Chip dot={state.dot}>{state.label}</Chip>
        <span className="ml-auto shrink-0 text-micro tabular-nums text-faint">v{draft.version}</span>
      </div>

      {/* What was read back, in the person's own language. */}
      <p lang="bn" className="whitespace-pre-wrap break-words text-body leading-relaxed text-foreground">{draft.summaryBn}</p>

      {/* What the agent will actually receive — the only text it ever gets. */}
      <div className="flex flex-col gap-1">
        <span className="text-micro uppercase tracking-wide text-faint">The agent receives</span>
        <p className="whitespace-pre-wrap break-words rounded-lg bg-surface px-2.5 py-2 font-mono text-caption leading-relaxed text-muted-strong">{draft.englishSpec}</p>
      </div>

      {draft.openQuestions.length > 0 && (
        <ul className="flex list-disc flex-col gap-1 pl-4 text-caption text-muted-foreground">
          {draft.openQuestions.map((q) => <li key={q} className="break-words">{q}</li>)}
        </ul>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        {ACTIONS.map((action) => (
          <Button key={action} size="sm" variant={action === "send" ? "primary" : "secondary"}
            aria-label={ARIA[action]} disabled={!actionEnabled(action, draft)} onClick={() => press(action)}>
            {action === "send" ? <Send /> : action === "discard" ? <Trash2 /> : <RotateCcw />}
            {LABEL[action]}
          </Button>
        ))}
      </div>

      {hint && <p className="text-caption leading-relaxed text-muted-foreground">{hint}</p>}
    </section>
  );
}
