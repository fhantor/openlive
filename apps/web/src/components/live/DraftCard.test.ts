// Guards key 3 of the confirmation gate (docs/04-REVIEWER-NOTES.md §2) and the
// Stage 8 rollback in docs/00-BUILD-ORDER.md §C: the DraftCard's Send must carry
// the CURRENT version, must not exist for a draft the spoken key has not
// validated, and a stale version must not produce an action at all.
//
// The root vitest runner resolves no "@/" alias and has no DOM, so the card's kit
// is stubbed here and only its pure decision functions are exercised. They are the
// same ones the buttons read for `disabled` and build their wire message with, so
// this is the button behaviour, minus React.
import type { ActiveDraftView } from "@openlive/shared";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/ui", () => ({ Button: () => null, Chip: () => null }));

import {
  DraftCard,
  actionEnabled,
  actionMatchesDraft,
  draftActionMessage,
  sendHint,
} from "./DraftCard.tsx";

/** A draft as /realtime sends it (`draft_state`), with the fields a test cares about overridden. */
const view = (over: Partial<ActiveDraftView> = {}): ActiveDraftView => ({
  draftId: "draft-1",
  version: 1,
  state: "proposing",
  intentType: "new_task",
  summaryBn: "হেডারের টাইটেলটা 'OpenLive Pro' করো",
  englishSpec: "Change the header title to 'OpenLive Pro'; do not modify the CSS",
  openQuestions: [],
  expiresAt: Date.now() + 120_000,
  ...over,
});

describe("DraftCard, key 3 of the confirmation gate", () => {
  it("Send carries the CURRENT version, including after a steer bumped it", () => {
    const first = view({ state: "verifying", version: 3 });
    expect(draftActionMessage(first, "send")).toEqual({
      t: "draft_action", action: "send", draftId: "draft-1", version: 3,
    });
    // The negation trap amends the draft and bumps the version. A card drawn for v3
    // must never dispatch v3 once v4 is live; it carries the version it shows.
    const amended = view({ state: "verifying", version: 4 });
    expect(draftActionMessage(amended, "send")?.version).toBe(4);
    expect(actionMatchesDraft(amended, draftActionMessage(amended, "send")!)).toBe(true);
  });

  it("Send is closed unless the spoken key has made the draft `verifying`", () => {
    for (const state of ["proposing", "confirming", "expired"] as const) {
      const d = view({ state });
      expect(actionEnabled("send", d), state).toBe(false);
      expect(draftActionMessage(d, "send"), state).toBeNull();
      expect(sendHint(d), state).not.toBeNull(); // the card always says why Send is locked
    }
    const ready = view({ state: "verifying" });
    expect(actionEnabled("send", ready)).toBe(true);
    expect(draftActionMessage(ready, "send")?.action).toBe("send");
    expect(sendHint(ready)).toBeNull();
  });

  it("a stale version cannot produce an action", () => {
    const live = view({ state: "verifying", version: 5 });
    // The action a card drew while v4 was on screen is stale the moment v5 lands.
    const before = draftActionMessage(view({ state: "verifying", version: 4 }), "send");
    expect(before).not.toBeNull();
    expect(actionMatchesDraft(live, before!)).toBe(false);
    // Same version, same draft: it may go (the agent still CASes it).
    expect(actionMatchesDraft(live, { draftId: "draft-1", version: 5 })).toBe(true);
    // No draft, or another chat's draft, means no action either.
    expect(actionMatchesDraft(null, before!)).toBe(false);
    expect(actionMatchesDraft(live, { draftId: "draft-2", version: 5 })).toBe(false);
  });

  it("no draft on screen means no action and no card (cascade mode is unchanged)", () => {
    for (const action of ["send", "discard", "reactivate"] as const) {
      expect(actionEnabled(action, null), action).toBe(false);
      expect(draftActionMessage(null, action), action).toBeNull();
    }
    expect(DraftCard({ draft: null, onAction: () => {} })).toBeNull();
  });

  it("Discard closes a live draft; Re-activate exists for the expired one", () => {
    for (const state of ["proposing", "confirming", "verifying"] as const) {
      expect(actionEnabled("discard", view({ state })), state).toBe(true);
      expect(actionEnabled("reactivate", view({ state })), state).toBe(false);
    }
    expect(actionEnabled("discard", view({ state: "expired" }))).toBe(false);
    expect(actionEnabled("reactivate", view({ state: "expired" }))).toBe(true);
    expect(draftActionMessage(view({ state: "expired", version: 2 }), "reactivate")).toEqual({
      t: "draft_action", action: "reactivate", draftId: "draft-1", version: 2,
    });
  });
});
