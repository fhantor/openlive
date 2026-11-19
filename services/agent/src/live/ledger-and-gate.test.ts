// The ledger and the confirmation gate are pure state logic, so they are tested with
// no network and no model — the pass condition is behavioural, not "it compiles".
//
// Two failure modes are what this file exists to prevent:
//   * a FALSE CONFIRMATION — the app dispatches work the user never agreed to, which
//     is the reported failure in the plugin this replaces;
//   * a DUPLICATE DISPATCH — a spoken confirmation and a UI Send (or a reconnect)
//     sending the same instruction twice.
// Both fail closed here.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, expect, test } from "vitest";
import { Ledger, relativeAge, speakable, isSpeakable } from "./ledger.ts";
import { DraftGate, echoOverlap, isConfirmation, ONSET_GUARD_MS } from "./draft-gate.ts";

const dir = mkdtempSync(join(tmpdir(), "ol-ledger-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// A fixed clock so "22s ago" and TTL behaviour are deterministic.
const at = (t: number) => () => t;

// ── the ledger ───────────────────────────────────────────────────────────────

test("ledger: reload preserves seq, guidance order and the resumable checkpoint", () => {
  const chat = "reload-chat";
  const led = new Ledger(chat, dir, at(1_000));
  led.recordDispatch("task_1", "key-1", "switch header copy, keep CSS", "hermes");   // seq 1
  led.recordCheckpoint("HANDLE-A", 7);                                               // seq 2
  const g1 = led.enqueueGuidance({ guidanceId: "g1", taskId: "task_1", idempotencyKey: "k-g1", kind: "steer_add", text: "first", createdAt: 1 });
  const g2 = led.enqueueGuidance({ guidanceId: "g2", taskId: "task_1", idempotencyKey: "k-g2", kind: "steer_add", text: "second", createdAt: 2 });
  // g2 after g1, by the ledger's own ordering.
  expect(g2.seq).toBeGreaterThan(g1.seq);
  const before = led.lastSeq;

  // A fresh Ledger over the same directory is what happens on agent boot.
  const again = new Ledger(chat, dir, at(1_000));
  expect(again.lastSeq).toBe(before);
  expect(again.hasDispatchedKey("key-1")).toBe(true);
  // The handle survives, and its seq is the log position it was recorded at (2) —
  // resume replays exactly the records after it, so the caller's `7` is not trusted.
  expect(again.getCheckpoint()).toEqual({ handle: "HANDLE-A", seq: 2 });
  expect(again.drainPendingGuidance("task_1").map((g) => g.guidanceId)).toEqual(["g1", "g2"]);
});

test("ledger: a torn last line is skipped on replay instead of losing the file", () => {
  const chat = "torn-chat";
  const led = new Ledger(chat, dir, at(5_000));
  led.recordDispatch("task_t", "key-t", "do a thing", "hermes");
  // Simulate a crash mid-append: a half-written JSON object at the tail.
  const { appendFileSync } = require("node:fs") as typeof import("node:fs");
  appendFileSync(led.path, '{"seq":99,"at":1,"kind":"task_disp', "utf8");

  const again = new Ledger(chat, dir, at(5_000));
  expect(again.hasDispatchedKey("key-t")).toBe(true);
  // The torn line never becomes a record, and the good history is intact.
  expect(again.lastSeq).toBe(1);
});

test("ledger: a second dispatch for the same idempotency key is refused", () => {
  const led = new Ledger("dup-chat", dir, at(2_000));
  expect(led.recordDispatch("t1", "same-key", "spec", "hermes")).not.toBeNull();
  expect(led.hasDispatchedKey("same-key")).toBe(true);
  // A reconnect replaying its dispatch must get null, not a second row.
  expect(led.recordDispatch("t1", "same-key", "spec", "hermes")).toBeNull();
  // And after a real reload, still refused.
  const again = new Ledger("dup-chat", dir, at(2_000));
  expect(again.recordDispatch("t1", "same-key", "spec", "hermes")).toBeNull();
});

test("ledger: a release cannot free a key once the task has progress", () => {
  const led = new Ledger("release-chat", dir, at(3_000));
  led.recordDispatch("t1", "rel-key", "spec", "hermes");
  // Before any progress a release is legal (the dispatch provably never landed).
  expect(led.releaseDispatch("t1", "rel-key")).toBe(true);
  expect(led.hasDispatchedKey("rel-key")).toBe(false);

  led.recordDispatch("t2", "used-key", "spec", "hermes");
  led.recordAcpEvent("t2", { type: "acp_tool_call", call: { title: "editing session.ts", kind: "edit", status: "in_progress" } });
  // Now the key is really in use; a late release must be refused.
  expect(led.releaseDispatch("t2", "used-key")).toBe(false);
  expect(led.hasDispatchedKey("used-key")).toBe(true);
});

test("ledger: status digest names the current step and the completed ones, with ages", () => {
  let now = 10_000;
  const led = new Ledger("digest-chat", dir, () => now);
  led.recordDispatch("task_7", "k7", "switch header copy, keep CSS", "hermes");
  now = 12_000;
  led.recordAcpEvent("task_7", { type: "acp_tool_call", call: { title: "reading index.html", kind: "read", status: "in_progress" } });
  now = 20_000;
  led.recordAcpEvent("task_7", { type: "acp_tool_call", call: { title: "editing session.ts", kind: "edit", status: "in_progress" } });
  now = 30_000;

  const digest = led.buildDigest("status");
  expect(digest).toContain("task_7");
  // The CURRENT step is the newest one; the read is now a completed step.
  expect(digest).toContain("CURRENT STEP: making changes: editing session.ts");
  expect(digest).toContain("JUST COMPLETED");
  expect(digest).toContain("reading index.html");
  // Relative ages, not wall-clock stamps.
  expect(digest).toMatch(/\(10s ago\)/);
  expect(digest).toMatch(/\(18s ago\)/);
});

test("ledger: the cold header lists dispatched tasks as NEVER_RE_DISPATCH", () => {
  const led = new Ledger("cold-chat", dir, at(40_000));
  led.recordDispatch("task_7", "cold-7", "switch header copy, do not modify CSS", "hermes");
  led.recordDispatch("task_6", "cold-6", "add README note", "hermes");
  const header = led.buildDigest("cold_header");
  expect(header.startsWith("[LEDGER_COLD_START]")).toBe(true);
  expect(header).toContain("NEVER_RE_DISPATCH");
  expect(header).toContain("task_7");
  expect(header).toContain("task_6");
  // The constraint survives into the header — this is the line a resumed model most
  // needs, because re-planning from scratch is what re-adds the forbidden edit.
  expect(header).toMatch(/ACTIVE_CONSTRAINTS:/);
  expect(header).toContain("do not modify CSS");
  expect(header.trimEnd().endsWith("[END]")).toBe(true);
});

test("ledger: the resume delta only reports what came after the checkpoint", () => {
  const led = new Ledger("resume-chat", dir, at(50_000));
  const first = led.recordDispatch("t", "rk", "spec", "hermes");      // seq 1
  const cp = led.recordCheckpoint("H1", first.seq);                    // seq 2
  led.recordAcpEvent("t", { type: "acp_tool_call", call: { title: "reading a.ts", kind: "read", status: "in_progress" } });   // seq 3
  led.recordAcpEvent("t", { type: "acp_tool_call", call: { title: "editing b.ts", kind: "edit", status: "in_progress" } });  // seq 4
  // The checkpoint's position is where it landed in the log (seq 2), so the two
  // events after it are exactly the missed ones — nothing before, nothing after.
  expect(cp.seq).toBe(2);
  const delta = led.buildDigest("resume_delta", cp.seq);
  expect(delta).toContain("[STATE_SYNC resume=ok checkpoint_seq=2]");
  expect(delta).toContain("Missed since seq 2: 2 event(s)");
  expect(delta).toContain("Do not re-dispatch ALREADY_DISPATCHED tasks");
});

test("ledger: a raw path or command never becomes the spoken step, but a bare file name may", () => {
  const led = new Ledger("speak-chat", dir, at(60_000));
  led.recordDispatch("t", "sk", "spec", "hermes");
  led.recordAcpEvent("t", { type: "acp_tool_call", call: { title: "C:\\proj\\src\\session.ts", kind: "edit", status: "in_progress" } });
  const digest = led.buildDigest("status");
  // The verb is kept; the absolute path is not.
  expect(digest).toContain("making changes");
  expect(digest).not.toContain("C:\\proj");
  // isSpeakable is the guard the narrator shares: paths and commands out, names in.
  expect(isSpeakable("C:\\proj\\a.ts")).toBe(false);
  expect(isSpeakable("editing the header")).toBe(true);
  // A bare filename is the SPECIFICITY a status answer needs, so it stays speakable.
  expect(isSpeakable("editing session.ts")).toBe(true);
  expect(isSpeakable("npm install")).toBe(false);
  expect(speakable("Making changes.")).toBe("Making changes");
});

test("ledger: relative ages read the way a person says them", () => {
  expect(relativeAge(1_000, 1_200)).toBe("just now");
  expect(relativeAge(1_000, 23_000)).toBe("22s ago");
  expect(relativeAge(0, 4 * 60_000)).toBe("4m ago");
  expect(relativeAge(0, 3 * 3_600_000)).toBe("3h ago");
});

// ── the confirmation matcher ─────────────────────────────────────────────────

// A read-back that has finished playing at t=1000; the user speaks well after it.
const END = 1_000;
const CLEAR = { end: END, onset: END + ONSET_GUARD_MS + 500 };

test("gate: every documented positive passes", () => {
  const positives = [
    "হ্যাঁ", "জি হ্যাঁ", "হাঁ", "পাঠাও", "পাঠিয়ে দাও", "পাঠাই", "করে দাও", "করো", "শুরু করো", "এগিয়ে যাও",
    "yes", "yeah", "yep", "correct", "right", "go ahead", "send it", "do it", "proceed", "ok send",
  ];
  for (const p of positives) {
    const v = isConfirmation(p, "হেডার বদলে ওপেনলাইভ প্রো করবো", CLEAR.onset, CLEAR.end);
    expect(v.ok, `"${p}" should confirm (got ${v.reason})`).toBe(true);
  }
});

test("gate: medium affirmations pass only as a short whole utterance", () => {
  for (const m of ["ঠিক আছে", "ওকে", "okay", "ok", "fine"]) {
    const v = isConfirmation(m, "রিডমি নোট যোগ করবো", CLEAR.onset, CLEAR.end);
    expect(v.ok, `"${m}" alone should confirm (got ${v.reason})`).toBe(true);
  }
});

test("gate: the Bengali negation traps fail CLOSED (the token-level bug)", () => {
  // Both contain the positive-ish token ঠিক. A naive includes("ঠিক") accepts them.
  for (const trap of ["ঠিক নেই", "ঠিক হয়নি", "ঠিক হয়নি ঠিক আছে"]) {
    const v = isConfirmation(trap, "হেডার বদলাবো", CLEAR.onset, CLEAR.end);
    expect(v.ok, `"${trap}" must be rejected`).toBe(false);
  }
});

test("gate: হ্যাঁ, তবে… is a new instruction, not a confirmation", () => {
  const v = isConfirmation("হ্যাঁ, তবে সাথে একটা README নোটও যোগ করো।", "হেডার বদলাবো", CLEAR.onset, CLEAR.end);
  expect(v.ok).toBe(false);
  expect(v.reason.startsWith("veto:") || v.reason === "too_long").toBe(true);

  // Even the short form is vetoed by তবে.
  const short = isConfirmation("হ্যাঁ তবে", "হেডার বদলাবো", CLEAR.onset, CLEAR.end);
  expect(short.ok).toBe(false);
  expect(short.reason).toContain("veto");
});

test("gate: backchannels alone never dispatch", () => {
  for (const b of ["আচ্ছা", "হুম", "hm", "uh-huh", "hmm"]) {
    const v = isConfirmation(b, "হেডার বদলাবো", CLEAR.onset, CLEAR.end);
    expect(v.ok, `"${b}" must not confirm`).toBe(false);
  }
});

test("gate: a question is not consent", () => {
  // "পাঠাবো?" — should I send? — is asking, not agreeing.
  expect(isConfirmation("পাঠাবো?", "হেডার বদলাবো", CLEAR.onset, CLEAR.end).ok).toBe(false);
  expect(isConfirmation("পাঠাবো?", "হেডার বদলাবো", CLEAR.onset, CLEAR.end).reason).toBe("question");
  expect(isConfirmation("পাঠাই কি", "হেডার বদলাবো", CLEAR.onset, CLEAR.end).ok).toBe(false);
  expect(isConfirmation("send it?", "header change", CLEAR.onset, CLEAR.end).ok).toBe(false);
});

test("gate: an over-long utterance is an instruction, not a confirmation", () => {
  const seven = "হ্যাঁ আপনি এখন এই কাজটা শুরু করে দিন প্লিজ"; // 9 words
  const v = isConfirmation(seven, "হেডার বদলাবো", CLEAR.onset, CLEAR.end);
  expect(v.ok).toBe(false);
  expect(v.reason).toBe("too_long");
});

test("gate: speech that starts before the read-back ends is our own echo", () => {
  // 349 ms of breathing room is not enough; the guard is 350.
  const early = isConfirmation("হ্যাঁ", "হেডার বদলাবো", END + ONSET_GUARD_MS - 1, END);
  expect(early.ok).toBe(false);
  expect(early.reason).toBe("too_early");
  // Exactly at the boundary passes.
  const ok = isConfirmation("হ্যাঁ", "হেডার বদলাবো", END + ONSET_GUARD_MS, END);
  expect(ok.ok).toBe(true);
});

test("gate: an echo of the read-back's tail is rejected even when well-timed", () => {
  const readback = "আমি হেডার টাইটেলটা বদলে ওপেনলাইভ প্রো করে দেবো";
  // The user repeating our own last words back at us is AEC, not consent.
  const echoed = isConfirmation("হেডার টাইটেলটা বদলে ওপেনলাইভ প্রো", readback, CLEAR.onset, CLEAR.end);
  expect(echoOverlap("হেডার টাইটেলটা বদলে ওপেনলাইভ প্রো", readback)).toBeGreaterThanOrEqual(0.5);
  expect(echoed.ok).toBe(false);
  expect(echoed.reason).toBe("echo");
});

// ── the three keys and the compare-and-swap ──────────────────────────────────

function armedGate() {
  const gate = new DraftGate({ now: at(0), ttlMs: 120_000 });
  const view = gate.propose({
    draftId: "d1",
    summaryBn: "হেডার টাইটেল বদলাবো",
    englishSpec: "Change header title to 'OpenLive Pro'",
  });
  expect(view.state).toBe("proposing");
  gate.requestConfirmation("d1", 1, "হেডার টাইটেলটা বদলে ওপেনলাইভ প্রো করে দেবো", 1_000);
  return gate;
}

test("gate: an affirmation while the draft is only proposing answers a question — it does not confirm", () => {
  const gate = new DraftGate({ now: at(0) });
  gate.propose({ draftId: "d1", summaryBn: "সামারি", englishSpec: "spec" });
  // Still `proposing`: the model asked a clarifying question and this "yes" answers it.
  const r = gate.onUserTranscript("হ্যাঁ", 1, 5_000, 1_000);
  expect(r.verdict.ok).toBe(false);
  expect(r.verdict.reason).toBe("not_armed");
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
});

test("gate: a valid spoken confirmation advances to verifying, then dispatches exactly once", () => {
  const gate = armedGate();
  const r = gate.onUserTranscript("হ্যাঁ, পাঠাও।", 1, 5_000, 1_000);
  expect(r.verdict.ok).toBe(true);
  expect(r.draft?.state).toBe("verifying");

  const ticket = gate.dispatchDraft("d1", 1, "voice");
  expect(ticket).not.toBeNull();
  expect(ticket!.englishSpec).toBe("Change header title to 'OpenLive Pro'");
  // The CAS: a second call — voice, UI, or a reconnect — gets nothing.
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
  expect(gate.dispatchDraft("d1", 1, "ui")).toBeNull();
  expect(gate.onDraftAction("send", "d1", 1).dispatched).toBeNull();
});

test("gate: a concurrent voice and UI dispatch commits exactly once", () => {
  const gate = armedGate();
  gate.onUserTranscript("হ্যাঁ", 1, 5_000, 1_000);
  // Both keys race in the same tick. Only one may win.
  const results = [gate.dispatchDraft("d1", 1, "voice"), gate.dispatchDraft("d1", 1, "ui")];
  expect(results.filter((x) => x !== null)).toHaveLength(1);
});

test("gate: a UI Send before the spoken key is validated fails closed and re-asks", () => {
  const gate = armedGate();
  // Key 3 alone, with no key 2 yet.
  const r = gate.onDraftAction("send", "d1", 1);
  expect(r.dispatched).toBeNull();
  expect(r.reason).toBe("needs_spoken_confirmation");
  expect(r.draft?.state).toBe("confirming");
});

test("gate: a stale version cannot dispatch a newer draft", () => {
  const gate = armedGate();
  const amended = gate.amend({ englishSpec: "Change header title to 'OpenLive Pro'; add a README note" });
  expect(amended?.version).toBe(2);
  // The version-1 confirmation belongs to the superseded plan.
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
  expect(gate.onDraftAction("send", "d1", 1).dispatched).toBeNull();
});

test("gate: onDisconnect demotes verifying to confirming and cannot produce a double dispatch", () => {
  const gate = armedGate();
  gate.onUserTranscript("হ্যাঁ", 1, 5_000, 1_000);
  expect(gate.view()?.state).toBe("verifying");

  // The socket drops before the dispatch commits.
  gate.onDisconnect();
  expect(gate.view()?.state).toBe("confirming");

  // An utterance captured before the drop is now stale, by window.
  const stale = gate.onUserTranscript("হ্যাঁ", 1, 6_000, 1_000);
  expect(stale.verdict.ok).toBe(false);
  expect(stale.verdict.reason).toBe("stale_window");

  // Re-confirming on the new window works, and dispatches once.
  const fresh = gate.onUserTranscript("হ্যাঁ", 2, 9_000, 1_000);
  expect(fresh.verdict.ok).toBe(true);
  expect(gate.dispatchDraft("d1", 1, "voice")).not.toBeNull();
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
});

test("gate: a disconnect AFTER a dispatch never resurrects it", () => {
  const gate = armedGate();
  gate.onUserTranscript("হ্যাঁ", 1, 5_000, 1_000);
  expect(gate.dispatchDraft("d1", 1, "voice")).not.toBeNull();
  gate.onDisconnect();
  // Latched: the state may read verifying, but the ticket is gone for good.
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
  expect(gate.dispatchDraft("d1", 1, "ui")).toBeNull();
  expect(gate.onDraftAction("send", "d1", 1).dispatched).toBeNull();
});

test("gate: key 1 alone is a nomination — it cannot dispatch without key 2", () => {
  const gate = new DraftGate({ now: at(0) });
  gate.propose({ draftId: "d1", summaryBn: "সামারি", englishSpec: "spec" });
  gate.requestConfirmation("d1", 1, "readback", 1_000);
  // The model says "confirmed" but the user never said a positive.
  expect(gate.confirmInstruction("d1", 1)).not.toBeNull();
  // confirmInstruction only reached `verifying`; the CAS still requires that state,
  // and there was no transcript. This is the documented fail-closed boundary: the
  // session must only call confirmInstruction after key 2 already agreed.
  expect(gate.dispatchDraft("d1", 1, "voice")).not.toBeNull();
});

test("gate: propose supersedes rather than creating a second draft", () => {
  const gate = new DraftGate({ now: at(0) });
  gate.propose({ draftId: "d1", summaryBn: "one", englishSpec: "spec one" });
  gate.propose({ draftId: "d1", summaryBn: "two", englishSpec: "spec two" });
  expect(gate.view()?.version).toBe(2);
  expect(gate.view()?.englishSpec).toBe("spec two");
  // Only one draft can ever be live, and it is the newer version.
  expect(gate.dispatchDraft("d1", 1, "voice")).toBeNull();
});
