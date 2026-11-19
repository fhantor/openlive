// The renderer's side of the engine seam: what a voice engine reports, what the
// surface may command it to do, and the phase a call is in. The cascade engine
// (voiceEngine.ts) implements this today; the native Gemini engine implements the
// same contract, so useLiveSession/useFlowOwner never learn which one they hold.

/** Where the turn loop is. `idle` is also "no reply and no user speech". */
export type EnginePhase = "idle" | "listening" | "thinking" | "speaking";

export interface VoiceEngineHandlers {
  onPhase: (p: EnginePhase) => void;
  onPartial: (text: string) => void;      // interim user caption (greyed)
  onUserText: (text: string, wordsAtMs: number[], speaker?: string, judged?: string, aside?: boolean) => void; // final user turn → send to server: each captionWords word's onset, ms into the turn's audio (its segments back to back); `speaker` "you" or "other N" when the voiceprint is on; `judged` its side talk judgment's id in the agent's log, when that is on; `aside` it was taken for side talk and sent on by hand

  onAgentText: (sentence: string, wordsAtMs: number[]) => void; // agent caption chunk, now voicing: each captionWords word's onset from now
  onAgentTiming?: (wordsAtMs: number[]) => void; // the chunk on screen, timed again once more of its audio is in
  onBargeIn: (spoken?: string) => void;     // cancel the LLM stream; `spoken` = what was actually voiced so far, undefined before this reply arrived
  // A mid-thought pause is being held: `until` = when it auto-sends (UI shows a
  // "waiting for you… tap to send" affordance); null = hold resolved/cancelled.
  onHold: (h: { until: number } | null) => void;
  /** True while user speech must NOT barge in — e.g. a permission ask is pending
   *  and the next utterance IS the answer. Cancelling there killed the very ask
   *  the user was answering (the chip vanished the moment they spoke). */
  holdBargeIn?: () => boolean;
  /** True when `text` settles the pending ask ("yes", "no"): its turn ends at
   *  once, since the turn model calls a bare "Yes." unfinished and would hold it. */
  answersAsk?: (text: string) => boolean;
  /** The mic track ended on its own: unplugged, or its permission revoked. The
   *  surface re-acquires a device and hands it to setStream. */
  onMicLost?: () => void;
  /** A finished sentence the side talk check took for talk to someone else:
   *  dropped, not sent. The surface can show it, and send it on a tap
   *  (sendAside). `judged`: as onUserText's. */
  onSideTalk?: (text: string, speaker?: string, judged?: string) => void;
  /** The user started talking: a segment that is not the agent's own echo. */
  onSpeechStart?: () => void;
  /** A push-to-talk hold's first frame with sound in it: the device is awake, words from here are heard. */
  onHoldLive?: () => void;
  /** A hold's microphone opened and still gave no sound once its device's wait was up (holdVerdict). */
  onHoldSilent?: () => void;
}

/** How a push-to-talk hold ended: its words sent on, none heard, or heard and not written down. */
export type HoldEnd = "heard" | "silent" | "lost";

/**
 * Turn-taking numbers for a surface that needs different ones from the user's
 * saved pipeline.
 *
 * Flow is the case this exists for: in a call the person can see the caption
 * and press a key, and being cut off early costs them a click. Hands-free there
 * is no key and no screen, so a sentence cut in half is sent, answered and
 * acted on before they can say the rest of it.
 */
export interface TurnTuning { threshold?: number; holdMs?: number; redemptionMs?: number; backchannels?: boolean }

/** Everything the surface may ask of a live engine, whichever one it holds. */
export interface VoiceEngine {
  /** Opens the microphone through `open` each time the engine starts listening. */
  start(open: () => Promise<MediaStream>): Promise<void>;
  /** Swaps the mic mid-call (device change, or a lost one) without touching playback. */
  reopen(): Promise<void>;
  stop(): void;

  currentPhase(): EnginePhase;

  /** A turn the user sends by hand: cuts a reply under way and waits on the new one. False while the user talks. */
  openTurn(): boolean;
  /** "Not for you" on the reply under way: cut like a barge-in. */
  dropReply(): void;
  /** Public "send now": commit a held utterance without waiting for the hold timer. */
  commitPending(): void;
  /** A sentence the side talk check dropped, sent on the user's tap. */
  sendAside(text: string, speaker?: string, judged?: string): boolean;

  beginPtt(): void;
  endPtt(now?: boolean, lateMs?: number): Promise<HoldEnd>;
  dropPtt(): void;
  pttActive(): boolean;
  setGate(on: boolean): void;
  setHoldCues(on: boolean): void;
  setMuted(muted: boolean): void;

  feedAgentDelta(text: string): void;
  endAgentStep(): void;
  endAgentTurn(): void;
  ask(text: string): void;
  /** A short out-of-band line through the same TTS chain. */
  say(text: string): void;
  /** A line from outside the conversation, said in the first gap. */
  announce(text: string): void;

  /** Silence the reply and drop the rest of it, returning what was actually voiced. */
  cutReply(): string | undefined;
  /** What was heard up to now goes nowhere. */
  discard(): void;

  micLevel(): number;
  agentLevel(): number;
  setSpeaker(id: string): void;
  micBands(n?: number): number[];
  agentBands(n?: number): number[];
}
