// Wire + state contracts shared by the agent (/realtime) and the renderer (RealtimeEngine).

export type SteeringCapability =
  | "meta_steering"      // the agent advertises _session/steering
  | "concurrent_prompt"  // a second session/prompt mid-run is accepted and steers
  | "mcp_guidance"       // the agent reliably polls a client-hosted MCP tool
  | "cancel_resume";     // fallback: session/cancel + continuation prompt

export type VoiceEngineMode = "cascade" | "realtime";

export type RealtimePhase =
  | "idle"
  | "connecting"
  | "live"
  | "sleeping"
  | "reconnecting"
  | "error";

export type DraftState = "proposing" | "confirming" | "verifying" | "expired";

export interface ActiveDraftView {
  draftId: string;
  version: number;
  state: DraftState;
  intentType: "new_task" | "steer_add" | "steer_pivot";
  summaryBn: string;      // what is spoken back, in Bengali
  englishSpec: string;    // exactly what the coding agent receives
  openQuestions: string[];
  expiresAt: number;
}

export interface GuidanceItem {
  guidanceId: string;
  seq: number;
  taskId: string;
  idempotencyKey: string;
  kind: "steer_add" | "steer_pivot" | "answer_clarify";
  text: string;           // English imperative, the only text the agent receives
  sourceSummaryBn?: string;
  status: "pending" | "delivered_native" | "delivered_mcp" | "applied_after_cancel" | "superseded";
  createdAt: number;
  deliveredAt?: number;
}

export interface TaskDigestView {
  taskId: string;
  agentId: string;
  status: "running" | "needs_input" | "done" | "failed" | "cancelled";
  currentStep: string;
  todos: Array<{ text: string; status: string }>;
  lastTools: string[];
  startedAt: number;
  voiceSummary?: string;
}

// WS URL:
//   ws://127.0.0.1:<port>/realtime?chat=<chatId>&lang=<lang>&device=<0|1>&token=<secret>
// Binary frames:
//   client -> server : raw PCM16LE mono 16000 Hz, 60–100 ms per frame, NO header
//   server -> client : [4-byte uint32LE generation][raw PCM16LE mono 24000 Hz]

export type RealtimeClientMsg =
  | { t: "barge_in"; epoch: number; vadOnsetMs: number }
  | { t: "mute"; muted: boolean }
  | { t: "vad_state"; speaking: boolean; onsetMs: number; playbackEndMs?: number; echoSafe: boolean }
  | { t: "draft_action"; action: "send" | "discard" | "reactivate"; draftId: string; version: number }
  | { t: "bind"; agentId: string | null; cwd?: string; resumeSessionId?: string }
  | { t: "permission_response"; id: string; optionId: string }
  | { t: "elicitation_response"; id: string; action: "accept" | "decline" | "cancel"; content?: Record<string, unknown> }
  | { t: "control"; action: "end" | "sleep" | "wake" };

export type RealtimeServerMsg =
  | { t: "phase"; phase: RealtimePhase; epoch: number; reason?: string }
  | { t: "transcript"; role: "user" | "assistant"; text: string; final: boolean; seq: number }
  | { t: "draft_state"; draft: ActiveDraftView | null }
  | { t: "task_status"; task: TaskDigestView | null }
  | { t: "barge_in_ack"; epoch: number }
  | { t: "sse"; event: Record<string, unknown> }
  | { t: "bound_state"; agentId: string | null; cwd?: string; sessionId?: string }
  | { t: "agent_meta"; [k: string]: unknown }
  | { t: "permission"; id: string; question: string; options: Array<{ id: string; label: string }> }
  | { t: "elicitation"; [k: string]: unknown }
  | { t: "error"; code: string; message: string };

export const GEMINI_TOOL_NAMES = [
  "propose_instruction",
  "request_confirmation",
  "confirm_instruction",
  "get_status",
  "cancel_task",
  "answer_agent",
] as const;
