// Ordering regression: LiveSession's tool providers read `host.cwd`, so the
// AcpHost has to be assigned before `callTools()` runs.
//
// This was a live crash, not a cosmetic one. Stage 4 moved the session's cwd
// onto the new AcpHost (`workspace: () => this.host.cwd`) but left the tools
// call above that assignment, so the FIRST websocket upgrade on /live or
// /realtime threw `TypeError: Cannot read properties of undefined (reading
// 'cwd')` and the agent process died on an uncaughtException.
//
// The reason 2,702 tests missed it: `skillTools()` — the one provider that
// dereferences `workspace()` — is registered in `server.ts`, not in the shared
// registry, so a plain `new LiveSession(...)` never walks that path. The
// registration below is what makes the ordering observable; without it this
// file would pass on the broken code too.
import { EventEmitter } from "node:events";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, expect, test } from "vitest";

const dir = mkdtempSync(join(tmpdir(), "ol-tools-order-"));
process.env.OPENLIVE_HOME = dir;
const { LiveSession } = await import("./session.ts");
const { registry } = await import("../capabilities/registry.ts");
const { skillTools } = await import("../skills/tools.ts");

// What the agent service does at boot. A workspace with a real skill in it, so
// the provider actually resolves a catalog rather than short-circuiting.
const project = join(dir, "project");
mkdirSync(join(project, ".openlive", "skills", "probe"), { recursive: true });
writeFileSync(join(project, ".openlive", "skills", "probe", "SKILL.md"), "---\nname: probe\ndescription: probe skill\n---\nbody\n");
registry.register(skillTools());

afterAll(() => {
  delete process.env.OPENLIVE_HOME;
  rmSync(dir, { recursive: true, force: true });
});

const socket = () => {
  const ws = Object.assign(new EventEmitter(), { OPEN: 1, readyState: 1, sent: [] as Record<string, any>[] });
  (ws as any).send = (raw: string) => ws.sent.push(JSON.parse(raw));
  return ws;
};

test("constructs with the skill provider registered, host assigned before tools", () => {
  const session = new LiveSession(socket() as never, "tools-order");
  // The invariant the crash broke: both exist, and the tools that read the cwd
  // were built against a live host.
  expect((session as any).host).toBeTruthy();
  expect((session as any).tools).toBeTruthy();
  expect((session as any).runner).toBeTruthy();
  expect(typeof (session as any).host.cwd).toBe("string");
});

test("the skill provider's workspace thunk resolves through the host", () => {
  const session = new LiveSession(socket() as never, "tools-order-cwd");
  const workspace = (session as any).toolSession.workspace;
  expect(workspace()).toBe((session as any).host.cwd);
});
