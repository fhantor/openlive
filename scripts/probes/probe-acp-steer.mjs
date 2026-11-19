// File: scripts/probes/probe-acp-steer.mjs
// Stage 0 probe: ACP has no standard steering, so measure what each installed
// agent actually does. The verdict is judged by DISK STATE, never by what the
// agent says it did.
//
//   node scripts/probes/probe-acp-steer.mjs                  # hermes acp
//   node scripts/probes/probe-acp-steer.mjs --all            # every STAGE0_AGENTS entry
//   node scripts/probes/probe-acp-steer.mjs --cmd hermes --args acp --keep
//
// Task: the agent is told to append to three files, one step each, 8 s apart:
//   step1_done.txt   created first
//   step2_ran.txt    created second
//   target_wrong.txt created third (the step the steer must prevent)
//
// At the first completed step four steering variants are tried, one per run:
//   (a) concurrent `session/prompt` while the task is running
//   (b) `_session/steering`, only if the agent advertises it
//   (c) `session/cancel` + a continuation prompt built from disk state
//   (d) an MCP `user_guidance` tool the agent is told to poll at every step
//
// Lossless steer means: target_wrong.txt was never created AND step1_done.txt is
// untouched (its bytes unchanged from the first step).

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

const SELF_PATH = fileURLToPath(import.meta.url);

// ── MCP child: one user_guidance tool over newline-delimited JSON-RPC ─────────
function runMcpGuidanceServer(guidanceFile) {
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  const send = (obj) => process.stdout.write(`${JSON.stringify(obj)}\n`);

  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let req;
    try {
      req = JSON.parse(trimmed);
    } catch {
      return;
    }
    if (req.id === undefined) return;

    if (req.method === "initialize") {
      send({
        jsonrpc: "2.0",
        id: req.id,
        result: {
          protocolVersion: "2024-11-05",
          capabilities: { tools: {} },
          serverInfo: { name: "openlive-probe-guidance", version: "0.1.0" },
        },
      });
      return;
    }
    if (req.method === "notifications/initialized" || req.method === "initialized") return;

    if (req.method === "tools/list") {
      send({
        jsonrpc: "2.0",
        id: req.id,
        result: {
          tools: [
            {
              name: "user_guidance",
              description:
                "Read any mid-task correction the user has just spoken. Call this before every step, and obey it instead of the original plan.",
              inputSchema: { type: "object", properties: {}, required: [] },
            },
          ],
        },
      });
      return;
    }

    if (req.method === "tools/call") {
      const name = req.params?.name;
      if (name !== "user_guidance") {
        send({ jsonrpc: "2.0", id: req.id, error: { code: -32601, message: `Unknown tool: ${name}` } });
        return;
      }
      let guidance = "";
      try {
        if (guidanceFile && fs.existsSync(guidanceFile)) {
          guidance = fs.readFileSync(guidanceFile, "utf8").trim();
          if (guidance) {
            fs.appendFileSync(`${guidanceFile}.ack`, `${Date.now()}\t${guidance}\n`);
            fs.writeFileSync(guidanceFile, "");
          }
        }
      } catch {
        guidance = "";
      }
      send({
        jsonrpc: "2.0",
        id: req.id,
        result: { content: [{ type: "text", text: guidance || "NO_NEW_GUIDANCE" }] },
      });
      return;
    }

    send({ jsonrpc: "2.0", id: req.id, error: { code: -32601, message: `Unsupported method: ${req.method}` } });
  });
}

// ── args ─────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const out = { cmd: "hermes", args: ["acp"], all: false, keep: false, timeoutMs: 150_000, stepGapMs: 8000 };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--cmd" && argv[i + 1]) out.cmd = argv[++i];
    else if (a === "--args" && argv[i + 1]) out.args = argv[++i].split(",").filter(Boolean);
    else if (a === "--all") out.all = true;
    else if (a === "--keep") out.keep = true;
    else if (a === "--timeout" && argv[i + 1]) out.timeoutMs = Number(argv[++i]);
    else if (a === "--gap" && argv[i + 1]) out.stepGapMs = Number(argv[++i]);
    else if (a === "--variant" && argv[i + 1]) out.variant = argv[++i];
  }
  return out;
}

// ── ACP client ──────────────────────────────────────────────────────────────
class AcpClient {
  constructor(cmd, args, cwd) {
    this.cwd = cwd;
    this.nextId = 1;
    this.pending = new Map();
    this.updates = [];
    this.stderrLines = [];
    // WHY: Windows .cmd/.bat shims (npx, hermes, opencode) need shell:true under Node 24 spawn.
    this.proc = spawn(cmd, args, { cwd, stdio: ["pipe", "pipe", "pipe"], shell: process.platform === "win32" });
    this.rl = readline.createInterface({ input: this.proc.stdout, crlfDelay: Infinity });
    this.rl.on("line", (line) => this.onLine(line));
    const errRl = readline.createInterface({ input: this.proc.stderr, crlfDelay: Infinity });
    errRl.on("line", (line) => {
      if (line.trim()) this.stderrLines.push(line);
      if (this.stderrLines.length > 400) this.stderrLines.shift();
    });
    this.proc.on("exit", (code, signal) => {
      this.exit = { code, signal };
      for (const { reject } of this.pending.values()) reject(new Error(`agent exited (${code}/${signal})`));
      this.pending.clear();
    });
  }

  onLine(line) {
    const text = line.trim();
    if (!text) return;
    let msg;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(`${msg.error.message ?? "ACP error"} (code ${msg.error.code})`));
      else resolve(msg.result);
      return;
    }
    if (msg.id !== undefined && msg.method) {
      if (msg.method === "session/request_permission") {
        const options = msg.params?.options ?? [];
        const allow = options.find((o) => /allow/.test(String(o.kind))) ?? options[0];
        this.respond(msg.id, allow ? { outcome: { outcome: "selected", optionId: allow.optionId } } : { outcome: { outcome: "cancelled" } });
      } else {
        this.respondError(msg.id, -32601, `unsupported client method ${msg.method}`);
      }
      return;
    }
    if (msg.method) this.updates.push(msg);
  }

  respond(id, result) {
    try { this.proc.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`); } catch { /* agent gone */ }
  }

  respondError(id, code, message) {
    try { this.proc.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, error: { code, message } })}\n`); } catch { /* agent gone */ }
  }

  request(method, params, timeoutMs = 60_000) {
    const id = this.nextId++;
    const payload = { jsonrpc: "2.0", id, method, params };
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} timeout after ${timeoutMs} ms`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      this.proc.stdin.write(`${JSON.stringify(payload)}\n`);
    });
  }

  notify(method, params) {
    this.proc.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, params })}\n`);
  }

  async dispose() {
    try { this.proc.stdin.end(); } catch { /* ignore */ }
    await new Promise((r) => setTimeout(r, 250));
    try { this.proc.kill(); } catch { /* ignore */ }
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const read = (f) => (fs.existsSync(f) ? fs.readFileSync(f, "utf8") : null);

/** Every file, its bytes and mtime right now — the only evidence that counts. */
function snapshot(dir) {
  const out = {};
  for (const name of ["step1_done.txt", "step2_ran.txt", "target_wrong.txt", "steer_note.txt"]) {
    const f = path.join(dir, name);
    if (fs.existsSync(f)) out[name] = { text: fs.readFileSync(f, "utf8"), mtime: fs.statSync(f).mtimeMs };
  }
  return out;
}

function agentText(prompt) {
  const step = (n, file) => `node -e "setTimeout(()=>require('fs').writeFileSync('${file}','STEP${n}'),8000)"`;
  const instructions =
    "Use your shell/terminal tool exactly once per step, in order. Each command below waits 8 seconds and then writes its file. " +
    "Run step 1, wait for the command to finish, then run step 2, then step 3. " +
    "Never run two steps in parallel and never combine them into one command. Do not create any other file.\n" +
    `Step 1 (run exactly): ${step(1, "step1_done.txt")}\n` +
    `Step 2 (run exactly): ${step(2, "step2_ran.txt")}\n` +
    `Step 3 (run exactly): ${step(3, "target_wrong.txt")}\n` +
    "After each step, say which step finished in one short line.";
  return `${instructions}\n\nTASK: ${prompt}`;
}

const STEER_TEXT =
  "CORRECTION from the user, take priority over the original plan: do NOT create target_wrong.txt at all. " +
  "Keep step1_done.txt exactly as it is. Instead create steer_note.txt containing the single line STEERED. " +
  "Then stop.";

const TOOL_POLL_HINT =
  "Before each step you MUST call the MCP tool user_guidance. If it returns any guidance text, that text " +
  "overrides your original plan and you must obey it instead. " +
  "Do this before step 1, before step 2 and before step 3.";

async function runVariant(variant, args) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `ol-probe-steer-${variant}-`));
  const guidanceFile = path.join(dir, "guidance.txt");
  fs.writeFileSync(guidanceFile, "");

  const mcpServers =
    variant === "d_mcp_guidance"
      ? [
          {
            name: "openlive-guidance",
            command: process.execPath,
            args: [SELF_PATH, "--mcp-child", guidanceFile],
            env: [],
          },
        ]
      : [];

  const client = new AcpClient(args.cmd, args.args, dir);
  const result = { variant, dir, advertisedSteering: false, stepsSeen: [], disk: {}, errors: [] };
  const startedAt = Date.now();
  const elapsed = () => Date.now() - startedAt;

  try {
    const init = await client.request(
      "initialize",
      {
        protocolVersion: 1,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
        clientInfo: { name: "openlive-steer-probe", version: "0.1.0" },
      },
      60_000,
    );
    result.agentInfo = init?.agentInfo ?? null;
    result.agentCapabilities = init?.agentCapabilities ?? null;
    // (b) is only legal when the agent advertises it. Look everywhere it could be.
    result.advertisedSteering = JSON.stringify(init ?? {}).includes("_session/steering");

    const session = await client.request(
      "session/new",
      { cwd: dir, mcpServers },
      120_000,
    );
    const sessionId = session?.sessionId;
    if (!sessionId) throw new Error("session/new returned no sessionId");
    result.sessionId = sessionId;
    result.availableCommands = session?.availableCommands ?? null;
    if (JSON.stringify(session ?? {}).includes("_session/steering")) result.advertisedSteering = true;

    const promptText =
      agentText(
        variant === "d_mcp_guidance"
          ? `${TOOL_POLL_HINT} Then run the three steps.`
          : "Run the three steps.",
      );

    // Kick off the task; do not await it — we steer while it runs.
    const promptPromise = client
      .request("session/prompt", { sessionId, prompt: [{ type: "text", text: promptText }] }, args.timeoutMs)
      .then((r) => { result.promptStopReason = r?.stopReason ?? null; result.promptEndedAtMs = elapsed(); })
      .catch((e) => { result.errors.push(`prompt: ${e.message}`); result.promptEndedAtMs = elapsed(); });

    // Watch disk continuously: WHEN a file appeared is what separates a real
    // mid-task steer from a queue-and-apply-afterwards.
    const appeared = {};
    const watcher = setInterval(() => {
      for (const name of ["step1_done.txt", "step2_ran.txt", "target_wrong.txt", "steer_note.txt"]) {
        if (appeared[name] === undefined && fs.existsSync(path.join(dir, name))) appeared[name] = elapsed();
      }
    }, 100);

    // Wait for the first step to land on disk.
    const deadline = Date.now() + args.timeoutMs;
    while (Date.now() < deadline) {
      if (fs.existsSync(path.join(dir, "step1_done.txt"))) break;
      if (client.exit) { result.errors.push(`agent exited before step 1 (${JSON.stringify(client.exit)})`); break; }
      await sleep(250);
    }
    if (!fs.existsSync(path.join(dir, "step1_done.txt"))) {
      result.errors.push("step1_done.txt never appeared — the agent did not run the task");
      result.disk = snapshot(dir);
      result.stderrTail = client.stderrLines.slice(-12);
      clearInterval(watcher);
      return result;
    }
    const step1At = elapsed();
    const step1Bytes = read(path.join(dir, "step1_done.txt"));
    const step1Mtime = fs.statSync(path.join(dir, "step1_done.txt")).mtimeMs;
    result.stepsSeen.push({ step: 1, atMs: step1At, bytes: step1Bytes });

    // Steer at the first completed step, before step 2 (the 8 s gap is the window).
    await sleep(500);
    const steerAt = elapsed();
    let steerPromise = null;
    if (variant === "a_concurrent_prompt") {
      steerPromise = client
        .request("session/prompt", { sessionId, prompt: [{ type: "text", text: STEER_TEXT }] }, 45_000)
        .then((r) => { result.steerResponse = { stopReason: r?.stopReason ?? null, atMs: elapsed() }; })
        .catch((e) => { result.steerResponse = { error: e.message, atMs: elapsed() }; });
    } else if (variant === "b_meta_steering") {
      if (!result.advertisedSteering) {
        result.steerResponse = { skipped: "agent did not advertise _session/steering" };
      } else {
        steerPromise = client
          .request("_session/steering", { sessionId, guidance: STEER_TEXT }, 30_000)
          .then((r) => { result.steerResponse = { ok: r ?? true, atMs: elapsed() }; })
          .catch((e) => { result.steerResponse = { error: e.message, atMs: elapsed() }; });
      }
    } else if (variant === "c_cancel_resume") {
      // ACP's session/cancel is a NOTIFICATION, not a request: sending an id gets -32601.
      const cancelStart = elapsed();
      client.notify("session/cancel", { sessionId });
      result.cancelSent = true;
      result.cancelMs = elapsed() - cancelStart;
      await sleep(600);
      const diskAtCancel = snapshot(dir);
      result.diskAtCancel = Object.keys(diskAtCancel);
      // Wait for the interrupted turn to settle, then build the continuation from DISK STATE.
      await Promise.race([promptPromise, sleep(15_000)]);
      const done = Object.keys(snapshot(dir));
      const continuation =
        `[CONTINUATION] The previous turn was interrupted on purpose. Files already on disk: ${done.join(", ") || "none"}. ` +
        "Do NOT redo those steps and do NOT modify them. " +
        `${STEER_TEXT}`;
      await sleep(400);
      await client
        .request("session/prompt", { sessionId, prompt: [{ type: "text", text: continuation }] }, args.timeoutMs)
        .then((r) => {
          result.continuationStopReason = r?.stopReason ?? null;
          result.promptEndedAtMs = elapsed();
        })
        .catch((e) => { result.errors.push(`continuation prompt: ${e.message}`); });
      result.steerResponse = { note: "session/cancel notification + continuation prompt" };
    } else if (variant === "d_mcp_guidance") {
      fs.writeFileSync(guidanceFile, STEER_TEXT);
      result.guidanceWrittenAtMs = elapsed();
      result.steerResponse = { note: "guidance file written; agent must poll" };
    }
    result.steerAtMs = steerAt;
    result.steerResponse ??= { note: "no steering response" };

    // Let the task play out: the remaining steps are 8 s apart.
    await promptPromise;
    // A queued steer runs after the first turn; give it its own turn to land.
    if (steerPromise) await steerPromise;
    await sleep(3000);

    const final = snapshot(dir);
    result.disk = final;
    if (final["step2_ran.txt"]) result.stepsSeen.push({ step: 2, atMs: elapsed(), bytes: final["step2_ran.txt"].text });
    if (final["target_wrong.txt"]) result.stepsSeen.push({ step: 3, atMs: elapsed(), bytes: final["target_wrong.txt"].text });

    // Evidence, not narration.
    result.step1Untouched =
      final["step1_done.txt"]?.text === step1Bytes && final["step1_done.txt"]?.mtime === step1Mtime;
    result.targetWrongCreated = Boolean(final["target_wrong.txt"]);
    result.steerNoteCreated = Boolean(final["steer_note.txt"]);
    result.lossless = !result.targetWrongCreated && result.step1Untouched;
    result.guidanceAcked = fs.existsSync(`${guidanceFile}.ack`);
    result.updates = client.updates
      .filter((u) => u.method === "session/update")
      .map((u) => {
        const upd = u.params?.update ?? {};
        const text = upd.content?.text ?? upd.content?.content?.text ?? "";
        return `${upd.sessionUpdate ?? "?"}${text ? `: ${String(text).slice(0, 160)}` : ""}`;
      })
      .slice(-15);
    result.finalFiles = Object.keys(final);
    clearInterval(watcher);
    // When each file appeared, relative to the steer. A steer that lands before
    // the next step is created is a mid-task steer; one that lands after the
    // whole turn is queue-only.
    result.appearedAtMs = appeared;
    result.steerAtMs = steerAt;
    result.steerLandedBeforeNextStep =
      appeared["steer_note.txt"] !== undefined &&
      appeared["step2_ran.txt"] === undefined &&
      appeared["target_wrong.txt"] === undefined;
    result.steerDeliveredBeforeTurnEnd =
      appeared["steer_note.txt"] !== undefined &&
      result.promptEndedAtMs !== undefined &&
      appeared["steer_note.txt"] <= result.promptEndedAtMs;
    result.stderrTail = client.stderrLines.slice(-8);    return result;
  } catch (e) {
    result.errors.push(e.message);
    result.disk = snapshot(dir);
    clearInterval(watcher);
    result.stderrTail = client.stderrLines.slice(-15);
    return result;
  } finally {
    result.elapsedMs = elapsed();
    await client.dispose();
    if (!args.keep) {
      try { fs.rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* Windows keeps a child-held folder */ }
    }
  }
}

function verdictFor(losslessVariants, results) {
  // Rank order from docs/04-REVIEWER-NOTES.md §4, best first — but only among
  // variants that were BOTH lossless and delivered mid-task. A lossless variant
  // that only lands after the turn ended is queue-only, not steering.
  const midTask = (v) =>
    results[v]?.lossless === true &&
    results[v]?.steerLandedBeforeNextStep === true &&
    results[v]?.steerDeliveredBeforeTurnEnd === true;
  if (midTask("b_meta_steering")) return "meta_steering";
  if (midTask("d_mcp_guidance")) return "mcp_guidance";
  if (midTask("a_concurrent_prompt")) return "concurrent_prompt";
  if (midTask("c_cancel_resume")) return "cancel_resume";
  // Nothing measured steers mid-task: the documented safe default is
  // boundary-cancel, which at worst loses the in-flight step.
  return "cancel_resume";
}

const STAGE0_AGENTS = [
  { id: "hermes", cmd: "hermes", args: ["acp"] },
  { id: "claude-code", cmd: "npx", args: ["-y", "@agentclientprotocol/claude-agent-acp@0.100.0"] },
];

async function main() {
  const args = parseArgs(process.argv);
  const agents = args.all ? STAGE0_AGENTS : [{ id: "hermes", cmd: args.cmd, args: args.args }];
  const variants = args.variant ? [args.variant] : ["a_concurrent_prompt", "b_meta_steering", "c_cancel_resume", "d_mcp_guidance"];
  const report = { probe: "acp-steer", at: new Date().toISOString(), agents: {} };

  for (const agent of agents) {
    args.cmd = agent.cmd;
    args.args = agent.args;
    const results = {};
    for (const variant of variants) {
      process.stderr.write(`[probe] ${agent.id}: ${variant}\n`);
      results[variant] = await runVariant(variant, args);
      process.stderr.write(
        `[probe] ${agent.id}: ${variant} lossless=${results[variant].lossless} files=${JSON.stringify(results[variant].finalFiles ?? [])}\n`,
      );
    }
    const lossless = variants.filter((v) => results[v].lossless === true);
    const verdict = verdictFor(lossless, results);
    report.agents[agent.id] = {
      verdict,
      losslessVariants: lossless,
      measured: Object.fromEntries(
        variants.map((v) => [
          v,
          {
            lossless: results[v].lossless ?? false,
            targetWrongCreated: results[v].targetWrongCreated ?? null,
            step1Untouched: results[v].step1Untouched ?? null,
            steerNoteCreated: results[v].steerNoteCreated ?? false,
            guidanceAcked: results[v].guidanceAcked ?? false,
            steerDeliveredBeforeTurnEnd: results[v].steerDeliveredBeforeTurnEnd ?? false,
            steerLandedBeforeNextStep: results[v].steerLandedBeforeNextStep ?? false,
            appearedAtMs: results[v].appearedAtMs ?? null,
            promptEndedAtMs: results[v].promptEndedAtMs ?? null,
            updates: results[v].updates ?? [],
            files: results[v].finalFiles ?? [],
            advertisedSteering: results[v].advertisedSteering ?? false,
            steerResponse: results[v].steerResponse ?? null,
            errors: results[v].errors ?? [],
          },
        ]),
      ),
    };
  }

  console.log(JSON.stringify(report, null, 2));
  const anyLossless = Object.values(report.agents).some((a) => a.losslessVariants.length > 0);
  if (!anyLossless) process.exitCode = 2;
}

// This file doubles as the stdio MCP server for variant (d): the probe writes a
// guidance file, the agent's MCP client calls user_guidance, the child drains it.
// The dispatch sits last so every class above is initialized before main() runs.
if (process.argv[2] === "--mcp-child") {
  runMcpGuidanceServer(process.argv[3]);
} else {
  main().catch((err) => {
    console.error(`probe failed: ${err?.message ?? err}`);
    process.exit(1);
  });
}
