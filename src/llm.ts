import { $, saveJSON } from "./utils";
export interface RunReport {
  version: 1;
  kind: string;
  model: string;
  steps: number;
  durationSeconds: number;
  adapterBytes: number;
  adapterSHA256: string;
  metrics: { step: number; split: string; loss: number }[];
  comparisons: {
    prompt: string;
    before: string;
    after: string;
    reference: string;
  }[];
  limitations: string;
  completedAt: string;
  [key: string]: unknown;
}
export function validateReport(value: unknown): RunReport {
  const r = value as RunReport;
  if (
    !r ||
    r.version !== 1 ||
    r.kind !== "actual-local-lora-run" ||
    typeof r.model !== "string" ||
    !Number.isFinite(r.steps) ||
    r.steps <= 0 ||
    r.steps > 1000000 ||
    !Number.isFinite(r.durationSeconds) ||
    r.durationSeconds < 0 ||
    !Number.isFinite(r.adapterBytes) ||
    r.adapterBytes <= 0 ||
    typeof r.adapterSHA256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(r.adapterSHA256) ||
    !Array.isArray(r.metrics) ||
    !Array.isArray(r.comparisons) ||
    r.comparisons.length > 20 ||
    r.comparisons.some(
      (c) =>
        !c ||
        [c.prompt, c.before, c.after, c.reference].some(
          (s) => typeof s !== "string" || s.length > 10000,
        ),
    ) ||
    r.metrics.some(
      (m) => !m || !Number.isFinite(m.step) || !Number.isFinite(m.loss),
    )
  )
    throw Error("Unsupported or invalid training report");
  return r;
}
let report: RunReport | null = null,
  token = "",
  poll: ReturnType<typeof setInterval> | null = null;
export function renderRun(r: RunReport) {
  report = validateReport(r);
  $("run-summary").textContent =
    `${r.steps} real optimizer steps · ${r.durationSeconds.toFixed(1)} seconds · ${(r.adapterBytes / 1024 / 1024).toFixed(2)} MB adapter`;
  $("run-model").textContent = r.model;
  $("run-sha").textContent = r.adapterSHA256;
  $("run-limit").textContent = r.limitations;
  $("comparisons").replaceChildren(
    ...r.comparisons.map((c) => {
      const card = document.createElement("article");
      const prompt = document.createElement("h3");
      prompt.textContent = c.prompt;
      card.append(prompt);
      for (const [label, text] of [
        ["BEFORE / BASE WEIGHTS", c.before],
        ["AFTER / TRAINED ADAPTER", c.after],
        ["REFERENCE / SYNTHETIC DATA", c.reference],
      ]) {
        const section = document.createElement("section"),
          small = document.createElement("small"),
          p = document.createElement("p");
        small.textContent = label;
        p.textContent = text;
        section.append(small, p);
        card.append(section);
      }
      return card;
    }),
  );
  const canvas = $<HTMLCanvasElement>("loss"),
    ctx = canvas.getContext("2d")!;
  canvas.width = 1000;
  canvas.height = 270;
  ctx.fillStyle = "#12232d";
  ctx.fillRect(0, 0, 1000, 270);
  const max = Math.max(1, ...r.metrics.map((m) => m.loss));
  for (const split of ["train", "val"]) {
    const rows = r.metrics.filter((m) => m.split === split);
    ctx.strokeStyle = split === "train" ? "#edb681" : "#8bc5c1";
    ctx.lineWidth = 3;
    ctx.beginPath();
    rows.forEach((m, i) => {
      const x = 50 + (m.step / r.steps) * 900,
        y = 220 - (m.loss / max) * 180;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.font = "12px monospace";
  ctx.fillStyle = "#b5c7c4";
  ctx.fillText(`ACTUAL LOGGED CROSS-ENTROPY / ${r.steps} STEPS`, 40, 30);
  ctx.fillStyle = "#edb681";
  ctx.fillText("TRAIN", 40, 250);
  ctx.fillStyle = "#8bc5c1";
  ctx.fillText("VALIDATION", 140, 250);
  ctx.fillStyle = "#b5c7c4";
  ctx.fillText(
    "A toy formatting task is not a general intelligence benchmark.",
    420,
    250,
  );
}
export async function initLLM() {
  try {
    const response = await fetch(
      import.meta.env.BASE_URL + "training-run.json",
    );
    if (!response.ok)
      throw Error(
        "No bundled training report. Run the local trainer or import a report.",
      );
    renderRun(await response.json());
  } catch (e) {
    $("trainer-status").textContent = (e as Error).message;
  }
  $("download-run").onclick = () => {
    if (report) saveJSON(report, "local-lora-report.json");
  };
  $<HTMLInputElement>("report-file").onchange = async () => {
    try {
      const file = $<HTMLInputElement>("report-file").files?.[0];
      if (!file) return;
      if (file.size > 2 * 1024 * 1024) throw Error("Report exceeds 2 MB");
      renderRun(JSON.parse(await file.text()));
      $("trainer-status").textContent =
        "Imported report · measurements are supplied by the file, not independently reverified";
    } catch (e) {
      $("trainer-status").textContent = (e as Error).message;
    } finally {
      $<HTMLInputElement>("report-file").value = "";
    }
  };
  if (!["localhost", "127.0.0.1"].includes(location.hostname)) return;
  try {
    const response = await fetch("/trainer/health");
    if (!response.ok) return;
    const health = await response.json();
    if (!health.localTrainer) return;
    token = health.token;
    $<HTMLButtonElement>("start-training").disabled = false;
    $("trainer-status").textContent =
      "Local MLX trainer connected. Starting a run changes real adapter weights on this machine.";
  } catch {}
  $("start-training").onclick = async () => {
    try {
      const response = await fetch("/trainer/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          steps: Number($<HTMLSelectElement>("steps").value),
          learningRate: 0.0003,
          dataset: $<HTMLTextAreaElement>("custom-data").value.trim() || null,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw Error(data.error);
      $<HTMLButtonElement>("start-training").disabled = true;
      $<HTMLButtonElement>("cancel-training").disabled = false;
      if (poll) clearInterval(poll);
      poll = setInterval(async () => {
        try {
          const response = await fetch("/trainer/status");
          const state = await response.json();
          $("trainer-log").textContent = state.log;
          $("trainer-status").textContent = `Local job: ${state.status}`;
          if (["complete", "failed", "cancelled"].includes(state.status)) {
            clearInterval(poll!);
            poll = null;
            $<HTMLButtonElement>("start-training").disabled = false;
            $<HTMLButtonElement>("cancel-training").disabled = true;
            if (state.report) renderRun(state.report);
          }
        } catch (e) {
          $("trainer-status").textContent =
            `Trainer connection lost: ${String(e)}`;
        }
      }, 1200);
    } catch (e) {
      $("trainer-status").textContent = (e as Error).message;
    }
  };
  $("cancel-training").onclick = async () => {
    await fetch("/trainer/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
  };
}
