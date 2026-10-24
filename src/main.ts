import "./style.css";
import { $, saveJSON } from "./utils";
import { stamps, drawStamp } from "./stamps";
import {
  fitClassifier,
  probability,
  confusion,
  parseLabels,
  type LinearModel,
} from "./classifier";
import { plot } from "./plot";
import { initLLM } from "./llm";
$("app").innerHTML =
  `<header><a href="./" class="brand">THE ALMOST-RIGHT DEPARTMENT <small>/ VISUAL MODEL LAB</small></a><span>A MACHINE CAN LEARN THE WRONG LESSON VERY WELL.</span></header><main><section class="intro"><div><p class="eyebrow">TWO EXPERIMENTS IN TEACHING SMALL MACHINES</p><h1>What did it<br><em>actually learn?</em></h1></div><p>A stamp inspector. A night clerk.<br>Two ways to change a model’s mind.</p></section><nav class="mode-tabs"><button id="vision-tab" aria-pressed="true">01 / The stamp inspector</button><button id="llm-tab" aria-pressed="false">02 / Teach the night clerk</button></nav><section id="vision-panel"><div class="experiment-note"><p><b>THE QUESTION</b> Is it learning sun versus moon—or just red ink versus blue?</p><span>FROZEN MOBILENET FEATURES + YOUR SMALL CLASSIFIER</span></div><div class="vision-grid"><section class="plot-wrap"><canvas id="plot" width="1000" height="610" aria-label="Projection of frozen image features; points are stamps and rings mark held-out examples."></canvas><div id="feature-status" role="status">Opening the feature cabinet…</div><div class="plot-controls"><button id="biased">1. Label the easy batch</button><button id="counter">2. Add the awkward stamps</button><button id="misfile">Deliberately misfile this drawer</button><button id="train" class="primary">Train this small classifier</button></div><div class="score-row"><span><b id="labels-count">0</b> LABELED TRAINING STAMPS</span><span><b id="mistakes">—</b> ERRORS / 12 HELD OUT</span><span><b id="dimensions">—</b> FROZEN DIMENSIONS</span></div><p id="lesson">The network that sees the images stays frozen. Your labels train only the small classification head.</p></section><aside><p class="eyebrow">YOU ARE THE SUPERVISION</p><h2>Teach it what<br>a stamp means.</h2><p>The easy batch pairs sun with red and moon with blue. The awkward batch breaks that shortcut. A strong feature lens may already separate the shapes. Deliberately misfile the labels to see what bad supervision does.</p><div class="label-tools"><button id="label-sun" aria-pressed="true">☀ Sun</button><button id="label-moon" aria-pressed="false">☾ Moon</button><button id="label-erase" aria-pressed="false">Erase</button></div><p class="small">Choose a label, then click training stamps. Held-out stamps never enter training.</p><button id="clear" class="outline">Clear all labels</button><button id="save-session" class="outline">↓ Save the lesson</button><label class="file-button" for="session-file">↑ Import labels</label><input id="session-file" class="sr-only" type="file" accept="application/json,.json"><details><summary>The machinery underneath</summary><p>Precomputed, real MobileNet V1 0.25 embeddings of original synthetic stamp illustrations. A logistic head trains on the full vectors. The plot is a PCA projection; its colored background is a 2D slice, not the entire decision surface.</p><button id="recompute">Recompute the frozen features locally</button></details></aside></div><section class="gallery"><div><p class="eyebrow">THE TRAINING DRAWER</p><h2>Some labels are more useful<br>than others.</h2></div><div id="training-stamps" class="stamp-grid"></div><details><summary>The sealed drawer · 12 held-out stamps</summary><p>True classes come from the generated sun/moon shape, not the ink. Evaluation always uses these untouched stamps.</p><div id="test-stamps" class="stamp-grid"></div><div id="confusion"></div></details></section></section><section id="llm-panel" hidden><div class="experiment-note"><p><b>THE QUESTION</b> Can a very small language model learn to stop embellishing a dispatch notice?</p><span>REAL LOCAL MLX LORA / SMOLLM2 135M / APPLE SILICON</span></div><div class="training-grid"><section><p class="eyebrow">A RECORDED, ACTUALLY EXECUTED RUN</p><h2 id="run-model">Loading the run…</h2><p id="run-summary"></p><canvas id="loss" width="1000" height="270" aria-label="Actual training and validation loss from a local LoRA run"></canvas><p id="run-limit" class="small"></p><div id="comparisons"></div></section><aside><p class="eyebrow">CHANGE REAL WEIGHTS</p><h2>Teach the<br>night clerk.</h2><p>The base model stays frozen. A small adapter learns a constrained dispatch format from original synthetic examples.</p><label for="steps">Optimizer steps</label><select id="steps"><option>20</option><option selected>60</option><option>120</option></select><details><summary>Use your own examples · JSONL</summary><p>At least 12 unique user/assistant pairs. The trainer creates disjoint 70/15/15 splits. Nothing is uploaded.</p><textarea id="custom-data" rows="6" placeholder='{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'></textarea></details><button id="start-training" class="primary" disabled>Run local fine-tuning</button><button id="cancel-training" class="outline" disabled>Stop this run</button><p id="trainer-status" role="status">Public demo: recorded evidence. For live training, run the local trainer below.</p><pre id="trainer-log"></pre><div class="local-command"><span>ON AN APPLE SILICON MAC</span><code>npm run llm:setup<br>npm run llm</code><p>Then open the local URL. First setup downloads the model; subsequent training is local.</p></div><button id="download-run" class="outline">↓ Export this run</button><label for="report-file" class="file-button">↑ Import a real run report</label><input id="report-file" class="sr-only" type="file" accept="application/json,.json"><details><summary>Adapter fingerprint</summary><code id="run-sha"></code></details></aside></div></section><section class="afterword"><h2>Confidence is a number.<br>Understanding is a question.</h2><p>Both experiments preserve the data, the assumptions, and the mistakes. A classifier can exploit ink color. A tiny language model can learn a format. Neither result proves it understands the job.</p></section></main><footer><span>BUILT WITH AI. DIRECTED BY BEN LEVICKI.</span><span>LABELS ARE DECISIONS. KEEP THE RECEIPTS.</span></footer>`;
let features: number[][] = [],
  model: LinearModel | null = null,
  active: 0 | 1 | null = 0,
  labels: Record<string, 0 | 1 | null> = {};
function drawGallery() {
  for (const split of ["train", "test"]) {
    $(split === "train" ? "training-stamps" : "test-stamps").replaceChildren(
      ...stamps
        .filter((s) => s.split === split)
        .map((s) => {
          const button = document.createElement("button"),
            canvas = document.createElement("canvas");
          drawStamp(canvas, s);
          button.append(canvas);
          const text = document.createElement("span");
          text.textContent =
            split === "train"
              ? labels[s.id] === 0
                ? "☀ LABELED SUN"
                : labels[s.id] === 1
                  ? "☾ LABELED MOON"
                  : "UNLABELED"
              : `HELD OUT / ${s.kind === 0 ? "SUN" : "MOON"}`;
          button.append(text);
          button.setAttribute("aria-label", `${s.id}: ${text.textContent}`);
          button.className =
            labels[s.id] === 0 ? "sun" : labels[s.id] === 1 ? "moon" : "";
          if (split === "train")
            button.onclick = () => {
              labels[s.id] = active;
              model = null;
              refresh();
            };
          else button.disabled = true;
          return button;
        }),
    );
  }
}
function refresh() {
  drawGallery();
  $("labels-count").textContent = String(
    Object.values(labels).filter((v) => v !== null).length,
  );
  $("dimensions").textContent = features[0] ? String(features[0].length) : "—";
  if (features.length) plot($<HTMLCanvasElement>("plot"), features, model);
  if (model) {
    const held = stamps
        .map((s, i) => ({ s, i }))
        .filter((r) => r.s.split === "test"),
      matrix = confusion(
        model,
        held.map((r) => features[r.i]),
        held.map((r) => r.s.kind),
      );
    $("mistakes").textContent = String(matrix[0][1] + matrix[1][0]);
    $("confusion").textContent =
      `Held-out confusion · true SUN → predicted SUN ${matrix[0][0]}, MOON ${matrix[0][1]} · true MOON → predicted SUN ${matrix[1][0]}, MOON ${matrix[1][1]}`;
  } else {
    $("mistakes").textContent = "—";
    $("confusion").textContent = "Train a classifier to inspect its mistakes.";
  }
}
$("biased").onclick = () => {
  labels = {};
  stamps
    .filter(
      (s) =>
        s.split === "train" &&
        ((s.kind === 0 && s.ink === "red") ||
          (s.kind === 1 && s.ink === "blue")),
    )
    .forEach((s) => (labels[s.id] = s.kind));
  model = null;
  $("lesson").textContent =
    "A biased lesson: all labeled suns are red, all labeled moons are blue. Train and inspect the held-out mistakes.";
  refresh();
};
$("misfile").onclick = () => {
  if (!Object.values(labels).some((v) => v !== null))
    stamps
      .filter((s) => s.split === "train")
      .forEach((s) => (labels[s.id] = s.kind));
  for (const id of Object.keys(labels))
    if (labels[id] !== null) labels[id] = labels[id] === 0 ? 1 : 0;
  model = null;
  $("lesson").textContent =
    "An intentional clerical error: existing labels are reversed. The held-out truth has not changed. Train and inspect the damage.";
  refresh();
};
$("counter").onclick = () => {
  stamps
    .filter((s) => s.split === "train")
    .forEach((s) => (labels[s.id] = s.kind));
  model = null;
  $("lesson").textContent =
    "Counterexamples added. Train again; inspect the measured result rather than assuming it improved.";
  refresh();
};
$("train").onclick = () => {
  try {
    if (!features.length)
      throw Error("Wait for or recompute the frozen image features first.");
    model = fitClassifier(
      features,
      stamps.map((s) => (s.split === "train" ? (labels[s.id] ?? null) : null)),
    );
    refresh();
    $("feature-status").textContent =
      "Classifier trained locally. The feature network has not changed.";
  } catch (e) {
    $("feature-status").textContent = (e as Error).message;
  }
};
for (const [id, value] of [
  ["sun", 0],
  ["moon", 1],
  ["erase", null],
] as const)
  $("label-" + id).onclick = () => {
    active = value;
    ["sun", "moon", "erase"].forEach((k) =>
      $("label-" + k).setAttribute("aria-pressed", String(k === id)),
    );
  };
$("clear").onclick = () => {
  labels = {};
  model = null;
  refresh();
};
$("save-session").onclick = () =>
  saveJSON({ version: 1, labels }, "stamp-inspector-labels.json");
$<HTMLInputElement>("session-file").onchange = async () => {
  try {
    const file = $<HTMLInputElement>("session-file").files?.[0];
    if (!file) return;
    if (file.size > 100000) throw Error("Label file exceeds 100 KB");
    labels = parseLabels(
      JSON.parse(await file.text()),
      stamps.filter((s) => s.split === "train").map((s) => s.id),
    );
    model = null;
    refresh();
    $("feature-status").textContent =
      "Imported training labels. Train to fit the head to this lesson.";
  } catch (e) {
    $("feature-status").textContent = (e as Error).message;
  } finally {
    $<HTMLInputElement>("session-file").value = "";
  }
};
async function compute() {
  const button = $<HTMLButtonElement>("recompute");
  button.disabled = true;
  try {
    const { embed, backend } = await import("./features");
    const next: number[][] = [];
    for (const [sindex, s] of stamps.entries()) {
      $("feature-status").textContent =
        `Computing real frozen image embeddings: ${sindex + 1} / ${stamps.length}`;
      next.push(await embed(s));
    }
    features = next;
    model = null;
    refresh();
    $("feature-status").textContent =
      `Real MobileNet embeddings computed locally · ${backend()} backend`;
    (window as any).visualModelLab.featureSnapshot = {
      version: 1,
      model: "MobileNet V1 0.25 224",
      backend: backend(),
      computedAt: new Date().toISOString(),
      ids: stamps.map((s) => s.id),
      features,
    };
  } catch (e) {
    $("feature-status").textContent =
      `Feature extraction failed: ${(e as Error).message}`;
  } finally {
    button.disabled = false;
  }
}
$("recompute").onclick = () => {
  void compute();
};
for (const mode of ["vision", "llm"])
  $(mode + "-tab").onclick = () => {
    $("vision-panel").hidden = mode !== "vision";
    $("llm-panel").hidden = mode !== "llm";
    $("vision-tab").setAttribute("aria-pressed", String(mode === "vision"));
    $("llm-tab").setAttribute("aria-pressed", String(mode === "llm"));
  };
refresh();
void (async () => {
  try {
    const response = await fetch(import.meta.env.BASE_URL + "embeddings.json");
    if (!response.ok)
      throw Error("No cached feature snapshot. Use “Recompute” to build it.");
    const data = await response.json();
    if (
      data.version !== 1 ||
      JSON.stringify(data.ids) !== JSON.stringify(stamps.map((s) => s.id)) ||
      data.features.length !== stamps.length ||
      !data.features.every(
        (v: number[]) =>
          v.length === data.features[0].length &&
          v.length <= 4096 &&
          v.every(Number.isFinite),
      )
    )
      throw Error("Feature snapshot is incompatible");
    features = data.features;
    refresh();
    $("feature-status").textContent =
      `Preserved real MobileNet image embeddings · ${data.backend} · ${data.computedAt.slice(0, 10)}`;
  } catch (e) {
    $("feature-status").textContent = (e as Error).message;
  }
})();
void initLLM();
(window as any).visualModelLab = {
  get features() {
    return features;
  },
  get labels() {
    return labels;
  },
  get model() {
    return model;
  },
  featureSnapshot: null,
  compute,
  probability,
};
