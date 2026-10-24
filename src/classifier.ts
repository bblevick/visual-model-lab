export interface LinearModel {
  weights: number[];
  bias: number;
}
export function normalize(v: number[]) {
  const length = Math.hypot(...v) || 1;
  return v.map((x) => x / length);
}
export const probability = (m: LinearModel, x: number[]) =>
  1 /
  (1 +
    Math.exp(
      -Math.max(
        -30,
        Math.min(30, m.bias + m.weights.reduce((s, w, i) => s + w * x[i], 0)),
      ),
    ));
export function fitClassifier(
  features: number[][],
  labels: (0 | 1 | null)[],
  epochs = 350,
) {
  if (
    features.length !== labels.length ||
    !features.length ||
    !features.every(
      (x) => x.length === features[0].length && x.every(Number.isFinite),
    )
  )
    throw Error("Feature dimensions do not agree");
  const rows = labels
    .map((y, i) => ({ y, x: features[i] }))
    .filter((r) => r.y !== null);
  if (!rows.some((r) => r.y === 0) || !rows.some((r) => r.y === 1))
    throw Error("Label at least one sun and one moon before training.");
  const m: LinearModel = {
    weights: Array(features[0].length).fill(0),
    bias: 0,
  };
  for (let t = 0; t < epochs; t++) {
    const dw = m.weights.map((w) => 0.0002 * w);
    let db = 0;
    for (const r of rows) {
      const error = probability(m, r.x) - r.y!;
      db += error / rows.length;
      for (let k = 0; k < dw.length; k++)
        dw[k] += (error * r.x[k]) / rows.length;
    }
    for (let k = 0; k < dw.length; k++) m.weights[k] -= 8 * dw[k];
    m.bias -= 8 * db;
  }
  return m;
}
export function projection(features: number[][]) {
  const dims = features[0].length,
    mean = Array.from(
      { length: dims },
      (_, k) => features.reduce((sum, v) => sum + v[k], 0) / features.length,
    ),
    centered = features.map((v) => v.map((x, k) => x - mean[k]));
  const dot = (a: number[], b: number[]) =>
    a.reduce((s, x, k) => s + x * b[k], 0);
  function axis(previous?: number[]) {
    let v = normalize(
      Array.from({ length: dims }, (_, k) => Math.sin(k * 1.7 + 0.5)),
    );
    for (let iter = 0; iter < 40; iter++) {
      const next = Array(dims).fill(0);
      for (const x of centered) {
        const factor = dot(x, v);
        for (let k = 0; k < dims; k++) next[k] += x[k] * factor;
      }
      if (previous) {
        const overlap = dot(next, previous);
        for (let k = 0; k < dims; k++) next[k] -= previous[k] * overlap;
      }
      v = normalize(next);
    }
    return v;
  }
  const u = axis(),
    v = axis(u),
    points = centered.map((x) => [dot(x, u), dot(x, v)]),
    range = Math.max(0.01, ...points.flat().map(Math.abs)) * 1.18;
  return { mean, u, v, points, range };
}
export function confusion(
  model: LinearModel,
  features: number[][],
  truth: (0 | 1)[],
) {
  const matrix = [
    [0, 0],
    [0, 0],
  ];
  truth.forEach(
    (y, i) => matrix[y][Number(probability(model, features[i]) >= 0.5)]++,
  );
  return matrix;
}
export function parseLabels(value: unknown, trainIds: string[]) {
  const v = value as { version: number; labels: Record<string, unknown> };
  if (
    !v ||
    v.version !== 1 ||
    !v.labels ||
    typeof v.labels !== "object" ||
    Array.isArray(v.labels)
  )
    throw Error("Use a version 1 stamp-label session.");
  const labels: Record<string, 0 | 1 | null> = {};
  for (const [id, label] of Object.entries(v.labels)) {
    if (!trainIds.includes(id) || ![0, 1, null].includes(label as any))
      throw Error("Labels may refer only to known training stamps.");
    labels[id] = label as 0 | 1 | null;
  }
  return labels;
}
