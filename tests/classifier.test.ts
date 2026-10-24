import { describe, it, expect } from "vitest";
import {
  fitClassifier,
  probability,
  confusion,
  parseLabels,
  projection,
} from "../src/classifier";
import { validateReport } from "../src/llm";
describe("small classifier and immutable evaluation", () => {
  it("learns separable labels and does not fit unlabeled holdouts", () => {
    const x = [
        [1, 0],
        [0.8, 0.2],
        [0, 1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.2, 0.8],
      ],
      labels: (0 | 1 | null)[] = [0, 0, 1, 1, null, null],
      m = fitClassifier(x, labels);
    expect(probability(m, x[4])).toBeLessThan(0.5);
    expect(probability(m, x[5])).toBeGreaterThan(0.5);
    expect(confusion(m, [x[4], x[5]], [0, 1])).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });
  it("requires both classes and protects held-out IDs", () => {
    expect(() =>
      fitClassifier(
        [
          [1, 0],
          [0, 1],
        ],
        [0, null],
      ),
    ).toThrow(/sun/);
    expect(() =>
      parseLabels({ version: 1, labels: { heldout: 1 } }, ["train"]),
    ).toThrow();
  });
  it("projects finite points with orthogonal axes", () => {
    const p = projection([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(p.points.flat().every(Number.isFinite)).toBe(true);
    expect(p.u.reduce((s, x, i) => s + x * p.v[i], 0)).toBeCloseTo(0, 6);
  });
  it("rejects fabricated report shapes and malformed losses", () => {
    expect(() => validateReport({ version: 1, kind: "demo" })).toThrow();
  });
});
