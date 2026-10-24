import { it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
it("published adapter and frozen feature manifest match the actual evidence", () => {
  const report = JSON.parse(readFileSync("public/training-run.json", "utf8"));
  const adapter = readFileSync(
    "artifacts/example-adapter/adapters.safetensors",
  );
  expect(createHash("sha256").update(adapter).digest("hex")).toBe(
    report.adapterSHA256,
  );
  expect(adapter.byteLength).toBe(report.adapterBytes);
  expect(report.metrics.some((m: { split: string }) => m.split === "val")).toBe(
    true,
  );
  const snapshot = JSON.parse(readFileSync("public/embeddings.json", "utf8"));
  expect(snapshot.features).toHaveLength(36);
  expect(snapshot.features[0]).toHaveLength(256);
  expect(
    createHash("sha256")
      .update(readFileSync("public/model/model.json"))
      .digest("hex"),
  ).toBe(snapshot.weightsManifestSHA256);
});
