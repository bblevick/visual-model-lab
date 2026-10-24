import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
const source =
  "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/";
await mkdir("public/model", { recursive: true });
const response = await fetch(source + "model.json");
if (!response.ok) throw Error("Model manifest download failed");
const manifest = await response.text(),
  model = JSON.parse(manifest),
  files = model.weightsManifest.flatMap((w) => w.paths),
  hashes = {};
await writeFile("public/model/model.json", manifest);
hashes["model.json"] = createHash("sha256").update(manifest).digest("hex");
for (let start = 0; start < files.length; start += 8)
  await Promise.all(
    files.slice(start, start + 8).map(async (file) => {
      const response = await fetch(source + file);
      if (!response.ok) throw Error(`Weight download failed: ${file}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      await writeFile("public/model/" + file, bytes);
      hashes[file] = createHash("sha256").update(bytes).digest("hex");
    }),
  );
await writeFile(
  "public/model-provenance.json",
  JSON.stringify(
    {
      source,
      model: "MobileNet V1 0.25 224",
      embeddingLayer: "global_average_pooling2d_1",
      inputRange: [-1, 1],
      license: "Apache-2.0; TensorFlow.js model distribution",
      sha256: hashes,
    },
    null,
    2,
  ),
);
console.log(`Preserved ${files.length} model shards.`);
