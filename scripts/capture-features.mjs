import { chromium } from "@playwright/test";
import { writeFile } from "node:fs/promises";
const browser = await chromium.launch({ channel: "chromium", headless: true });
try {
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") console.log(m.text());
  });
  await page.goto(process.env.APP_URL || "http://127.0.0.1:4197");
  await page.evaluate(() => window.visualModelLab.compute());
  const snapshot = await page.evaluate(
    () => window.visualModelLab.featureSnapshot,
  );
  if (!snapshot)
    throw Error(await page.locator("#feature-status").textContent());
  await writeFile("public/embeddings.json", JSON.stringify(snapshot));
  console.log(
    `Preserved ${snapshot.features.length} real ${snapshot.features[0].length}-dimensional embeddings (${snapshot.backend}).`,
  );
} finally {
  await browser.close();
}
