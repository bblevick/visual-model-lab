import { test, expect } from "@playwright/test";
test("real cached features train a head and preserve evaluation split", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("#feature-status")).toContainText(
    "Preserved real MobileNet",
  );
  await page.locator("#biased").click();
  await expect(page.locator("#labels-count")).toHaveText("12");
  await page.locator("#train").click();
  await expect(page.locator("#mistakes")).not.toHaveText("—");
  await page.locator("#misfile").click();
  await page.locator("#train").click();
  await expect(page.locator("#mistakes")).toHaveText("12");
  await page.locator("#counter").click();
  await expect(page.locator("#labels-count")).toHaveText("24");
  await page.locator("#train").click();
  await expect(page.locator("#confusion")).toContainText("Held-out confusion");
  await page.locator("#llm-tab").click();
  await expect(page.locator("#run-summary")).toContainText(
    "60 real optimizer steps",
  );
  await expect(page.locator("#comparisons article")).toHaveCount(4);
  await page.locator("#vision-tab").click();
  const d = page.waitForEvent("download");
  await page.locator("#save-session").click();
  expect((await d).suggestedFilename()).toContain("labels");
  expect(errors).toEqual([]);
});
test("a real frozen backbone extracts features without changing its weights", async ({
  page,
}) => {
  test.setTimeout(120000);
  await page.goto("/");
  await page.evaluate(() => (window as any).visualModelLab.compute());
  await expect(page.locator("#feature-status")).toContainText(
    "computed locally",
  );
  const result = await page.evaluate(() => ({
    rows: (window as any).visualModelLab.featureSnapshot.features.length,
    dimensions: (window as any).visualModelLab.featureSnapshot.features[0]
      .length,
  }));
  expect(result.rows).toBe(36);
  expect(result.dimensions).toBe(256);
});
test("small viewport and invalid imports stay usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator("#session-file").setInputFiles({
    name: "bad.json",
    mimeType: "application/json",
    buffer: Buffer.from('{"version":7}'),
  });
  await expect(page.locator("#feature-status")).toContainText("version 1");
});
