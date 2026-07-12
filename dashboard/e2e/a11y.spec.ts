import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// The dashboard is one page; wait until the async data has rendered all three views.
async function ready(page: import("@playwright/test").Page) {
  await page.goto("/");
  // Scroll acts reveal on intersection; force them present so axe scans every act and the
  // full-page screenshot isn't blank below the fold.
  await page.addStyleTag({ content: ".reveal{opacity:1 !important;transform:none !important}" });
  // biome-ignore lint/performance/useTopLevelRegex: The regex is only used for a single string match, so it doesn't need to be top-level.
  await expect(page.getByRole("heading", { name: /life-cycle climate impact/i })).toBeVisible({
    timeout: 20_000,
  });
  // open the disclosure tables so axe also scans them
  for (const d of await page.locator("details.datatable > summary").all()) await d.click();
}

for (const colorScheme of ["light", "dark"] as const) {
  test.describe(`accessibility (${colorScheme})`, () => {
    test.use({ colorScheme });

    test("no serious or critical WCAG 2 A/AA violations", async ({ page }) => {
      await ready(page);
      const { violations } = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa"])
        .analyze();
      const impactful = violations.filter((v) => ["serious", "critical"].includes(v.impact ?? ""));
      expect(impactful, JSON.stringify(impactful, null, 2)).toEqual([]);
    });
  });
}

test("reduced motion disables the reveal and bar-grow", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /choose the path to 2050/i })).toBeVisible({
    timeout: 20_000,
  });
  // A below-the-fold reveal act is fully opaque without being scrolled into view.
  const opacity = await page.locator("#payoff").evaluate((el) => getComputedStyle(el).opacity);
  expect(opacity).toBe("1");
  // Smooth anchor scrolling is off.
  const scrollBehavior = await page.evaluate(
    () => getComputedStyle(document.documentElement).scrollBehavior,
  );
  expect(scrollBehavior).toBe("auto");
  // The bar-grow entrance is suppressed even once the charts are in view (grow class added).
  await page.locator("#payoff").scrollIntoViewIfNeeded();
  await expect(page.locator(".payoff--grow")).toBeVisible();
  const animation = await page
    .locator(".payoff__bar")
    .first()
    .evaluate((el) => getComputedStyle(el).animationName);
  expect(animation).toBe("none");
});

test("captures a screenshot for the README", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 900 });
  await ready(page);
  // Un-stick the header so it doesn't overlap content in the full-page capture.
  await page.addStyleTag({ content: ".masthead{position:static !important}" });
  // Wait for the CARTO basemap tiles and both maps' polygons to settle before capture.
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(2500);
  await page.screenshot({ path: "docs/screenshot.png", fullPage: true });
});
