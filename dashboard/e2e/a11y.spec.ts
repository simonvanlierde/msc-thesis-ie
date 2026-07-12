import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const LCA_HEADING = /life-cycle climate impact/i;
const FORK_HEADING = /choose the path to 2050/i;

// The dashboard is one page; wait until the async data has rendered all three views.
async function ready(page: import("@playwright/test").Page) {
  await page.goto("/");
  // Scroll acts reveal on intersection; force them present so axe scans every act and the
  // full-page screenshot isn't blank below the fold.
  await page.addStyleTag({ content: ".reveal{opacity:1 !important;transform:none !important}" });
  await expect(page.getByRole("heading", { name: LCA_HEADING })).toBeVisible({
    timeout: 20_000,
  });
  // open the disclosure tables so axe also scans them
  await page.locator("details.datatable").evaluateAll((tables) => {
    for (const t of tables) t.setAttribute("open", "");
  });
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
  await expect(page.getByRole("heading", { name: FORK_HEADING })).toBeVisible({
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
  await page.setViewportSize({ width: 1440, height: 1000 });
  await ready(page);
  // Un-stick the header so it doesn't overlap content in the capture.
  await page.addStyleTag({ content: ".masthead{position:static !important}" });
  // Wait for the CARTO basemap tiles and both maps' polygons to settle before capture.
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(2500);

  // ready() force-opens the disclosure tables so axe can scan them. The figure encloses its
  // table, so leaving it open would drag 15 rows into the crop — collapse them again.
  await page.locator("details.datatable").evaluateAll((tables) => {
    for (const t of tables) t.removeAttribute("open");
  });

  // The map view, not the whole page: a full-page capture is a ~9,000px ribbon that renders in
  // the README as a strip nobody scrolls. Clip in document coordinates (hence fullPage) — a
  // viewport clip would depend on where the page happened to be scrolled, which is not stable.
  const clip = await page.evaluate(() => {
    const PAD = 24;
    const section = document.querySelector("#map")?.getBoundingClientRect();
    const heading = document.querySelector("#map-h")?.getBoundingClientRect();
    const figure = document.querySelector("#map .figure")?.getBoundingClientRect();
    if (!(section && heading && figure)) throw new Error("map section not found");
    // Top edge on the heading, not the section: the section's padding reaches up into the
    // sticky scenario switcher above it, which would bleed a sliver into the crop.
    const top = heading.top + window.scrollY;
    const bottom = figure.bottom + window.scrollY;
    return {
      x: Math.round(section.left - PAD),
      y: Math.round(top - PAD),
      width: Math.round(section.width + PAD * 2),
      height: Math.round(bottom - top + PAD * 2),
    };
  });
  await page.screenshot({ path: "docs/screenshot.png", fullPage: true, clip });
});
