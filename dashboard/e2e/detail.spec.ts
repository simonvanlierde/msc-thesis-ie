import { expect, test } from "@playwright/test";
import { elecFactors } from "../src/lib/transform";

const SHOWN_MONTHS = /^(Jan|Mar|May|Jul|Sep|Nov)$/;
const OMITTED_MONTHS = /^(Feb|Apr|Jun|Aug|Oct|Dec)$/;
const DAY_TICK = /^\d+ [A-Z][a-z]{2}$/;
const PEAK_DESCRIPTION = /peak: .* GW at/;

test("all plotted data has a table that follows the selected quantity", async ({ page }) => {
  await page.goto("/");
  const buurten = await (await page.request.get("/data/cooling_by_buurt.geojson")).json();
  const temporal = await (await page.request.get("/data/temporal.json")).json();
  const scenarios = await (await page.request.get("/data/scenarios.json")).json();
  await expect(page.locator("#map table tbody tr")).toHaveCount(buurten.features.length);
  const hourly = page.locator("#when details").filter({ hasText: "Data table: hourly" });
  await expect(hourly.locator("tbody tr")).toHaveCount(
    temporal.by_scenario["2050_M"].heatwave.dates.length,
  );
  await hourly.locator("summary").click();
  const firstValue = hourly
    .locator("tbody tr")
    .first()
    .locator("td")
    .nth(temporal.uses.indexOf("residential") + 1);
  const cooling = temporal.by_scenario["2050_M"].heatwave.series.residential[0];
  const text = (n: number) => n.toLocaleString("en-GB", { maximumFractionDigits: 3 });
  await expect(firstValue).toHaveText(text(cooling));
  await page.locator("#when").getByRole("radio", { name: "Electricity use" }).check();
  await expect(hourly.locator("caption")).toContainText("Electric power");
  await expect(firstValue).toHaveText(
    text(cooling * elecFactors(scenarios.scenarios["2050_M"].archetypes).residential),
  );
});

test("320px layout keeps tables locally scrollable", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto("/");
  await expect(page.locator("#when table").first()).toBeAttached();
  await page.locator("details.datatable").evaluateAll((tables) => {
    for (const table of tables) table.setAttribute("open", "");
  });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(320);
  const scrollArea = page.getByRole("region", { name: "Scenario totals" });
  await scrollArea.focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => scrollArea.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
});

test("optional map data can recover without a page reload", async ({ page }) => {
  await page.route("**/data/cooling_by_buurt.geojson", (route) => route.abort());
  await page.goto("/");
  await expect(page.locator("#when")).toBeAttached();
  await page.unroute("**/data/cooling_by_buurt.geojson");
  await page.locator("#map").getByRole("button", { name: "Try again" }).click();
  await expect(page.locator("#map [data-map-idle]")).toBeAttached({ timeout: 30_000 });
  await expect(page.locator("#map canvas")).toBeAttached();
  await expect(page.locator("#map table tbody tr")).toHaveCount(112);
});

test("phone charts keep month and day ticks apart", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const monthly = page.locator("#when .chart").first();
  await expect(monthly.locator("svg")).toBeAttached();
  await expect(monthly.locator("text").filter({ hasText: SHOWN_MONTHS })).toHaveCount(6);
  await expect(monthly.locator("text").filter({ hasText: OMITTED_MONTHS })).toHaveCount(0);
  await expect(
    page.locator("#when .chart").nth(1).locator("text").filter({ hasText: DAY_TICK }),
  ).toHaveCount(4);
  await expect(page.locator("#when .figure").nth(1).locator("figcaption")).toContainText(
    PEAK_DESCRIPTION,
  );
});

test("map table retains neighbourhoods without a scenario value", async ({ page }) => {
  const response = await page.request.get("/data/cooling_by_buurt.geojson");
  const data = await response.json();
  const missing = data.features[0].properties;
  missing["2050_M__E_cooling_kWh"] = undefined;
  await page.route("**/data/cooling_by_buurt.geojson", (route) => route.fulfill({ json: data }));
  await page.goto("/");
  const table = page.locator("#map details.datatable");
  await table.locator("summary").click();
  await expect(table.locator("tbody tr").filter({ hasText: missing.buurtnaam })).toContainText(
    "No data",
  );
  await expect(table.locator("tbody tr")).toHaveCount(data.features.length);
});
