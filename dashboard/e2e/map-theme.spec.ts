import { expect, test } from "@playwright/test";

test("a theme change cancels an obsolete initial map style", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  let releaseLight!: () => void;
  const lightGate = new Promise<void>((resolve) => {
    releaseLight = resolve;
  });
  let requestedLight!: () => void;
  const lightPending = new Promise<void>((resolve) => {
    requestedLight = resolve;
  });
  const style = (color: string) => ({
    version: 8,
    sources: {
      fixture: {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        attribution: "Map fixture attribution",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": color } },
      { id: "fixture", type: "fill", source: "fixture" },
    ],
  });
  await page.route("**/gl/positron-gl-style/style.json", async (route) => {
    requestedLight();
    await lightGate;
    await route.fulfill({ json: style("#ffffff") });
  });
  await page.route("**/gl/dark-matter-gl-style/style.json", (route) =>
    route.fulfill({ json: style("#000000") }),
  );
  await page.goto("/");
  await lightPending;
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  releaseLight();
  const canvas = page.locator("#map canvas");
  await expect(canvas).toBeAttached();
  const background = () =>
    canvas.evaluate((element: HTMLCanvasElement) => {
      const copy = document.createElement("canvas");
      copy.width = element.width;
      copy.height = element.height;
      const context = copy.getContext("2d");
      if (!context) throw new Error("2D canvas is unavailable");
      context.drawImage(element, 0, 0);
      return Array.from(context.getImageData(2, 2, 1, 1).data).slice(0, 3);
    });
  await expect.poll(background).toEqual([0, 0, 0]);
  await expect(page.locator("#map .maplibregl-ctrl-attrib-inner")).toContainText(
    "Map fixture attribution",
  );
  const initialCanvas = await canvas.elementHandle();
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect.poll(background).toEqual([255, 255, 255]);
  expect(await canvas.evaluate((element, original) => element === original, initialCanvas)).toBe(
    true,
  );
});
