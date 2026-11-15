import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import temporal from "../../public/data/temporal.json";
import { getPalette } from "../lib/palette";
import type { TemporalData } from "../lib/types";
import { TemporalView } from "./TemporalView";

vi.mock("@nivo/line", () => ({ ResponsiveLine: () => null }));

const heading = /When cooling is needed/;

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("keeps the temporal data available without ResizeObserver", () => {
  vi.stubGlobal("ResizeObserver", undefined);
  render(
    <TemporalView
      temporal={temporal as TemporalData}
      scenario="SQ"
      elec={{ residential: 1, office: 1 }}
      palette={getPalette("light")}
    />,
  );
  expect(screen.getByRole("heading", { name: heading })).toBeVisible();
  expect(screen.getAllByRole("table", { hidden: true })).toHaveLength(2);
});
