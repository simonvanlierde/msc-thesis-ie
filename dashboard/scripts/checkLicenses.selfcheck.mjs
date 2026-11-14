import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

for (const [license, succeeds] of [
  ["MIT", true],
  ["MIT OR GPL-2.0-only", true],
  ["MIT AND GPL-2.0-only", false],
  ["(MIT OR ISC) AND Apache-2.0", true],
  ["MIT WITH Classpath-exception-2.0", true],
  ["Unknown", false],
  ["MIT OR", false],
  ["(MIT", false],
  ["MIT WITH", false],
  ["", false],
]) {
  test(`checks ${license || "an empty expression"}`, () => {
    const result = spawnSync(process.execPath, ["scripts/checkLicenses.mjs"], {
      input: JSON.stringify({ [license]: [{ name: "sample", versions: ["1.0.0"] }] }),
      encoding: "utf8",
    });
    assert.equal(result.status, succeeds ? 0 : 1, result.stderr);
  });
}
