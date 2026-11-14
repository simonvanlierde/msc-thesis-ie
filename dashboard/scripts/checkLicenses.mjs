// Lists the license of each production dependency and fails on one this project's
// GPL-3.0-or-later license cannot take. Reads `pnpm licenses list --prod --json` on stdin.
import process from "node:process";
import { text } from "node:stream/consumers";

// SPDX ids, matched without regard to case. Widen it only for a license that GPL-3.0-or-later
// can take; for one package, ask its authors or replace it instead.
const ALLOWED = new Set(
  [
    "0BSD",
    "Apache-2.0",
    "BlueOak-1.0.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "CDLA-Permissive-2.0",
    "ISC",
    "MIT",
    "MIT-0",
    "MPL-2.0",
    "PSF-2.0",
    "Python-2.0",
    "Unicode-3.0",
    "Unlicense",
    "Zlib",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
  ].map((id) => id.toLowerCase()),
);

// Whether an SPDX expression is allowed: an OR needs one side allowed, an AND both. AND
// binds tighter than OR, and parentheses group. An exception (`WITH`) only adds
// permissions, so the license it qualifies decides. A malformed expression is refused.
const allowed = (expression) => {
  const tokens = expression.match(/[()]|[^\s()]+/g) ?? [];
  let at = 0;
  const peek = () => tokens[at]?.toUpperCase();
  const malformed = () => new Error("Malformed SPDX expression");
  const term = () => {
    const token = tokens[at++];
    if (token === "(") {
      const ok = anyOf();
      if (tokens[at++] !== ")") throw malformed();
      return ok;
    }
    if (
      token === undefined ||
      token === ")" ||
      ["AND", "OR", "WITH"].includes(token.toUpperCase())
    ) {
      throw malformed();
    }
    if (peek() === "WITH") {
      at++;
      const exception = tokens[at++];
      if (
        exception === undefined ||
        exception === "(" ||
        exception === ")" ||
        ["AND", "OR", "WITH"].includes(exception.toUpperCase())
      ) {
        throw malformed();
      }
    }
    return ALLOWED.has(token.toLowerCase());
  };
  const allOf = () => {
    let ok = term();
    while (peek() === "AND") {
      at++;
      // term() runs first, so it consumes its tokens even when ok is already false.
      ok = term() && ok;
    }
    return ok;
  };
  const anyOf = () => {
    let ok = allOf();
    while (peek() === "OR") {
      at++;
      ok = allOf() || ok;
    }
    return ok;
  };
  try {
    const ok = anyOf();
    return ok && at === tokens.length;
  } catch {
    return false;
  }
};

const byLicense = JSON.parse(await text(process.stdin));
let failed = false;
for (const [license, packages] of Object.entries(byLicense)) {
  const names = packages.map((p) => `${p.name}@${p.versions.join(",")}`).join(", ");
  const ok = allowed(license);
  failed ||= !ok;
  process.stdout.write(`${ok ? "ok" : "NOT ALLOWED"}  ${license}: ${names}\n`);
}
if (failed) {
  process.stderr.write(
    "A dependency's license is not in the allow-list in scripts/checkLicenses.mjs.\n",
  );
  process.exit(1);
}
