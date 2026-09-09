import "@testing-library/jest-dom/vitest";

// `localStorage` is unreliable across the environments this suite runs in: Node's own global
// is inert without --localstorage-file, and jsdom exposes the property as a getter-only
// accessor, so a plain assignment throws. useTheme persists through it, so install a working
// one with defineProperty, which overwrites either shape.
const store = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    get length() {
      return store.size;
    },
    key: (i: number) => [...store.keys()][i] ?? null,
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => {
      store.set(k, String(v));
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    clear: () => store.clear(),
  } satisfies Storage,
});
