import "@testing-library/jest-dom";

// Neither jsdom nor Node 26 gives us a working `localStorage` in this environment (Node's
// global is inert without --localstorage-file), and useTheme persists through it.
const store = new Map<string, string>();
globalThis.localStorage = {
  get length() {
    return store.size;
  },
  key: (i) => [...store.keys()][i] ?? null,
  getItem: (k) => store.get(k) ?? null,
  setItem: (k, v) => {
    store.set(k, String(v));
  },
  removeItem: (k) => {
    store.delete(k);
  },
  clear: () => store.clear(),
};
