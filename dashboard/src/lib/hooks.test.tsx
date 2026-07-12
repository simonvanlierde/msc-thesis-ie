import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useInView } from "./useInView";
import { useScrollSpy } from "./useScrollSpy";
import { useTheme } from "./useTheme";

type Cb = (entries: { target: Element; isIntersecting: boolean }[]) => void;

/** Records observers so a test can fire intersections by hand. jsdom has no real one. */
function stubIntersectionObserver() {
  const observers: { cb: Cb; disconnect: () => void }[] = [];
  vi.stubGlobal(
    "IntersectionObserver",
    class {
      disconnect = vi.fn();
      observe = vi.fn();
      unobserve = vi.fn();
      constructor(public cb: Cb) {
        observers.push(this as never);
      }
    },
  );
  return observers;
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
  document.documentElement.removeAttribute("data-theme");
  localStorage.clear();
});

describe("useTheme", () => {
  it("reads the mode index.html stamped, persists a toggle", () => {
    document.documentElement.dataset.theme = "dark";
    const { result } = renderHook(() => useTheme());
    expect(result.current[0]).toBe("dark");

    act(() => result.current[1]());
    expect(result.current[0]).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("cooling-dashboard.theme")).toBe("light");
  });

  it("falls back to the OS preference when nothing is stamped", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: true }));
    expect(renderHook(() => useTheme()).result.current[0]).toBe("dark");
  });
});

describe("useInView", () => {
  function Reveal() {
    const [ref, inView] = useInView<HTMLDivElement>();
    return (
      <div ref={ref} data-testid="el">
        {String(inView)}
      </div>
    );
  }

  it("flips once on entry, then stops observing", () => {
    const observers = stubIntersectionObserver();
    render(<Reveal />);
    const el = screen.getByTestId("el");
    expect(el).toHaveTextContent("false");

    act(() => observers[0].cb([{ target: el, isIntersecting: true }]));
    expect(el).toHaveTextContent("true");
    expect(observers[0].disconnect).toHaveBeenCalled();
  });

  it("is visible-always without IntersectionObserver", () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    render(<Reveal />);
    expect(screen.getByTestId("el")).toHaveTextContent("true");
  });
});

describe("useScrollSpy", () => {
  it("activates the topmost visible section", () => {
    const observers = stubIntersectionObserver();
    document.body.innerHTML = `<div id="a"></div><div id="b"></div>`;
    const { result } = renderHook(() => useScrollSpy(["a", "b"]));
    expect(result.current).toBe("a");

    const b = document.getElementById("b") as HTMLElement;
    act(() => observers[0].cb([{ target: b, isIntersecting: true }]));
    expect(result.current).toBe("b");

    // "a" back in the band wins over "b": document order, not firing order.
    const a = document.getElementById("a") as HTMLElement;
    act(() => observers[0].cb([{ target: a, isIntersecting: true }]));
    expect(result.current).toBe("a");
  });

  it("stays on the first id while disabled", () => {
    stubIntersectionObserver();
    expect(renderHook(() => useScrollSpy(["a", "b"], false)).result.current).toBe("a");
  });
});
