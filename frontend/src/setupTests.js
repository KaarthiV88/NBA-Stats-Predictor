// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// jsdom ships an AbortSignal without the static `timeout()` helper that the app
// uses to bound its fetches. Every browser the app targets has supported it
// since 2022, so this fills the gap in the test environment only.
if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout !== 'function') {
  AbortSignal.timeout = (ms) => {
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(new DOMException('TimeoutError', 'TimeoutError')),
      ms
    );
    // Don't keep the Jest process alive waiting on these timers.
    if (typeof timer === 'object' && timer !== null && 'unref' in timer) {
      timer.unref();
    }
    return controller.signal;
  };
}
