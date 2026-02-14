import { randn } from "./utils";
import { Value } from "./value";

export function matrix(nout: number, nin: number, std = 0.02) {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(randn(0, std))),
  );
}
