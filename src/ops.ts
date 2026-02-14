import { Value } from "./value";

export function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((row) =>
    row.reduce((sum, wi, i) => sum.add(wi.mul(x[i]!)), new Value(0)),
  );
}

export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((a, b) => a.add(b));
  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[]): Value[] {
  const ms = x
    .map((xi) => xi.mul(xi))
    .reduce((a, b) => a.add(b))
    .div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}
