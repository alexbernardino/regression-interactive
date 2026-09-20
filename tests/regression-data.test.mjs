import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import test from "node:test";

// Exercise the actual pure data functions used by the interface.
const source = readFileSync(new URL("../app/page.tsx", import.meta.url), "utf8");
const pure = source.slice(source.indexOf("type Experiment"), source.indexOf("function NumberField"));
const context = vm.createContext({});
vm.runInContext(ts.transpile(pure + "\nglobalThis.api = { createInitialLabState, updateLabState, evaluate };", { target: ts.ScriptTarget.ES2022 }), context);
const { createInitialLabState, updateLabState, evaluate } = context.api;
const snapshot = value => JSON.stringify(value);
const subset = (state, training) => state.result.points.filter(p => p.training === training);

test("ridge covariance shrinks at fixed data despite increasing fit residuals", () => {
  let state = createInitialLabState();
  const initial = state;
  const training = subset(initial, true);
  const n = training.length;
  const mean = training.reduce((s, p) => s + p.x, 0) / n;
  const sxx = training.reduce((s, p) => s + (p.x - mean) ** 2, 0);
  let previous = initial.result.covariance;
  for (const lambda of [0.1, 1, 10, 100, 1e6]) {
    state = updateLabState(state, { type: "set-value", key: "l2", value: lambda });
    assert.equal(snapshot(state.result.points), snapshot(initial.result.points));
    const c = state.result.covariance;
    const expected = initial.result.covariance.slopeVariance * (sxx / (sxx + n * lambda)) ** 2;
    assert.ok(Math.abs(c.slopeVariance - expected) < 1e-12);
    assert.ok(c.slopeVariance < previous.slopeVariance);
    assert.ok(c.interceptVariance >= 0);
    assert.equal(c.simulations, 100);
    assert.equal(c.noiseVariance, initial.config.noiseVariance);
    previous = c;
  }
  assert.ok(state.result.rmse > initial.result.rmse);
  assert.ok(previous.interceptVariance > 0);
});

test("L1 thresholding changes empirical covariance and simulation mean", () => {
  const initial = createInitialLabState();
  const after = updateLabState(initial, { type: "set-value", key: "l1", value: 1e6 });
  assert.equal(snapshot(initial.result.points), snapshot(after.result.points));
  assert.equal(after.result.covariance.meanSlope, 0);
  assert.equal(after.result.covariance.slopeVariance, 0);
  assert.equal(after.result.covariance.slopeIntercept, 0);
  assert.ok(after.result.covariance.interceptVariance > 0);
  assert.notEqual(initial.result.covariance.meanSlope, 0);
});

test("simulation is deterministic and its covariance is positive semidefinite", () => {
  const first = createInitialLabState();
  assert.equal(snapshot(first.result.covariance), snapshot(createInitialLabState().result.covariance));
  const c = first.result.covariance;
  assert.ok(c.slopeVariance * c.interceptVariance - c.slopeIntercept ** 2 >= -1e-12);
  assert.equal(c.simulations, 100);
});

test("fixed outliers and manual points add no simulated variability when normal noise is zero", () => {
  let state = updateLabState(createInitialLabState(), { type: "set-value", key: "noiseVariance", value: 0 });
  state = updateLabState(state, { type: "set-value", key: "outliers", value: 5 });
  state = updateLabState(state, { type: "add", x: 3, y: 100 });
  const c = state.result.covariance;
  assert.equal(c.slopeVariance, 0);
  assert.equal(c.interceptVariance, 0);
  assert.ok(Math.abs(c.meanSlope - state.result.fittedSlope) < 1e-12);
  assert.ok(Math.abs(c.meanIntercept - state.result.fittedIntercept) < 1e-12);
});

test("zero generating noise gives zero sampling covariance even with ridge bias", () => {
  let state = updateLabState(createInitialLabState(), { type: "set-value", key: "noiseVariance", value: 0 });
  state = updateLabState(state, { type: "set-value", key: "l2", value: 10 });
  assert.ok(state.result.rmse > 0);
  assert.equal(state.result.covariance.slopeVariance, 0);
  assert.equal(state.result.covariance.interceptVariance, 0);
});

test("resampling test preserves training, coefficients and covariance exactly", () => {
  const before = createInitialLabState();
  const after = updateLabState(before, { type: "resample", subset: "test" });
  assert.equal(snapshot(subset(before, true)), snapshot(subset(after, true)));
  assert.notEqual(snapshot(subset(before, false)), snapshot(subset(after, false)));
  for (const key of ["fittedSlope", "fittedIntercept", "rmse", "r2", "covariance"]) {
    assert.equal(snapshot(before.result[key]), snapshot(after.result[key]));
  }
});

test("resampling training preserves the test observations", () => {
  const before = createInitialLabState();
  const after = updateLabState(before, { type: "resample", subset: "training" });
  assert.equal(snapshot(subset(before, false)), snapshot(subset(after, false)));
  assert.notEqual(snapshot(subset(before, true)), snapshot(subset(after, true)));
});

test("manual points affect the fit, survive resampling, and can be removed", () => {
  const original = createInitialLabState();
  const added = updateLabState(original, { type: "add", x: 4, y: 100 });
  assert.notEqual(added.result.fittedSlope, original.result.fittedSlope);
  const resampled = updateLabState(added, { type: "resample" });
  assert.equal(resampled.result.points.find(p => p.manual).y, 100);
  const restored = updateLabState(added, { type: "remove", id: "manual-1" });
  assert.equal(restored.result.fittedSlope, original.result.fittedSlope);
});

test("removal persists through regularization and cannot empty training", () => {
  let state = createInitialLabState();
  const id = subset(state, true)[0].id;
  state = updateLabState(state, { type: "remove", id });
  state = updateLabState(state, { type: "set-value", key: "l2", value: 1 });
  assert.ok(!state.result.points.some(p => p.id === id));
  for (const point of subset(state, true)) state = updateLabState(state, { type: "remove", id: point.id });
  assert.equal(subset(state, true).length, 2);
  assert.ok(Number.isFinite(state.result.fittedSlope));
});

test("evaluation uses each set's own mean and reports undefined R² honestly", () => {
  const metrics = evaluate([{ x: 0, y: 0 }, { x: 1, y: 2 }], 1, 0);
  assert.equal(metrics.rmse, Math.sqrt(.5));
  assert.equal(metrics.r2, .5);
  assert.ok(Number.isNaN(evaluate([{ x: 1, y: 2 }], 1, 0).r2));
});

test("outlier test fraction supports both endpoints without changing normal points", () => {
  const initial = createInitialLabState();
  for (const fraction of [0, 40, 100]) {
    const state = updateLabState(initial, { type: "set-value", key: "outlierTestFraction", value: fraction });
    assert.equal(snapshot(state.result.points.filter(p => !p.outlier)), snapshot(initial.result.points.filter(p => !p.outlier)));
    assert.equal(state.result.points.filter(p => p.outlier && !p.training).length, Math.round(initial.config.outliers * fraction / 100));
    assert.equal(state.result.points.filter(p => p.outlier && p.training).length, initial.config.outliers - Math.round(initial.config.outliers * fraction / 100));
  }
});
