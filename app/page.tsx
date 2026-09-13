"use client";

import { useId, useMemo, useReducer, useState, type ReactNode } from "react";

type Experiment = {
  xMin: number;
  xMax: number;
  slope: number;
  intercept: number;
  samples: number;
  noiseVariance: number;
  outlierMin: number;
  outlierMax: number;
  outliers: number;
  l1: number;
  l2: number;
  testFraction: number;
};

type Point = {
  x: number;
  y: number;
  outlier: boolean;
  training: boolean;
  id: string;
  manual?: boolean;
};

type Result = {
  config: Experiment;
  points: Point[];
  fittedSlope: number;
  fittedIntercept: number;
  rmse: number;
  r2: number;
  testRmse: number;
  testR2: number;
  covariance: {
    slopeVariance: number;
    interceptVariance: number;
    slopeIntercept: number;
    residualVariance: number;
  };
  run: number;
};

type DataAxes = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

type ParameterAxes = {
  slopeMin: number;
  slopeMax: number;
  interceptMin: number;
  interceptMax: number;
};

const DEFAULTS: Experiment = {
  xMin: -5,
  xMax: 5,
  slope: 1.8,
  intercept: 2,
  samples: 40,
  noiseVariance: 2,
  outlierMin: -14,
  outlierMax: 18,
  outliers: 5,
  l1: 0,
  l2: 0,
  testFraction: 30,
};

const round = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";

function seededRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(random: () => number) {
  let u = 0;
  let v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function evaluate(points: Point[], slope: number, intercept: number) {
  if (!points.length) return { rmse: NaN, r2: NaN };
  const mean = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  const error = points.reduce((sum, point) => sum + (point.y - slope * point.x - intercept) ** 2, 0);
  const spread = points.reduce((sum, point) => sum + (point.y - mean) ** 2, 0);
  return { rmse: Math.sqrt(error / points.length), r2: points.length < 2 || spread === 0 ? NaN : 1 - error / spread };
}

function runRegression(config: Experiment, run: number, testRun = run, manual: Point[] = [], removed: string[] = []): Result {
  const points: Point[] = [];
  const xSpan = config.xMax - config.xMin;
  const noiseStdDev = Math.sqrt(config.noiseVariance);
  for (const training of [true, false]) {
  const testCount = Math.round(config.samples * config.testFraction / 100);
  const count = training ? config.samples - testCount : testCount;
  const random = seededRandom((training ? 8147 : 95131) + (training ? run : testRun) * 104729);
  for (let index = 0; index < count; index += 1) {
    const x = config.xMin + random() * xSpan;
    const y =
      config.slope * x +
      config.intercept +
      gaussianRandom(random) * noiseStdDev;
    points.push({ x, y, outlier: false, training, id: `${training ? "train" : "test"}-sample-${index}` });
  }

  const outlierRandom = seededRandom((training ? 37139 : 192811) + (training ? run : testRun) * 7919);
  const testOutliers = Math.round(config.outliers * config.testFraction / 100);
  for (let index = 0; index < (training ? config.outliers - testOutliers : testOutliers); index += 1) {
    const x = config.xMin + outlierRandom() * xSpan;
    const y =
      config.outlierMin +
      outlierRandom() * (config.outlierMax - config.outlierMin);
    points.push({ x, y, outlier: true, training, id: `${training ? "train" : "test"}-outlier-${index}` });
  }
  }
  const allPoints = [...points.filter(point => !removed.includes(point.id)), ...manual];
  points.splice(0, points.length, ...allPoints.filter(point => point.training));

  const xMean =
    points.reduce((total, point) => total + point.x, 0) / points.length;
  const yMean =
    points.reduce((total, point) => total + point.y, 0) / points.length;

  const numerator = points.reduce(
    (total, point) => total + (point.x - xMean) * (point.y - yMean),
    0,
  );
  const denominator = points.reduce(
    (total, point) => total + (point.x - xMean) ** 2,
    0,
  );

  const normalizedCrossProduct = numerator / points.length;
  const normalizedSquaredX = denominator / points.length;
  const softThresholdedSlope =
    Math.sign(normalizedCrossProduct) *
    Math.max(Math.abs(normalizedCrossProduct) - config.l1, 0);
  const fittedSlope =
    normalizedSquaredX + config.l2 > 0 ? softThresholdedSlope / (normalizedSquaredX + config.l2) : 0;
  const fittedIntercept = yMean - fittedSlope * xMean;
  const residualSum = points.reduce((total, point) => {
    const prediction = fittedSlope * point.x + fittedIntercept;
    return total + (point.y - prediction) ** 2;
  }, 0);
  const totalSum = points.reduce(
    (total, point) => total + (point.y - yMean) ** 2,
    0,
  );
  const residualVariance = residualSum / Math.max(points.length - 2, 1);
  const regularizedDenominator = denominator + points.length * config.l2;
  const slopeVariance =
    regularizedDenominator > 0 ? (residualVariance * denominator) / regularizedDenominator ** 2 : 0;
  const interceptVariance =
    residualVariance / points.length + xMean ** 2 * slopeVariance;
  const slopeIntercept = -xMean * slopeVariance;

  return {
    config: { ...config },
    points: allPoints,
    fittedSlope,
    fittedIntercept,
    rmse: Math.sqrt(residualSum / points.length),
    r2: totalSum === 0 ? NaN : 1 - residualSum / totalSum,
    testRmse: evaluate(allPoints.filter(point => !point.training), fittedSlope, fittedIntercept).rmse,
    testR2: evaluate(allPoints.filter(point => !point.training), fittedSlope, fittedIntercept).r2,
    covariance: {
      slopeVariance,
      interceptVariance,
      slopeIntercept,
      residualVariance,
    },
    run,
  };
}

function validateExperiment(config: Experiment) {
  if (!Object.values(config).every(Number.isFinite)) return "Enter finite values for every parameter.";
  if (config.testFraction < 10 || config.testFraction > 50) return "Choose a test fraction between 10% and 50%.";
  if (config.xMax <= config.xMin) {
    return "The maximum x value must be greater than the minimum.";
  }
  if (config.outlierMax <= config.outlierMin && config.outliers > 0) {
    return "The outlier maximum must be greater than its minimum.";
  }
  if (config.samples < 4 || config.samples > 500) {
    return "Choose between 4 and 500 regular samples.";
  }
  if (config.outliers < 0 || config.outliers > 100) {
    return "Choose between 0 and 100 outliers.";
  }
  if (config.noiseVariance < 0) {
    return "Noise variance cannot be negative.";
  }
  if (config.l1 < 0 || config.l2 < 0) {
    return "Regularization strengths cannot be negative.";
  }
  return "";
}

type LabState = {
  config: Experiment;
  run: number;
  result: Result;
  testRun: number;
  manual: Point[];
  removed: string[];
  nextId: number;
};

type LabAction =
  | { type: "set-value"; key: keyof Experiment; value: number }
  | { type: "resample"; subset?: "training" | "test" }
  | { type: "add"; x: number; y: number }
  | { type: "remove"; id: string };

function createInitialLabState(): LabState {
  return {
    config: DEFAULTS,
    run: 1,
    testRun: 1,
    manual: [],
    removed: [],
    nextId: 1,
    result: runRegression(DEFAULTS, 1),
  };
}

function updateLabState(state: LabState, action: LabAction): LabState {
  if (action.type === "add" || action.type === "remove") {
    if (action.type === "add" && (!Number.isFinite(action.x) || !Number.isFinite(action.y))) return state;
    if (action.type === "remove" && !state.result.points.some(p => p.id === action.id && p.training)) return state;
    if (action.type === "remove" && state.result.points.filter(p => p.training).length <= 2) return state;
    const manual = action.type === "add"
      ? [...state.manual, { x: action.x, y: action.y, training: true, outlier: false, manual: true, id: `manual-${state.nextId}` }]
      : state.manual.filter(p => p.id !== action.id);
    const removed = action.type === "remove" ? [...state.removed, action.id] : state.removed;
    return { ...state, manual, removed, nextId: state.nextId + 1,
      result: runRegression(state.result.config, state.run, state.testRun, manual, removed) };
  }
  if (action.type === "resample") {
    if (validateExperiment(state.config)) return state;
    const run = state.run + (action.subset === "test" ? 0 : 1);
    const testRun = state.testRun + (action.subset === "training" ? 0 : 1);
    const removed = action.subset === "test" ? state.removed : [];
    return {
      ...state,
      run,
      testRun,
      removed,
      result: runRegression(state.config, run, testRun, state.manual, removed),
    };
  }

  const config = { ...state.config, [action.key]: action.value };
  const removed = ["samples", "outliers", "testFraction"].includes(action.key) ? [] : state.removed;
  return {
    ...state,
    config,
    removed,
    result: validateExperiment(config)
      ? state.result
      : runRegression(config, state.run, state.testRun, state.manual, removed),
  };
}

function NumberField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
}) {
  return (
    <label className="number-field">
      <span>
        {label}
        {hint ? <small>{hint}</small> : null}
      </span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  hint?: string;
}) {
  const digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;

  return (
    <label className="slider-field">
      <span className="slider-heading">
        <span>
          {label}
          {hint ? <small>{hint}</small> : null}
        </span>
        <output>{round(value, digits)}</output>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        aria-label={label}
        aria-valuetext={round(value, digits)}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="slider-scale" aria-hidden="true">
        <span>{min}</span>
        <span>{max}</span>
      </span>
    </label>
  );
}

function ControlSection({
  number,
  title,
  id,
  open,
  onOpen,
  children,
}: {
  number: string;
  title: string;
  id: string;
  open: boolean;
  onOpen: () => void;
  children: ReactNode;
}) {
  const heading = (
    <>
      <span>{number}</span> {title}
    </>
  );

  return (
    <section className={`control-section${open ? " is-open" : ""}`}>
      <div className="desktop-section-heading">{heading}</div>
      <button
        className="mobile-section-toggle"
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={onOpen}
      >
        {heading}
      </button>
      <div id={id} className="control-section-body">
        {children}
      </div>
    </section>
  );
}

function fitDataAxes(result: Result): DataAxes {
  const xCandidates = [
    result.config.xMin,
    result.config.xMax,
    ...result.points.map((point) => point.x),
  ];
  const xMin = Math.floor(Math.min(...xCandidates));
  const xMax = Math.ceil(Math.max(...xCandidates));
  const safeXMin = xMin === xMax ? xMin - 1 : xMin;
  const safeXMax = xMin === xMax ? xMax + 1 : xMax;
  const yCandidates = [
    ...result.points.map((point) => point.y),
    result.config.slope * safeXMin + result.config.intercept,
    result.config.slope * safeXMax + result.config.intercept,
    result.fittedSlope * safeXMin + result.fittedIntercept,
    result.fittedSlope * safeXMax + result.fittedIntercept,
  ];
  const rawYMin = Math.min(...yCandidates);
  const rawYMax = Math.max(...yCandidates);
  const yPadding = Math.max((rawYMax - rawYMin) * 0.12, 1);

  return {
    xMin: safeXMin,
    xMax: safeXMax,
    yMin: Math.floor(rawYMin - yPadding),
    yMax: Math.ceil(rawYMax + yPadding),
  };
}

function fitParameterAxes(result: Result): ParameterAxes {
  const slopeRadius95 = Math.sqrt(
    Math.max(5.991 * result.covariance.slopeVariance, 0),
  );
  const interceptRadius95 = Math.sqrt(
    Math.max(5.991 * result.covariance.interceptVariance, 0),
  );
  const rawSlopeMin = Math.min(
    result.config.slope,
    result.fittedSlope - slopeRadius95,
  );
  const rawSlopeMax = Math.max(
    result.config.slope,
    result.fittedSlope + slopeRadius95,
  );
  const rawInterceptMin = Math.min(
    result.config.intercept,
    result.fittedIntercept - interceptRadius95,
  );
  const rawInterceptMax = Math.max(
    result.config.intercept,
    result.fittedIntercept + interceptRadius95,
  );
  const commonSpan =
    Math.max(rawSlopeMax - rawSlopeMin, rawInterceptMax - rawInterceptMin, 1) *
    1.18;
  const slopeCenter = (rawSlopeMin + rawSlopeMax) / 2;
  const interceptCenter = (rawInterceptMin + rawInterceptMax) / 2;

  return {
    slopeMin: slopeCenter - commonSpan / 2,
    slopeMax: slopeCenter + commonSpan / 2,
    interceptMin: interceptCenter - commonSpan / 2,
    interceptMax: interceptCenter + commonSpan / 2,
  };
}

function RegressionPlot({
  result,
  axes,
  query,
  mode,
  onPlace,
  onRemove,
}: {
  result: Result;
  axes: DataAxes;
  query: { x: number; y: number } | null;
  mode: "query" | "add" | "remove";
  onPlace: (x: number, y: number) => void;
  onRemove: (id: string) => void;
}) {
  const plotId = useId();
  const width = 920;
  const height = 520;
  const margin = { top: 30, right: 42, bottom: 62, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const { xMin, xMax, yMin, yMax } = axes;
  const config = result.config;

  const scaleX = (x: number) =>
    margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const scaleY = (y: number) =>
    margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;
  const xTickStart = Math.ceil(xMin);
  const xTickEnd = Math.floor(xMax);
  const yTickStart = Math.ceil(yMin);
  const yTickEnd = Math.floor(yMax);
  const xTicks = Array.from(
    { length: Math.max(xTickEnd - xTickStart + 1, 0) },
    (_, index) => xTickStart + index,
  );
  const yTicks = Array.from(
    { length: Math.max(yTickEnd - yTickStart + 1, 0) },
    (_, index) => yTickStart + index,
  );
  const labelInterval = (tickCount: number) =>
    tickCount <= 14 ? 1 : tickCount <= 30 ? 2 : tickCount <= 60 ? 5 : 10;
  const xLabelInterval = labelInterval(xTicks.length);
  const yLabelInterval = labelInterval(yTicks.length);

  const trueStart = config.slope * xMin + config.intercept;
  const trueEnd = config.slope * xMax + config.intercept;
  const fitStart =
    result.fittedSlope * xMin + result.fittedIntercept;
  const fitEnd = result.fittedSlope * xMax + result.fittedIntercept;

  return (
    <svg
      className="regression-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Regression data: solid training points, outlined test points, and query residual"
      onClick={(event) => {
        const svg = event.currentTarget;
        const matrix = svg.getScreenCTM();
        if (!matrix) return;
        const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
        if (local.x < margin.left || local.x > margin.left + plotWidth || local.y < margin.top || local.y > margin.top + plotHeight) return;
        if (mode === "remove") {
          const nearest = result.points.filter(point => point.training).map(point => {
            const screen = new DOMPoint(scaleX(point.x), scaleY(point.y)).matrixTransform(matrix);
            return { point, distance: Math.hypot(screen.x - event.clientX, screen.y - event.clientY) };
          }).sort((a, b) => a.distance - b.distance)[0];
          if (nearest && nearest.distance <= 14) onRemove(nearest.point.id);
          return;
        }
        onPlace(xMin + (local.x - margin.left) / plotWidth * (xMax - xMin), yMax - (local.y - margin.top) / plotHeight * (yMax - yMin));
      }}
    >
      <title>Linear regression experiment</title>
      <desc>
        A scatter plot showing regular samples, outliers, the ground-truth line,
        and the ordinary least-squares regression line.
      </desc>
      <defs>
        <clipPath id={plotId}>
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            rx="8"
          />
        </clipPath>
      </defs>

      <rect
        className="plot-background"
        x={margin.left}
        y={margin.top}
        width={plotWidth}
        height={plotHeight}
        rx="8"
      />

      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line
            className={`grid-line ${tick === 0 ? "zero-line" : ""}`}
            x1={scaleX(tick)}
            y1={margin.top}
            x2={scaleX(tick)}
            y2={margin.top + plotHeight}
          />
          {tick % xLabelInterval === 0 ? (
            <text
              className="tick-label"
              x={scaleX(tick)}
              y={margin.top + plotHeight + 29}
              textAnchor="middle"
            >
              {tick}
            </text>
          ) : null}
        </g>
      ))}

      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line
            className={`grid-line ${tick === 0 ? "zero-line" : ""}`}
            x1={margin.left}
            y1={scaleY(tick)}
            x2={margin.left + plotWidth}
            y2={scaleY(tick)}
          />
          {tick % yLabelInterval === 0 ? (
            <text
              className="tick-label"
              x={margin.left - 17}
              y={scaleY(tick) + 5}
              textAnchor="end"
            >
              {tick}
            </text>
          ) : null}
        </g>
      ))}

      <g clipPath={`url(#${plotId})`}>
        <line
          className="truth-line"
          x1={scaleX(xMin)}
          y1={scaleY(trueStart)}
          x2={scaleX(xMax)}
          y2={scaleY(trueEnd)}
        />
        <line
          className="fit-line"
          x1={scaleX(xMin)}
          y1={scaleY(fitStart)}
          x2={scaleX(xMax)}
          y2={scaleY(fitEnd)}
        />

        {result.points
          .filter((point) => !point.outlier)
          .map((point, index) => (
            <circle
              className={`sample-point ${point.training ? "training" : "test-point"}`}
              style={!point.training ? { fill: "var(--panel)", stroke: "var(--teal)", strokeWidth: 2.5 } : undefined}
              onClick={(event) => { if (mode === "remove" && point.training) { event.stopPropagation(); onRemove(point.id); } }}
              key={`sample-${index}`}
              cx={scaleX(point.x)}
              cy={scaleY(point.y)}
              r="5"
            />
          ))}

        {result.points
          .filter((point) => point.outlier)
          .map((point, index) => (
            <path
              className={`outlier-point ${point.training ? "training" : "test-point"}`}
              style={!point.training ? { fill: "var(--panel)", stroke: "var(--magenta)", strokeWidth: 2.5 } : undefined}
              onClick={(event) => { if (mode === "remove" && point.training) { event.stopPropagation(); onRemove(point.id); } }}
              key={`outlier-${index}`}
              d={`M ${scaleX(point.x)} ${scaleY(point.y) - 8} L ${
                scaleX(point.x) + 8
              } ${scaleY(point.y)} L ${scaleX(point.x)} ${
                scaleY(point.y) + 8
              } L ${scaleX(point.x) - 8} ${scaleY(point.y)} Z`}
            />
          ))}
        {result.points.filter(p => p.manual).map(p => <circle key={p.id} cx={scaleX(p.x)} cy={scaleY(p.y)} r="2" fill="black" pointerEvents="none" />)}
        {query && <g pointerEvents="none">
          <line x1={scaleX(query.x)} x2={scaleX(query.x)} y1={scaleY(query.y)} y2={scaleY(result.fittedSlope * query.x + result.fittedIntercept)} stroke="#172033" strokeWidth="3" strokeDasharray="6 4" />
          <circle cx={scaleX(query.x)} cy={scaleY(query.y)} r="8" fill="#d7ef49" stroke="#172033" strokeWidth="3" />
          <circle cx={scaleX(query.x)} cy={scaleY(result.fittedSlope * query.x + result.fittedIntercept)} r="5" fill="#172033" />
        </g>}
      </g>

      <text
        className="axis-label"
        x={margin.left + plotWidth / 2}
        y={height - 12}
        textAnchor="middle"
      >
        input x
      </text>
      <text
        className="axis-label"
        transform={`translate(20 ${margin.top + plotHeight / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        observed y
      </text>
    </svg>
  );
}

function ParameterPlot({
  result,
  axes,
}: {
  result: Result;
  axes: ParameterAxes;
}) {
  const width = 480;
  const height = 480;
  const margin = { top: 24, right: 24, bottom: 56, left: 56 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const clipId = useId();
  const {
    slopeVariance,
    interceptVariance,
    slopeIntercept,
  } = result.covariance;

  const trace = slopeVariance + interceptVariance;
  const difference = slopeVariance - interceptVariance;
  const discriminant = Math.sqrt(
    Math.max(difference ** 2 + 4 * slopeIntercept ** 2, 0),
  );
  const eigenvalue1 = Math.max((trace + discriminant) / 2, 0);
  const eigenvalue2 = Math.max((trace - discriminant) / 2, 0);
  const angle = 0.5 * Math.atan2(2 * slopeIntercept, difference);

  const ellipse = (chiSquareRadius: number) =>
    Array.from({ length: 73 }, (_, index) => {
      const theta = (index / 72) * Math.PI * 2;
      const major =
        chiSquareRadius * Math.sqrt(eigenvalue1) * Math.cos(theta);
      const minor =
        chiSquareRadius * Math.sqrt(eigenvalue2) * Math.sin(theta);
      return {
        slope:
          result.fittedSlope +
          major * Math.cos(angle) -
          minor * Math.sin(angle),
        intercept:
          result.fittedIntercept +
          major * Math.sin(angle) +
          minor * Math.cos(angle),
      };
    });

  const ellipse68 = ellipse(Math.sqrt(2.3));
  const ellipse95 = ellipse(Math.sqrt(5.991));
  const truthSlope = result.config.slope;
  const truthIntercept = result.config.intercept;
  const { slopeMin, slopeMax, interceptMin, interceptMax } = axes;
  const commonSpan = slopeMax - slopeMin;

  const scaleX = (slope: number) =>
    margin.left +
    ((slope - slopeMin) / (slopeMax - slopeMin)) * plotWidth;
  const scaleY = (intercept: number) =>
    margin.top +
    ((interceptMax - intercept) / (interceptMax - interceptMin)) *
      plotHeight;
  const xTicks = Array.from(
    { length: 5 },
    (_, index) => slopeMin + (index / 4) * (slopeMax - slopeMin),
  );
  const yTicks = Array.from(
    { length: 5 },
    (_, index) =>
      interceptMin + (index / 4) * (interceptMax - interceptMin),
  );
  const pathFor = (points: { slope: number; intercept: number }[]) =>
    points
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"} ${scaleX(point.slope)} ${scaleY(
            point.intercept,
          )}`,
      )
      .join(" ") + " Z";

  return (
    <svg
      className="parameter-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby="parameter-title parameter-description"
    >
      <title id="parameter-title">Slope and intercept parameter space</title>
      <desc id="parameter-description">
        Ground-truth and estimated parameter points with joint 68 and 95
        percent covariance ellipses.
      </desc>
      <defs>
        <clipPath id={clipId}>
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            rx="8"
          />
        </clipPath>
      </defs>
      <rect
        className="plot-background"
        x={margin.left}
        y={margin.top}
        width={plotWidth}
        height={plotHeight}
        rx="8"
      />

      {xTicks.map((tick) => (
        <g key={`parameter-x-${tick}`}>
          <line
            className="grid-line"
            x1={scaleX(tick)}
            y1={margin.top}
            x2={scaleX(tick)}
            y2={margin.top + plotHeight}
          />
          <text
            className="tick-label"
            x={scaleX(tick)}
            y={margin.top + plotHeight + 25}
            textAnchor="middle"
          >
            {round(tick, commonSpan < 2 ? 2 : 1)}
          </text>
        </g>
      ))}

      {yTicks.map((tick) => (
        <g key={`parameter-y-${tick}`}>
          <line
            className="grid-line"
            x1={margin.left}
            y1={scaleY(tick)}
            x2={margin.left + plotWidth}
            y2={scaleY(tick)}
          />
          <text
            className="tick-label"
            x={margin.left - 12}
            y={scaleY(tick) + 4}
            textAnchor="end"
          >
            {round(tick, commonSpan < 2 ? 2 : 1)}
          </text>
        </g>
      ))}

      <g clipPath={`url(#${clipId})`}>
        <line
          className="parameter-centerline"
          x1={scaleX(truthSlope)}
          y1={margin.top}
          x2={scaleX(truthSlope)}
          y2={margin.top + plotHeight}
        />
        <line
          className="parameter-centerline"
          x1={margin.left}
          y1={scaleY(truthIntercept)}
          x2={margin.left + plotWidth}
          y2={scaleY(truthIntercept)}
        />

        <path className="ellipse ellipse-95" d={pathFor(ellipse95)} />
        <path className="ellipse ellipse-68" d={pathFor(ellipse68)} />

        <circle
          className="estimated-parameter"
          cx={scaleX(result.fittedSlope)}
          cy={scaleY(result.fittedIntercept)}
          r="7"
        />
        <path
          className="truth-parameter"
          d={`M ${scaleX(result.config.slope)} ${
            scaleY(result.config.intercept) - 9
          } L ${scaleX(result.config.slope) + 9} ${scaleY(
            result.config.intercept,
          )} L ${scaleX(result.config.slope)} ${
            scaleY(result.config.intercept) + 9
          } L ${scaleX(result.config.slope) - 9} ${scaleY(
            result.config.intercept,
          )} Z`}
        />
      </g>

      <text
        className="axis-label"
        x={margin.left + plotWidth / 2}
        y={height - 10}
        textAnchor="middle"
      >
        slope m
      </text>
      <text
        className="axis-label"
        transform={`translate(18 ${margin.top + plotHeight / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        intercept b
      </text>
    </svg>
  );
}

export default function Home() {
  const [query, setQuery] = useState<{ x: number; y: number } | null>(null);
  const [editMode, setEditMode] = useState<"query" | "add" | "remove">("query");
  const [{ config, result }, dispatch] = useReducer(
    updateLabState,
    undefined,
    createInitialLabState,
  );
  const [mobileControlSection, setMobileControlSection] = useState<
    "sampling" | "truth" | "outliers" | "regularization"
  >("sampling");
  const [dataAxes, setDataAxes] = useState<DataAxes>(() =>
    fitDataAxes(result),
  );
  const [parameterAxes, setParameterAxes] = useState<ParameterAxes>(() =>
    fitParameterAxes(result),
  );
  const error = validateExperiment(config);

  const setValue = (key: keyof Experiment) => (value: number) =>
    dispatch({ type: "set-value", key, value });

  const outlierShare = useMemo(
    () =>
      result.points.length
        ? (result.points.filter((point) => point.outlier).length /
            result.points.length) *
          100
        : 0,
    [result],
  );

  const resample = () => {
    dispatch({ type: "resample" });
  };
  const plotTools = {
    query,
    mode: editMode,
    onPlace: (x: number, y: number) => editMode === "add" ? dispatch({ type: "add", x, y }) : setQuery({ x, y }),
    onRemove: (id: string) => dispatch({ type: "remove", id }),
  };
  const queryReadout = query && <p className="plot-query-readout">Query y = {round(query.y)} · ŷ = {round(result.fittedSlope * query.x + result.fittedIntercept)} · y − ŷ = {round(query.y - result.fittedSlope * query.x - result.fittedIntercept)}</p>;
  const performance = <div className="split-performance" aria-label="Training and test evaluation">
    <span>Set</span><span>n</span><span>RMSE</span><span>R²</span>
    <b>Training</b><strong>{result.points.filter(p => p.training).length}</strong><strong>{round(result.rmse)}</strong><strong>{round(result.r2)}</strong>
    <b>Test</b><strong>{result.points.filter(p => !p.training).length}</strong><strong>{round(result.testRmse)}</strong><strong>{round(result.testR2)}</strong>
  </div>;
  const editor = <div className="plot-editor">
    <div className="editor-modes">
      {(["query", "add", "remove"] as const).map(mode => <button key={mode} type="button" aria-pressed={editMode === mode} onClick={() => setEditMode(mode)}>{mode === "query" ? "Query" : mode === "add" ? "Add training" : "Remove training"}</button>)}
      <button type="button" disabled={!query} onClick={() => setQuery(null)}>Clear query</button>
    </div>
    <p>{editMode === "remove" ? "Tap a solid training point to remove it (minimum two)." : editMode === "add" ? "Tap the data plot to add a training point." : "Tap the data plot to place a query."}</p>
  </div>;

  return (
    <main>
      <header className="site-header">
        <a
          className="brand"
          href="#experiment"
          aria-label="Machine Learning regression interactive demo"
        >
          <span className="brand-mark" aria-hidden="true" />
          <span>
            MACHINE LEARNING
            <small>Regression interactive demo</small>
          </span>
        </a>
        <div className="header-context">
          <small>Change a parameter and watch the model respond.</small>
        </div>
        <div className="lesson-tag">
          Regularized regression
        </div>
      </header>

      <section className="lab" id="experiment">
        <section
          className="mobile-visualization"
          aria-label="Live regression visualization"
        >
          <div className="mobile-stage-actions">
            <div className="mobile-stage-title">
              <span>Live comparison</span>
              <strong>Data &amp; parameter space</strong>
            </div>
            <div className="mobile-resample-group" role="group" aria-label="Resample data">
            <button
              className="mobile-resample"
              type="button"
              onClick={resample}
              disabled={Boolean(error)}
              aria-label="Resample all data"
              title="Resample all data"
            >
              All ↻
            </button>
            <button className="mobile-resample" type="button" disabled={Boolean(error)} onClick={() => dispatch({ type: "resample", subset: "training" })} aria-label="Resample training only" title="Resample training only">Train ↻</button>
            <button className="mobile-resample" type="button" disabled={Boolean(error)} onClick={() => dispatch({ type: "resample", subset: "test" })} aria-label="Resample test only" title="Resample test only">Test ↻</button>
            </div>
          </div>

          {editor}
          <div className="mobile-plots-grid">
            <section className="mobile-plot-pane mobile-data-pane" aria-label="Data plot">
              <div className="mobile-plot-heading">
                <strong>Data</strong>
                <button
                  className="mobile-fit-axes"
                  type="button"
                  onClick={() => setDataAxes(fitDataAxes(result))}
                  aria-label="Fit axes to the data plot"
                >
                  Fit axes
                </button>
              </div>
              <div className="mobile-plot-frame">
                <RegressionPlot result={result} axes={dataAxes} {...plotTools} />
                <div className="plot-r2-readout" aria-label="Training and test R squared">
                  <span>Train R² <strong>{round(result.r2)}</strong></span>
                  <span>Test R² <strong>{round(result.testR2)}</strong></span>
                </div>
              </div>
              {queryReadout}
              <div className="mobile-plot-caption">
              <div className="legend" aria-label="Data plot legend">
                <span><i className="legend-line truth" />Truth</span>
                <span><i className="legend-line fit" />Fit</span>
                <span><i className="legend-dot sample" />Sample</span>
                <span><i className="legend-dot outlier" />Outlier</span>
              </div>
              </div>
            </section>

            <section className="mobile-plot-pane mobile-parameter-pane" aria-label="Parameter-space plot">
              <div className="mobile-plot-heading">
                <strong>Parameters</strong>
                <button
                  className="mobile-fit-axes"
                  type="button"
                  onClick={() => setParameterAxes(fitParameterAxes(result))}
                  aria-label="Fit axes to the parameter-space plot"
                >
                  Fit axes
                </button>
              </div>
              <div className="mobile-plot-frame">
                <ParameterPlot result={result} axes={parameterAxes} />
              </div>
              <div className="mobile-plot-caption">
                <div className="parameter-legend" aria-label="Parameter plot legend">
                <span><i className="legend-dot parameter-truth" />Truth</span>
                <span><i className="legend-dot parameter-estimate" />Estimate</span>
                <span>68% / 95% covariance</span>
                </div>
              </div>
            </section>
          </div>

        </section>

        <aside className="controls">
          <div className="controls-heading">
            <div>
              <p className="step-label">Experiment setup</p>
              <h2>Define the data</h2>
            </div>
            <div className="run-status">
              <span className="live-badge">Live</span>
              <span className="run-number">
                RUN {String(result.run).padStart(2, "0")}
              </span>
            </div>
          </div>

          <ControlSection
            number="01"
            title="Sampling window"
            id="sampling-controls"
            open={mobileControlSection === "sampling"}
            onOpen={() => setMobileControlSection("sampling")}
          >
              <div className="field-grid">
                <NumberField
                  label="x minimum"
                  value={config.xMin}
                  step={0.5}
                  onChange={setValue("xMin")}
                />
                <NumberField
                  label="x maximum"
                  value={config.xMax}
                  step={0.5}
                  onChange={setValue("xMax")}
                />
              </div>
              <SliderField
                label="Total regular points"
                value={config.samples}
                min={4}
                max={500}
                onChange={setValue("samples")}
              />
              <SliderField label="Test fraction" value={config.testFraction} min={10} max={50} step={5} onChange={setValue("testFraction")} />
              <p className="field-note">{config.samples + config.outliers} generated points in total (regular + outliers), split into training and test. Added training points are extra.</p>
          </ControlSection>

          <ControlSection
            number="02"
            title="Ground truth"
            id="truth-controls"
            open={mobileControlSection === "truth"}
            onOpen={() => setMobileControlSection("truth")}
          >
              <SliderField
                label="Slope"
                hint="m"
                value={config.slope}
                min={-10}
                max={10}
                step={0.1}
                onChange={setValue("slope")}
              />
              <SliderField
                label="Intercept"
                hint="b"
                value={config.intercept}
                min={-20}
                max={20}
                step={0.5}
                onChange={setValue("intercept")}
              />
              <SliderField
                label="Noise variance"
                hint="σ²"
                value={config.noiseVariance}
                step={0.1}
                min={0}
                max={25}
                onChange={setValue("noiseVariance")}
              />
          </ControlSection>

          <ControlSection
            number="03"
            title="Outliers"
            id="outlier-controls"
            open={mobileControlSection === "outliers"}
            onOpen={() => setMobileControlSection("outliers")}
          >
              <div className="field-grid">
                <NumberField
                  label="y minimum"
                  value={config.outlierMin}
                  step={0.5}
                  onChange={setValue("outlierMin")}
                />
                <NumberField
                  label="y maximum"
                  value={config.outlierMax}
                  step={0.5}
                  onChange={setValue("outlierMax")}
                />
              </div>
              <SliderField
                label="Number of outliers"
                value={config.outliers}
                min={0}
                max={100}
                onChange={setValue("outliers")}
              />
              <p className="field-note">
                Outlier x values use the sampling window; their y values are
                uniform in the range above.
              </p>
          </ControlSection>

          <ControlSection
            number="04"
            title="Regularization"
            id="regularization-controls"
            open={mobileControlSection === "regularization"}
            onOpen={() => setMobileControlSection("regularization")}
          >
              <SliderField
                label="L1 strength"
                hint="λ₁"
                value={config.l1}
                min={0}
                max={5}
                step={0.05}
                onChange={setValue("l1")}
              />
              <SliderField
                label="L2 strength"
                hint="λ₂"
                value={config.l2}
                min={0}
                max={5}
                step={0.05}
                onChange={setValue("l2")}
              />
              <p className="field-note regularization-note">
                ½ mean squared error + λ₁|m| + ½λ₂m². The intercept is not
                penalized.
              </p>
          </ControlSection>

          {error ? (
            <p className="error-message" role="alert">
              {error}
            </p>
          ) : null}

          <button
            className="run-button"
            type="button"
            onClick={resample}
            disabled={Boolean(error)}
          >
            <span>Resample data</span>
            <span aria-hidden="true">→</span>
          </button>
          <p className="button-note">
            Parameters update the current experiment instantly. Resample draws
            fresh random points.
          </p>
          <div className="editor-modes resample-subsets">
            <button type="button" disabled={Boolean(error)} onClick={() => dispatch({ type: "resample", subset: "training" })}>Training only ↻</button>
            <button type="button" disabled={Boolean(error)} onClick={() => dispatch({ type: "resample", subset: "test" })}>Test only ↻</button>
          </div>
          <p className="field-note">Added points stay fixed. Resampling training restores removed generated points. Axes stay fixed; use Fit axes when needed.</p>
        </aside>

        <div className="workspace">
          <div className="plots-grid">
            <div className="plot-card">
              <div className="plot-toolbar">
                <div className="plot-heading">
                  <div>
                    <p className="step-label">Observed sample</p>
                    <h2>Signal, noise &amp; influence</h2>
                  </div>
                  <div className="sample-summary">
                    <strong>{result.points.length}</strong>
                    <span>total points</span>
                  </div>
                </div>

                <div className="plot-actions">
                  <div className="legend" aria-label="Plot legend">
                    <span><i className="legend-line truth" />Ground truth</span>
                    <span><i className="legend-line fit" />Regression</span>
                    <span><i className="legend-dot sample" />Sample</span>
                    <span><i className="legend-dot outlier" />Outlier</span>
                  </div>
                  <button
                    className="fit-axes-button"
                    type="button"
                    onClick={() => setDataAxes(fitDataAxes(result))}
                    title="Fit the axes to the current data"
                  >
                    Fit axes
                  </button>
                </div>
              </div>

              {editor}
              <RegressionPlot result={result} axes={dataAxes} {...plotTools} />
              {queryReadout}
              <p className="field-note">Solid: training · outlined: test · black center: added · yellow: query; dashed segment: residual</p>
            </div>

            <div className="parameter-card">
              <div className="parameter-heading">
                <div>
                  <p className="step-label">Parameter space</p>
                  <h2>Estimation uncertainty</h2>
                </div>
                <div className="parameter-actions">
                  <div
                    className="parameter-legend"
                    aria-label="Parameter plot legend"
                  >
                    <span><i className="legend-dot parameter-truth" />Truth</span>
                    <span><i className="legend-dot parameter-estimate" />Estimate</span>
                  </div>
                  <button
                    className="fit-axes-button"
                    type="button"
                    onClick={() =>
                      setParameterAxes(fitParameterAxes(result))
                    }
                    title="Fit the axes to the estimate and covariance"
                  >
                    Fit axes
                  </button>
                </div>
              </div>
              <ParameterPlot result={result} axes={parameterAxes} />
              <p className="covariance-note">
                Locked equal scales · press Fit axes to reframe · 68% / 95%
              </p>
            </div>
          </div>

          {performance}
          <div className="results-grid equations-only">
            <section className="equation-card">
              <p className="step-label">Compare the models</p>
              <div className="equation-row truth-equation">
                <span>Ground truth</span>
                <strong>
                  y = {round(result.config.slope)}x{" "}
                  {result.config.intercept < 0 ? "−" : "+"}{" "}
                  {round(Math.abs(result.config.intercept))}
                </strong>
              </div>
              <div className="equation-row fit-equation">
                <span>
                  {result.config.l1 > 0 || result.config.l2 > 0
                    ? "Elastic Net estimate"
                    : "OLS estimate"}
                </span>
                <strong>
                  ŷ = {round(result.fittedSlope)}x{" "}
                  {result.fittedIntercept < 0 ? "−" : "+"}{" "}
                  {round(Math.abs(result.fittedIntercept))}
                </strong>
              </div>
            </section>

          </div>

          <div className="observation-strip">
            <p>
              <strong>{round(outlierShare, 1)}%</strong> of the requested sample
              are outliers. Compare the blue and coral lines to see how those
              influential observations and regularization move the fitted
              estimate.
            </p>
            <div className="author-credit">
              Made by Alexandre Bernardino with Codex
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
