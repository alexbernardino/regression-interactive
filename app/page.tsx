"use client";

import { useMemo, useReducer, useState, type ReactNode } from "react";

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
};

type Point = {
  x: number;
  y: number;
  outlier: boolean;
};

type Result = {
  config: Experiment;
  points: Point[];
  fittedSlope: number;
  fittedIntercept: number;
  rmse: number;
  r2: number;
  covariance: {
    slopeVariance: number;
    interceptVariance: number;
    slopeIntercept: number;
    residualVariance: number;
  };
  run: number;
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

function runRegression(config: Experiment, run: number): Result {
  const points: Point[] = [];
  const xSpan = config.xMax - config.xMin;
  const noiseStdDev = Math.sqrt(config.noiseVariance);
  const random = seededRandom(8147 + run * 104729);

  for (let index = 0; index < config.samples; index += 1) {
    const x = config.xMin + random() * xSpan;
    const y =
      config.slope * x +
      config.intercept +
      gaussianRandom(random) * noiseStdDev;
    points.push({ x, y, outlier: false });
  }

  for (let index = 0; index < config.outliers; index += 1) {
    const x = config.xMin + random() * xSpan;
    const y =
      config.outlierMin +
      random() * (config.outlierMax - config.outlierMin);
    points.push({ x, y, outlier: true });
  }

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
    softThresholdedSlope / (normalizedSquaredX + config.l2);
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
    (residualVariance * denominator) / regularizedDenominator ** 2;
  const interceptVariance =
    residualVariance / points.length + xMean ** 2 * slopeVariance;
  const slopeIntercept = -xMean * slopeVariance;

  return {
    config: { ...config },
    points,
    fittedSlope,
    fittedIntercept,
    rmse: Math.sqrt(residualSum / points.length),
    r2: totalSum === 0 ? 1 : 1 - residualSum / totalSum,
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
  if (config.xMax <= config.xMin) {
    return "The maximum x value must be greater than the minimum.";
  }
  if (config.outlierMax <= config.outlierMin && config.outliers > 0) {
    return "The outlier maximum must be greater than its minimum.";
  }
  if (config.samples < 2 || config.samples > 500) {
    return "Choose between 2 and 500 regular samples.";
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
};

type LabAction =
  | { type: "set-value"; key: keyof Experiment; value: number }
  | { type: "resample" };

function createInitialLabState(): LabState {
  return {
    config: DEFAULTS,
    run: 1,
    result: runRegression(DEFAULTS, 1),
  };
}

function updateLabState(state: LabState, action: LabAction): LabState {
  if (action.type === "resample") {
    if (validateExperiment(state.config)) return state;
    const run = state.run + 1;
    return {
      ...state,
      run,
      result: runRegression(state.config, run),
    };
  }

  const config = { ...state.config, [action.key]: action.value };
  return {
    ...state,
    config,
    result: validateExperiment(config)
      ? state.result
      : runRegression(config, state.run),
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

function RegressionPlot({
  config,
  result,
}: {
  config: Experiment;
  result: Result;
}) {
  const width = 920;
  const height = 520;
  const margin = { top: 30, right: 42, bottom: 62, left: 70 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const yCandidates = [
    ...result.points.map((point) => point.y),
    config.slope * config.xMin + config.intercept,
    config.slope * config.xMax + config.intercept,
    result.fittedSlope * config.xMin + result.fittedIntercept,
    result.fittedSlope * config.xMax + result.fittedIntercept,
  ];
  const rawYMin = Math.min(...yCandidates);
  const rawYMax = Math.max(...yCandidates);
  const yPadding = Math.max((rawYMax - rawYMin) * 0.12, 1);
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;

  const scaleX = (x: number) =>
    margin.left + ((x - config.xMin) / (config.xMax - config.xMin)) * plotWidth;
  const scaleY = (y: number) =>
    margin.top + ((yMax - y) / (yMax - yMin)) * plotHeight;
  const xTickStart = Math.ceil(config.xMin);
  const xTickEnd = Math.floor(config.xMax);
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

  const trueStart = config.slope * config.xMin + config.intercept;
  const trueEnd = config.slope * config.xMax + config.intercept;
  const fitStart =
    result.fittedSlope * config.xMin + result.fittedIntercept;
  const fitEnd = result.fittedSlope * config.xMax + result.fittedIntercept;

  return (
    <svg
      className="regression-plot"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby="plot-title plot-description"
    >
      <title id="plot-title">Linear regression experiment</title>
      <desc id="plot-description">
        A scatter plot showing regular samples, outliers, the ground-truth line,
        and the ordinary least-squares regression line.
      </desc>
      <defs>
        <clipPath id="plot-area">
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

      <g clipPath="url(#plot-area)">
        <line
          className="truth-line"
          x1={scaleX(config.xMin)}
          y1={scaleY(trueStart)}
          x2={scaleX(config.xMax)}
          y2={scaleY(trueEnd)}
        />
        <line
          className="fit-line"
          x1={scaleX(config.xMin)}
          y1={scaleY(fitStart)}
          x2={scaleX(config.xMax)}
          y2={scaleY(fitEnd)}
        />

        {result.points
          .filter((point) => !point.outlier)
          .map((point, index) => (
            <circle
              className="sample-point"
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
              className="outlier-point"
              key={`outlier-${index}`}
              d={`M ${scaleX(point.x)} ${scaleY(point.y) - 8} L ${
                scaleX(point.x) + 8
              } ${scaleY(point.y)} L ${scaleX(point.x)} ${
                scaleY(point.y) + 8
              } L ${scaleX(point.x) - 8} ${scaleY(point.y)} Z`}
            />
          ))}
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

function ParameterPlot({ result }: { result: Result }) {
  const width = 480;
  const height = 480;
  const margin = { top: 24, right: 24, bottom: 56, left: 56 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
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
  const allSlopes = [
    result.config.slope,
    result.fittedSlope,
    ...ellipse95.map((point) => point.slope),
  ];
  const allIntercepts = [
    result.config.intercept,
    result.fittedIntercept,
    ...ellipse95.map((point) => point.intercept),
  ];
  const rawSlopeMin = Math.min(...allSlopes);
  const rawSlopeMax = Math.max(...allSlopes);
  const rawInterceptMin = Math.min(...allIntercepts);
  const rawInterceptMax = Math.max(...allIntercepts);
  const truthSlope = result.config.slope;
  const truthIntercept = result.config.intercept;
  const requiredHalfSpan = Math.max(
    Math.abs(rawSlopeMin - truthSlope),
    Math.abs(rawSlopeMax - truthSlope),
    Math.abs(rawInterceptMin - truthIntercept),
    Math.abs(rawInterceptMax - truthIntercept),
    0.5,
  );
  const commonSpan = requiredHalfSpan * 2.35;
  const slopeMin = truthSlope - commonSpan / 2;
  const slopeMax = truthSlope + commonSpan / 2;
  const interceptMin = truthIntercept - commonSpan / 2;
  const interceptMax = truthIntercept + commonSpan / 2;

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
  const [{ config, result }, dispatch] = useReducer(
    updateLabState,
    undefined,
    createInitialLabState,
  );
  const [mobileView, setMobileView] = useState<"data" | "parameters">("data");
  const [mobileControlSection, setMobileControlSection] = useState<
    "sampling" | "truth" | "outliers" | "regularization"
  >("sampling");
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
            <div className="mobile-view-tabs" role="tablist" aria-label="Plot view">
              <button
                id="mobile-data-tab"
                type="button"
                role="tab"
                aria-selected={mobileView === "data"}
                aria-controls="mobile-plot-panel"
                onClick={() => setMobileView("data")}
              >
                Data view
              </button>
              <button
                id="mobile-parameters-tab"
                type="button"
                role="tab"
                aria-selected={mobileView === "parameters"}
                aria-controls="mobile-plot-panel"
                onClick={() => setMobileView("parameters")}
              >
                Parameter space
              </button>
            </div>
            <button
              className="mobile-resample"
              type="button"
              onClick={resample}
              disabled={Boolean(error)}
              aria-label="Resample data"
              title="Resample data"
            >
              ↻
            </button>
          </div>

          <div
            className={`mobile-plot-frame mobile-${mobileView}-view`}
            id="mobile-plot-panel"
            role="tabpanel"
            aria-labelledby={
              mobileView === "data"
                ? "mobile-data-tab"
                : "mobile-parameters-tab"
            }
          >
            {mobileView === "data" ? (
              <RegressionPlot config={result.config} result={result} />
            ) : (
              <ParameterPlot result={result} />
            )}
          </div>

          <div className="mobile-plot-caption" aria-live="polite">
            {mobileView === "data" ? (
              <div className="legend" aria-label="Data plot legend">
                <span><i className="legend-line truth" />Truth</span>
                <span><i className="legend-line fit" />Fit</span>
                <span><i className="legend-dot sample" />Sample</span>
                <span><i className="legend-dot outlier" />Outlier</span>
              </div>
            ) : (
              <div className="parameter-legend" aria-label="Parameter plot legend">
                <span><i className="legend-dot parameter-truth" />Truth</span>
                <span><i className="legend-dot parameter-estimate" />Estimate</span>
                <span>68% / 95% covariance</span>
              </div>
            )}
          </div>

          <div className="mobile-metrics" aria-label="Live regression metrics">
            <div>
              <span>Slope m̂</span>
              <strong>{round(result.fittedSlope)}</strong>
            </div>
            <div>
              <span>Intercept b̂</span>
              <strong>{round(result.fittedIntercept)}</strong>
            </div>
            <div>
              <span>RMSE</span>
              <strong>{round(result.rmse)}</strong>
            </div>
            <div>
              <span>R²</span>
              <strong>{round(result.r2)}</strong>
            </div>
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
                label="Regular samples"
                value={config.samples}
                min={2}
                max={500}
                onChange={setValue("samples")}
              />
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

                <div className="legend" aria-label="Plot legend">
                  <span><i className="legend-line truth" />Ground truth</span>
                  <span><i className="legend-line fit" />Regression</span>
                  <span><i className="legend-dot sample" />Sample</span>
                  <span><i className="legend-dot outlier" />Outlier</span>
                </div>
              </div>

              <RegressionPlot config={result.config} result={result} />
            </div>

            <div className="parameter-card">
              <div className="parameter-heading">
                <div>
                  <p className="step-label">Parameter space</p>
                  <h2>Estimation uncertainty</h2>
                </div>
                <div
                  className="parameter-legend"
                  aria-label="Parameter plot legend"
                >
                  <span><i className="legend-dot parameter-truth" />Truth</span>
                  <span><i className="legend-dot parameter-estimate" />Estimate</span>
                </div>
              </div>
              <ParameterPlot result={result} />
              <p className="covariance-note">
                Truth-anchored equal scales · local covariance · 68% / 95%
              </p>
            </div>
          </div>

          <div className="results-grid">
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

            <section className="metric-card rmse-card">
              <div className="metric-top">
                <span>RMSE</span>
                <small>prediction error</small>
              </div>
              <strong>{round(result.rmse)}</strong>
              <p>Typical vertical distance between an observation and the fitted line.</p>
            </section>

            <section className="metric-card r2-card">
              <div className="metric-top">
                <span>R²</span>
                <small>explained variance</small>
              </div>
              <strong>{round(result.r2)}</strong>
              <p>Share of observed y variation explained by the fitted line.</p>
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
