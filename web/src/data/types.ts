export type AgentName = string;

export interface Trajectory {
  soc: number[];
  batteryPower: number[];
  gridPower: number[];
  costPerStep?: number[];
  cumulativeProfit?: number[];
  totalProfit: number;
  energyProfit?: number;
  peakPenalty?: number;
  degradationCost?: number;
  unservedCost?: number;
  terminalCost?: number;
  unservedKwh?: number;
  curtailedKwh?: number;
  throughput: number;
  peakViolations?: number;
  finalSoc: number;
}

export interface RobustnessPoint {
  noise: number;
  meanProfit: number;
  stdProfit: number;
  minProfit: number;
  maxProfit: number;
}

/** Price band on the replayed day (TOU window, flat rate or RTP tier). */
export interface Band {
  from: number;
  to: number;
  label: string;
  price: number;
  color: string;
  /** shaded on charts; defaults to label === "On-peak" */
  peak?: boolean;
}

/** Present only in the multi-grid exports (web/public/data/grids/*.json). */
export interface GridInfo {
  id: string;
  name: string;
  description: string;
  strategy: string;
  loadLabel: string;
  tariff: string;
  pvKwp: number;
  exportRatio: number;
  demandChargePerKwDay: number;
  importCapKw: number | null;
  exportCapKw: number | null;
  voll: number;
  terminalPenaltyPerUnitSoc: number;
  replayDate: string;
  nTestDays: number;
  robustnessRuns: number;
  obsDim: number;
  seeds: number;
}

export interface SimulationData {
  meta: {
    evalEpisodes: number;
    trajectoryInitialSoc: number;
    agentsTrained: string[];
    config: {
      batteryCapacityKwh: number;
      maxPowerKw: number;
      efficiency: number;
      rampRateKwPerH: number;
      peakThresholdKw: number | null;
      peakPenaltyRate: number;
      degradationCostPerKwh: number;
      forecastHorizonH: number;
      minSocRatio: number;
    };
    grid?: GridInfo;
    blurbs?: Record<string, string>;
  };
  profiles: {
    hours: number[];
    solar: number[];
    load: number[];
    netLoad: number[];
    price: number[];
    priceExport?: number[];
    bands?: Band[];
    forecastDemo: {
      noiseLevel: number;
      actualSolar: number[];
      actualLoad: number[];
    };
  };
  summary: Record<string, string | number>[];
  trajectories: Record<string, Trajectory>;
  perEpisode: Record<string, { profits: number[]; gaps?: number[] }>;
  robustness: {
    curves: Record<string, RobustnessPoint[]>;
    capture?: Record<string, { noise: number; capture: number; std?: number }[]>;
    summary: Record<string, string | number>[];
  } | null;
}

/** web/public/data/grids/index.json */
export interface GridIndexEntry {
  id: string;
  name: string;
  description: string;
  batteryKwh: number;
  batteryKw: number;
  pvKwp: number;
  tariff: string;
  loadLabel: string;
  capture: Record<string, number>;
  capture30: Record<string, number>;
}

export interface GridIndex {
  grids: GridIndexEntry[];
  methods: string[];
  robustMethods: string[];
  transfer: Record<string, Record<string, number>>;
  note: string;
}

/** "base" = the original single-home study, otherwise a grid id (G1..G6). */
export type Source = string;
