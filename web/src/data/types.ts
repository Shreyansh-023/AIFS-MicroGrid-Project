export type AgentName = "LP" | "SAC" | "R-SAC" | "PPO" | "Rule-Based" | "NoOp";

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
      peakThresholdKw: number;
      peakPenaltyRate: number;
      degradationCostPerKwh: number;
      forecastHorizonH: number;
      minSocRatio: number;
    };
  };
  profiles: {
    hours: number[];
    solar: number[];
    load: number[];
    netLoad: number[];
    price: number[];
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
    summary: Record<string, string | number>[];
  } | null;
}
