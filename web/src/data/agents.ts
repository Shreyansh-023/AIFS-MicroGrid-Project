export const AGENT_COLOR: Record<string, string> = {
  LP: "#d8dee6",
  SAC: "#4da3ff",
  "R-SAC": "#35d6a4",
  PPO: "#c48bff",
  "Rule-Based": "#f0a23c",
  NoOp: "#5d7490",
};

export const AGENT_BLURB: Record<string, string> = {
  LP: "Linear program with perfect foresight. Solves the whole day at once, so it marks the theoretical ceiling — but it commits to its schedule up front and cannot react.",
  SAC: "Soft Actor-Critic. Off-policy, entropy-regularised; learns a stochastic policy from a 50k replay buffer.",
  "R-SAC": "SAC trained under domain randomisation — every episode draws a fresh noise level up to 30%, so the policy learns to hedge against bad forecasts.",
  PPO: "Proximal Policy Optimisation. On-policy with clipped updates, trained on normalised observations.",
  "Rule-Based": "Fixed heuristic: charge on surplus solar, discharge when load exceeds generation and the price is high.",
  NoOp: "Battery idle for all 24 hours. The do-nothing floor.",
};

export const AGENT_KIND: Record<string, string> = {
  LP: "benchmark",
  SAC: "learned",
  "R-SAC": "learned",
  PPO: "learned",
  "Rule-Based": "heuristic",
  NoOp: "baseline",
};

export const DISPLAY_ORDER = ["LP", "R-SAC", "SAC", "PPO", "Rule-Based", "NoOp"];

export function orderAgents(names: string[]): string[] {
  return [...names].sort((a, b) => {
    const ia = DISPLAY_ORDER.indexOf(a);
    const ib = DISPLAY_ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
}

export function agentColor(name: string): string {
  return AGENT_COLOR[name] ?? "#7f8da0";
}

/** TOU bands as defined in data_loader.get_tou_prices(). */
export const TOU_BANDS = [
  { from: 0, to: 6, label: "Off-peak", price: 0.08, color: "#3fbf87" },
  { from: 6, to: 14, label: "Mid-peak", price: 0.15, color: "#e8a33c" },
  { from: 14, to: 21, label: "On-peak", price: 0.28, color: "#e8576b" },
  { from: 21, to: 24, label: "Mid-peak", price: 0.15, color: "#e8a33c" },
];

export function touAt(hour: number) {
  return TOU_BANDS.find((b) => hour >= b.from && hour < b.to) ?? TOU_BANDS[0];
}
