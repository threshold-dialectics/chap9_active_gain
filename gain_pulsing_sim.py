from __future__ import annotations
import math, random, json, pathlib
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------- helpers
def calculate_gain_cost(g: float, phi_1: float, k_g: float) -> float:
    """Marginal cost of the *g* lever."""
    return math.inf if g < 0 else k_g * g ** phi_1


def calculate_tolerance_sheet(g: float, beta: float, F: float,
                              w1: float, w2: float, w3: float, C: float) -> float:
    """Tolerance function Θₜ."""
    if min(g, beta, F) <= 0:
        return 0.0
    return C * g ** w1 * beta ** w2 * F ** w3


def solve_for_initial_F_crit(theta: float, g0: float, b0: float,
                             w1: float, w2: float, w3: float, C: float) -> float:
    """Choose F₍crit₎ so that Θₜ(g₀,β₀,F₀)=θ₀."""
    if w3 <= 1e-6:
        return 200.0
    factor = C * g0 ** w1 * b0 ** w2
    if factor <= 1e-9:
        return 200.0
    return (theta / factor) ** (1.0 / w3)


def mean_sd(x: List[float]) -> Tuple[float, float]:
    return (float(np.mean(x)) if x else float("nan"),
            float(np.std(x))  if x else float("nan"))


# -------------------------------------------------------------------- Agent
class AgentGainPulsing:
    """Implements the adaptive pulse-decision logic."""

    def __init__(self, p: Dict[str, Any]):
        self.p = p
        # initial levers -------------------------------------------------
        self.g_lever = self.initial_g = p.get("initial_g", 1.0)
        self.beta_lever = self.initial_beta = p.get("initial_beta", 1.0)
        # tolerance parameters ------------------------------------------
        self.w_1, self.w_2, self.w_3 = p["w_1"], p["w_2"], p["w_3"]
        self.C_const = p.get("C_const", 1.0)
        self.theta0 = p.get("target_initial_Theta_T", 8.0)
        self.initial_F_crit = solve_for_initial_F_crit(
            self.theta0, self.initial_g, self.initial_beta,
            self.w_1, self.w_2, self.w_3, self.C_const)
        self.F_crit = p.get("initial_F_crit_override", self.initial_F_crit)
        p["calculated_initial_F_crit"] = self.F_crit  # expose to caller
        # cost model -----------------------------------------------------
        self.phi_1, self.k_g = p["phi_1"], p.get("k_g", 0.1)
        self.phi_beta, self.k_beta = p.get("phi_beta", 1.0), p.get("k_beta", 0.01)
        self.baseline_cost_F = p.get("baseline_cost_F_crit", 0.05)
        # policy knobs ---------------------------------------------------
        self.g_base_target = p.get("g_baseline_target", 1.0)
        self.g_pulse_target = p.get("g_pulse_target_val", 3.0)
        self.g_conservative = p.get("g_conservative_level", 0.3)
        self.beta_base = p.get("beta_baseline_target", 1.0)
        self.beta_conservative = p.get("beta_conservative_level", 0.5)
        self.strain_threat_factor = p.get("strain_threat_factor_for_pulse", 0.9)
        # thresholds -----------------------------------------------------
        self.SMCT = p.get("safety_margin_critical_threshold", self.theta0 * 0.05)
        self.SMRT = p.get("safety_margin_rapid_decrease_threshold", self.theta0 * 0.03)
        self.F_pulse_min = p.get("F_crit_pulse_min_requirement", 20.0)
        self.F_cons_thresh = p.get("F_crit_conservative_threshold", 12.0)
        self.F_proactive_abs = p.get("F_crit_proactive_abs_threshold", 25.0)
        # ranges ---------------------------------------------------------
        self.g_min, self.g_max = p.get("g_min", 0.1), p.get("g_max", 7.0)
        self.beta_min, self.beta_max = p.get("beta_min", 0.1), p.get("beta_max", 3.0)
        # pulse machinery ------------------------------------------------
        self.is_pulsing = False
        self.g_pulse_dur_max = p.get("g_pulse_duration_max", 10)
        self.g_pulse_dur_min = p.get("g_pulse_duration_min", 3)
        self.g_pulse_counter = 0
        # histories ------------------------------------------------------
        self.SM_hist = [0.0] * 5
        self.F_hist = [self.F_crit] * 5
        # heuristics -----------------------------------------------------
        self.use_heuristic = p.get("use_cost_benefit_heuristic", False)
        self.heuristic_type = p.get("heuristic_logic_type", "absolute_gain_vs_smct_thresh")
        self.norm_gain_thresh = p.get("normalized_gain_heuristic_threshold", 0.1)

    # ---------------------------------------------------------------- util
    def tolerance(self, g=None, beta=None, F=None) -> float:
        return calculate_tolerance_sheet(
            g if g is not None else self.g_lever,
            beta if beta is not None else self.beta_lever,
            F if F is not None else self.F_crit,
            self.w_1, self.w_2, self.w_3, self.C_const)

    # ---------------------------------------------------------- pulse heuristic
    def _pulse_allowed(self, tol_now: float, g_base: float) -> Tuple[bool, int]:
        gain_with_pulse = self.tolerance(g=self.g_pulse_target) - tol_now
        cost_base = calculate_gain_cost(g_base, self.phi_1, self.k_g)
        cost_pulse = calculate_gain_cost(self.g_pulse_target, self.phi_1, self.k_g)
        marg_cost = cost_pulse - cost_base
        F_avail = self.F_crit - self.F_cons_thresh
        steps_afford = math.inf if marg_cost <= 1e-9 else F_avail / marg_cost
        dur = int(max(self.g_pulse_dur_min, min(self.g_pulse_dur_max, steps_afford)))
        if self.heuristic_type == "absolute_gain_vs_smct_thresh":
            ok = gain_with_pulse > self.SMCT * 0.5 and steps_afford >= self.g_pulse_dur_min
        else:  # normalized_gain_vs_fixed_thresh
            norm_gain = gain_with_pulse / tol_now if tol_now > 1e-6 else math.inf
            ok = norm_gain > self.norm_gain_thresh and steps_afford >= self.g_pulse_dur_min
        return ok, dur

    # --------------------------------------------------------------- decision
    def decide(self, strain_avg: float, Theta_now: float):
        # rolling stats
        self.SM_hist.pop(0); self.SM_hist.append(Theta_now - strain_avg)
        self.F_hist.pop(0); self.F_hist.append(self.F_crit)
        SM_now = self.SM_hist[-1]
        SM_trend = np.mean(np.diff(self.SM_hist[-3:])) if len(self.SM_hist) >= 3 else 0.0
        F_trend = np.mean(np.diff(self.F_hist[-3:])) if len(self.F_hist) >= 3 else 0.0
        # baseline g
        g_base = self.g_base_target
        if self.F_crit < self.F_proactive_abs or F_trend < -0.3:
            red = max(0.0, (self.F_crit - self.F_cons_thresh) /
                      (self.F_proactive_abs - self.F_cons_thresh + 1e-6))
            g_base = self.g_conservative + (self.g_base_target - self.g_conservative) * red
        # pulse engine
        if self.is_pulsing:
            self.g_pulse_counter -= 1
            if self.g_pulse_counter <= 0 or self.F_crit < self.F_pulse_min * 0.5:
                self.is_pulsing = False
            g_target = self.g_pulse_target if self.is_pulsing else g_base
        else:
            trigger = (
                (Theta_now > 0 and strain_avg > self.strain_threat_factor * Theta_now) or
                (SM_now < self.SMCT) or
                (SM_trend < -self.SMRT)
            )
            g_target = g_base
            if trigger and self.F_crit > self.F_pulse_min:
                allow = True; dur = self.g_pulse_dur_max
                if self.use_heuristic:
                    allow, dur = self._pulse_allowed(Theta_now, g_base)
                if allow:
                    self.is_pulsing = True
                    self.g_pulse_counter = dur
                    g_target = self.g_pulse_target
            elif self.F_crit < self.F_cons_thresh:
                g_target = self.g_conservative
        # beta
        beta_target = (self.beta_base if
                       (self.F_crit >= self.F_cons_thresh * 0.9 and
                        SM_now   >= self.SMCT       * 0.8) else self.beta_conservative)
        # clamp
        self.g_lever   = float(np.clip(g_target,   self.g_min,   self.g_max))
        self.beta_lever = float(np.clip(beta_target, self.beta_min, self.beta_max))

    # ------------------------------------------------------- F-crit depletion
    def update_F_crit(self):
        cost_g    = calculate_gain_cost(self.g_lever, self.phi_1, self.k_g)
        cost_beta = self.k_beta * self.beta_lever ** self.phi_beta
        self.F_crit = max(0.0, self.F_crit - (cost_g + cost_beta + self.baseline_cost_F))


# ---------------------------------------------------------------- Environment
class EnvironmentStrainShocks:
    """Simple stochastic shock generator."""

    def __init__(self, p: Dict[str, Any]):
        self.baseline = p.get("baseline_strain", 0.5)
        self.mag      = p.get("shock_magnitude_factor", 7.0)
        self.dur      = p.get("shock_duration", 15)
        self.mu       = p.get("shock_interval_mean", 100)
        self.sd       = p.get("shock_interval_std", 20)
        self.tau      = p.get("strain_avg_tau", 20)
        self.time_since = 0
        self.next_shock = self._sample_next()
        self.in_shock   = 0
        self.avg_hist: List[float] = []

    def _sample_next(self) -> int:
        return max(self.dur + 10, int(random.gauss(self.mu, self.sd)))

    def instant(self) -> float:
        self.time_since += 1
        if self.in_shock:  # currently in shock
            self.in_shock -= 1
            if self.in_shock == 0:
                self.time_since = 0
                self.next_shock = self._sample_next()
            x = self.baseline * self.mag
        elif self.time_since >= self.next_shock:  # new shock starts
            self.in_shock = self.dur
            x = self.baseline * self.mag
        else:
            x = self.baseline
        x += random.gauss(0.0, x * 0.02)
        self.avg_hist.append(x)
        if len(self.avg_hist) > self.tau:
            self.avg_hist.pop(0)
        return max(0.0, x)

    def average(self) -> float:
        return float(np.mean(self.avg_hist)) if self.avg_hist else 0.0


# -------------------------------------------------------------- single run
def run_sim(sim_p: Dict[str, Any],
            a_p: Dict[str, Any],
            e_p: Dict[str, Any],
            run_id: int):
    if (seed := sim_p.get("seed")) is not None:
        random.seed(seed + run_id); np.random.seed(seed + run_id)

    agent = AgentGainPulsing(a_p.copy())
    env   = EnvironmentStrainShocks(e_p.copy())

    hist: Dict[str, List[Any]] = {k: [] for k in
        ("time","g","beta","F","Theta","strain","strain_avg","SM","pulse","surv")}

    pulses: List[int] = []; pulse_start = -1
    breached = False; T = sim_p["duration"]

    for t in range(T):
        x_inst = env.instant()
        x_avg  = env.average()
        Theta  = agent.tolerance()

        was_pulsing = agent.is_pulsing
        agent.decide(x_avg, Theta)
        if not was_pulsing and agent.is_pulsing:
            pulse_start = t
        if was_pulsing and not agent.is_pulsing and pulse_start != -1:
            pulses.append(t - pulse_start); pulse_start = -1

        agent.update_F_crit()

        # log
        hist["time"].append(t)
        hist["g"].append(agent.g_lever)
        hist["beta"].append(agent.beta_lever)
        hist["F"].append(agent.F_crit)
        hist["Theta"].append(Theta)
        hist["strain"].append(x_inst)
        hist["strain_avg"].append(x_avg)
        hist["SM"].append(Theta - x_avg)
        hist["pulse"].append(int(agent.is_pulsing))

        breached_now = x_avg > Theta or agent.F_crit <= 0.0
        hist["surv"].append(int(not breached_now))
        if breached_now:
            breached = True
            if agent.is_pulsing and pulse_start != -1:
                pulses.append(t - pulse_start + 1)
            break

    if agent.is_pulsing and pulse_start != -1 and not breached:
        pulses.append(T - pulse_start)

    df = pd.DataFrame(hist)
    total_pulse_time = int(df["pulse"].sum())
    return df, (not breached), agent.initial_F_crit, pulses, total_pulse_time


# ----------------------------------------------------------- plot helper
def plot_smct_curves(df_smct: pd.DataFrame):
    """Generate three headline plots for the SMCT sweep."""
    plt.figure(figsize=(8,6))
    plt.plot(df_smct["smct"], df_smct["survival_rate"], "-o")
    plt.xlabel("SMCT"); plt.ylabel("Survival rate"); plt.grid(True)
    plt.savefig("sweep_survival_vs_smct.png"); plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(df_smct["smct"], df_smct["avg_pulse_count"], "-o")
    plt.xlabel("SMCT"); plt.ylabel("Avg pulse count"); plt.grid(True)
    plt.savefig("sweep_pulse_count_vs_smct.png"); plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(df_smct["smct"], df_smct["avg_Fcrit_surv"], "-o")
    plt.xlabel("SMCT"); plt.ylabel("Avg Fcrit (survivors)"); plt.grid(True)
    plt.savefig("sweep_fcrit_survivors_vs_smct.png"); plt.close()


# ---------------------------------------------------------------- main
if __name__ == "__main__":

    # --------------------------- global meta constants (for Methods header)
    THETA0 = 8.0
    META: Dict[str, Any] = dict(
        duration             = 500,
        baseline_strain      = THETA0 * 0.3,
        shock_factor         = 3.2,
        shock_duration       = 20,
        shock_interval_mu    = 150,
        shock_interval_sd    = 25,
        strain_avg_tau       = 15,
        baseline_cost_F_crit = 0.01,
        k_beta               = 0.001,
        phi_beta             = 1.0,
        g_setpoints          = dict(baseline=1.0, pulse=3.0, conservative=0.3),
        g_pulse_duration_min = 5,
        g_pulse_duration_max = 15,
        beta_setpoints       = dict(baseline=1.0, conservative=0.5),
        F_crit_thresholds    = dict(pulse_min=20.0, conservative=12.0, proactive_abs=25.0),
        safety_margin_rapid_decrease_threshold = 0.03 * THETA0,
        w                    = [0.3, 0.3, 0.4],
    )

    # --------------------------- base parameter dictionaries
    sim_params = {"duration": META["duration"], "seed": 123}

    base_agent: Dict[str, Any] = dict(
        initial_g=1.0, initial_beta=1.0, target_initial_Theta_T=THETA0,
        w_1=0.3, w_2=0.3, w_3=0.4, C_const=1.0,
        k_g=0.02, k_beta=META["k_beta"], baseline_cost_F_crit=META["baseline_cost_F_crit"],
        phi_1=0.30, phi_beta=META["phi_beta"],
        g_baseline_target=META["g_setpoints"]["baseline"],
        g_pulse_target_val=META["g_setpoints"]["pulse"],
        g_conservative_level=META["g_setpoints"]["conservative"],
        g_pulse_duration_max=META["g_pulse_duration_max"],
        g_pulse_duration_min=META["g_pulse_duration_min"],
        beta_baseline_target=META["beta_setpoints"]["baseline"],
        beta_conservative_level=META["beta_setpoints"]["conservative"],
        strain_threat_factor_for_pulse=0.9,
        F_crit_pulse_min_requirement=META["F_crit_thresholds"]["pulse_min"],
        F_crit_conservative_threshold=META["F_crit_thresholds"]["conservative"],
        F_crit_proactive_abs_threshold=META["F_crit_thresholds"]["proactive_abs"],
        g_min=0.1, g_max=6.0, beta_min=0.1, beta_max=2.5,
        use_cost_benefit_heuristic=False,
        heuristic_logic_type="absolute_gain_vs_smct_thresh",
        normalized_gain_heuristic_threshold=0.1,
        safety_margin_rapid_decrease_threshold=META["safety_margin_rapid_decrease_threshold"],
    )

    base_env: Dict[str, Any] = dict(
        baseline_strain=META["baseline_strain"],
        shock_magnitude_factor=META["shock_factor"],
        shock_duration=META["shock_duration"],
        shock_interval_mean=META["shock_interval_mu"],
        shock_interval_std=META["shock_interval_sd"],
        strain_avg_tau=META["strain_avg_tau"],
    )

    # --------------------------- scenario catalogue
    scenarios: Dict[str, Dict[str, Any]] = {}
    smct_keys: List[str] = []

    # (1) SMCT sweep
    phi1, kg = 0.30, 0.020
    scenarios["SMCTSweep_Baseline_HeurOFF"] = {"agent": {"phi_1": phi1, "k_g": kg, "use_cost_benefit_heuristic": False}}
    for mult in [0.05,0.15,0.25,0.35,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]:
        name = f"SMCTSweep_x{mult:.2f}_AbsGainHeurON"
        scenarios[name] = {"agent": {"phi_1": phi1, "k_g": kg,
                                     "use_cost_benefit_heuristic": True,
                                     "heuristic_logic_type": "absolute_gain_vs_smct_thresh",
                                     "safety_margin_critical_threshold": THETA0 * mult}}
        smct_keys.append(name)

    # (2) PDM stress test
    scenarios["PDMStrong_phi1_0.75_kg_0.05_HeurOFF"] = {
        "agent": {"phi_1": 0.75, "k_g": 0.05, "use_cost_benefit_heuristic": False}}
    scenarios["PDMStrong_phi1_0.75_kg_0.05_SMCTx0.25_HeurON"] = {
        "agent": {"phi_1": 0.75, "k_g": 0.05,
                  "use_cost_benefit_heuristic": True,
                  "heuristic_logic_type": "absolute_gain_vs_smct_thresh",
                  "safety_margin_critical_threshold": THETA0 * 0.25}}

    # (3) Normalised gain heuristic sweep
    scenarios["NormGain_Baseline_HeurOFF_phi1_0.30_kg_0.02"] = {
        "agent": {"phi_1": 0.30, "k_g": 0.02, "use_cost_benefit_heuristic": False}}
    for th in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        scenarios[f"NormGain_Threshx{th:.2f}_HeurON"] = {
            "agent": {"phi_1": 0.30, "k_g": 0.02,
                      "use_cost_benefit_heuristic": True,
                      "heuristic_logic_type": "normalized_gain_vs_fixed_thresh",
                      "normalized_gain_heuristic_threshold": th}}

    # (4) anomaly deep-dive (single run)
    scenarios["AnomalyInvestigate_SMCTx0.70_AbsGainHeurON"] = {
        "agent": {"phi_1": phi1, "k_g": kg,
                  "use_cost_benefit_heuristic": True,
                  "heuristic_logic_type": "absolute_gain_vs_smct_thresh",
                  "safety_margin_critical_threshold": THETA0 * 0.70},
        "runs": 1}

    # --------------------------- Monte-Carlo execution
    default_runs = 50
    results: Dict[str, Dict[str, Any]] = {}

    for scen, cfg in scenarios.items():
        n_runs = cfg.get("runs", default_runs)
        a_cfg = base_agent.copy(); a_cfg.update(cfg.get("agent", {}))
        e_cfg = base_env.copy();   e_cfg.update(cfg.get("env",   {}))

        # ensure w₁+w₂+w₃ = 1
        w_sum = a_cfg["w_1"] + a_cfg["w_2"] + a_cfg["w_3"]
        if not math.isclose(w_sum, 1.0):
            for w in ("w_1","w_2","w_3"):
                a_cfg[w] /= w_sum if w_sum else 1.0

        # run n times
        t_surv=[]; pulses=[]; p_durs=[]; tot_p=[]; F_surv=[]; F_breach=[]; init_F=[]
        for run in range(n_runs):
            df,surv,iF,pdurs,tp = run_sim(sim_params, a_cfg, e_cfg, run)
            init_F.append(iF); pulses.append(int((df["pulse"].diff() > 0).sum()))
            p_durs.extend(pdurs); tot_p.append(tp)
            (F_surv if surv else F_breach).append(df["F"].iloc[-1])
            t_surv.append(df["time"].iloc[-1] if not surv else META["duration"])

        n_surv = sum(1 for t in t_surv if t == META["duration"])

        # summarise
        results[scen] = dict(
            n_runs=n_runs, n_survived=n_surv, survival_rate=n_surv/n_runs,
            avg_time_surv=mean_sd(t_surv)[0], sd_time_surv=mean_sd(t_surv)[1],
            avg_pulse_count=mean_sd(pulses)[0], sd_pulse_count=mean_sd(pulses)[1],
            avg_total_pulse_time=mean_sd(tot_p)[0], sd_total_pulse_time=mean_sd(tot_p)[1],
            avg_indiv_pulse_dur=mean_sd(p_durs)[0], sd_indiv_pulse_dur=mean_sd(p_durs)[1],
            avg_Fcrit_surv=mean_sd(F_surv)[0], sd_Fcrit_surv=mean_sd(F_surv)[1],
            avg_Fcrit_breach=mean_sd(F_breach)[0], sd_Fcrit_breach=mean_sd(F_breach)[1],
            avg_init_Fcrit=mean_sd(init_F)[0], sd_init_Fcrit=mean_sd(init_F)[1],
            phi_1=a_cfg["phi_1"], k_g=a_cfg["k_g"],
            heuristic=("OFF" if not a_cfg["use_cost_benefit_heuristic"]
                       else "AbsGain" if a_cfg["heuristic_logic_type"]=="absolute_gain_vs_smct_thresh"
                       else "NormGain"),
            smct=a_cfg.get("safety_margin_critical_threshold"),
            norm_gain_thresh=a_cfg.get("normalized_gain_heuristic_threshold"),
        )
        print(f"{scen:42s}  surv={results[scen]['survival_rate']*100:6.2f}%")

    # --------------------------- export tables
    df_all = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index":"scenario"})
    df_all.to_csv("all_scenarios_summary.csv", index=False, encoding="utf-8")

    # meta-header (JSON for easy parsing)
    meta_header = "# Global simulation parameters\n" + json.dumps(META, indent=2) + "\n\n"

    # plain-text mirror
    hdr=("Scenario".ljust(42)+" n  surv%  phi1  k_g  heuristic smct normG"
         "  pulses±SD  t_surv±SD  totPulse±SD  indivDur±SD  Fsurv±SD  Fbreach±SD  initF±SD\n")
    with open("all_scenarios_summary.txt","w",encoding="utf-8") as f:
        f.write(meta_header); f.write(hdr)
        for _,r in df_all.iterrows():
            f.write(
                f"{r['scenario'][:42].ljust(42)}{int(r['n_runs']):3d} {r['survival_rate']*100:6.2f} "
                f"{r['phi_1']:.2f} {r['k_g']:.3f} {r['heuristic']:<9} "
                f"{r['smct']:4.1f} "
                f"{r['norm_gain_thresh'] if not pd.isna(r['norm_gain_thresh']) else '':5} "
                f"{r['avg_pulse_count']:.2f}±{r['sd_pulse_count']:<4.2f} "
                f"{r['avg_time_surv']:.1f}±{r['sd_time_surv']:<4.1f} "
                f"{r['avg_total_pulse_time']:.1f}±{r['sd_total_pulse_time']:<4.1f} "
                f"{r['avg_indiv_pulse_dur']:.2f}±{r['sd_indiv_pulse_dur']:<4.2f} "
                f"{r['avg_Fcrit_surv']:.1f}±{r['sd_Fcrit_surv']:<4.1f} "
                f"{r['avg_Fcrit_breach']:.1f}±{r['sd_Fcrit_breach']:<4.1f} "
                f"{r['avg_init_Fcrit']:.1f}±{r['sd_init_Fcrit']:<4.1f}\n")

    # SMCT table
    df_smct = df_all[df_all["heuristic"]=="AbsGain"].sort_values("smct")
    with open("smct_sweep_summary.txt","w",encoding="utf-8") as f:
        f.write("Scenario".ljust(40)+"SMCT   n  SurvRate  Pulse±SD  Fsurv±SD\n")
        for _,r in df_smct.iterrows():
            f.write(f"{r['scenario'][:40].ljust(40)}{r['smct']:5.2f} {int(r['n_runs']):3d} "
                    f"{r['survival_rate']*100:7.2f}% "
                    f"{r['avg_pulse_count']:5.2f}±{r['sd_pulse_count']:<4.2f} "
                    f"{r['avg_Fcrit_surv']:.1f}±{r['sd_Fcrit_surv']:<4.1f}\n")

    # --------------------------- plots
    if not df_smct.empty:
        plot_smct_curves(df_smct)

    print("\n Finished --- CSV/TXT summaries and plots are in", pathlib.Path().resolve())
