import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # Added SubprocVecEnv for potential future use
from stable_baselines3.common.monitor import Monitor # Good for wrapping envs
import os
import time
from collections import deque

# --- Constants for Reward Shaping (Tunable) ---
R_STEP_SURVIVAL = 0.1
R_TERMINAL_SUCCESS = 100.0
R_TERMINAL_FAILURE = -100.0
PENALTY_OPERATING_COST_FACTOR = 0.025 # Slightly increased
PENALTY_LOW_SM_CRIT_THRESHOLD_FACTOR_VERY_LOW = 0.25
PENALTY_LOW_SM_VERY_LOW = 0.6 # Slightly increased
PENALTY_LOW_SM_CRIT_THRESHOLD_FACTOR_LOW = 0.75
PENALTY_LOW_SM_LOW = 0.25 # Slightly increased
REWARD_SM_IMPROVEMENT_FACTOR_POSITIVE = 0.15 # Slightly increased
PENALTY_PULSING_HIGH_SM_TARGET_THETA_FACTOR = 1.2
PENALTY_PULSING_HIGH_SM = 0.2 # Slightly increased
PENALTY_INEFFECTIVE_PULSE_START_TARGET_THETA_FACTOR = 0.05
PENALTY_INEFFECTIVE_PULSE_START = 0.4 # Slightly increased
PENALTY_PULSE_START = 0.75 # Increased to further discourage unnecessary pulses


# --- Core Lever Mechanics (Unchanged) ---
def calculate_gain_cost(g_level, phi_1, k_g):
    if g_level < 0: return float('inf')
    return k_g * (g_level ** phi_1)

def calculate_tolerance_sheet(g_level, beta_level, F_crit_level, w_1, w_2, w_3, C_const):
    if g_level <= 0 or beta_level <= 0 or F_crit_level <= 0:
        return 0.0
    return C_const * (g_level**w_1) * (beta_level**w_2) * (F_crit_level**w_3)

def solve_for_initial_F_crit(target_Theta_T, g_init, beta_init, w_1, w_2, w_3, C_const):
    if w_3 <= 1e-6 : return 200.0
    try:
        factor = C_const * (g_init**w_1) * (beta_init**w_2)
        if factor <= 1e-9: return 200.0
        val_for_F_power = target_Theta_T / factor
        if val_for_F_power < 0: return 200.0
        initial_F_crit = val_for_F_power ** (1.0 / w_3)
        return initial_F_crit
    except (OverflowError, ZeroDivisionError, ValueError): return 200.0

# --- Agent Class (State Holder) ---
class AgentStateHolder:
    def __init__(self, params):
        self.params = params.copy() # Work with a copy
        self.initial_g = self.params.get("initial_g", 1.0)
        self.initial_beta = self.params.get("initial_beta", 1.0)
        self.target_initial_Theta_T = self.params.get("target_initial_Theta_T", 8.0)
        self.w_1 = self.params["w_1"]; self.w_2 = self.params["w_2"]; self.w_3 = self.params["w_3"]
        self.C_const = self.params.get("C_const", 1.0)
        self.initial_F_crit_calculated = solve_for_initial_F_crit(
            self.target_initial_Theta_T, self.initial_g, self.initial_beta,
            self.w_1, self.w_2, self.w_3, self.C_const)
        if "initial_F_crit_override" in self.params:
             self.initial_F_crit_calculated = self.params["initial_F_crit_override"]
        self.max_initial_F_crit = self.params.get("max_initial_F_crit_for_norm", 250.0) # For normalization
        self.phi_1 = self.params["phi_1"]; self.k_g = self.params.get("k_g", 0.1)
        self.phi_beta = self.params.get("phi_beta", 1.0); self.k_beta = self.params.get("k_beta", 0.01)
        self.baseline_cost_F_crit = self.params.get("baseline_cost_F_crit", 0.05)
        self.g_baseline_target = self.params.get("g_baseline_target", 1.0)
        self.g_pulse_target_val = self.params.get("g_pulse_target_val", 3.0)
        self.g_conservative_level = self.params.get("g_conservative_level", 0.5)
        self.beta_baseline_target = self.params.get("beta_baseline_target", 1.0)
        self.beta_conservative_level = self.params.get("beta_conservative_level", 0.5)
        self.strain_threat_factor_for_pulse = self.params.get("strain_threat_factor_for_pulse", 0.90)
        self.safety_margin_critical_threshold = self.params.get("safety_margin_critical_threshold", self.target_initial_Theta_T * 0.05)
        self.safety_margin_rapid_decrease_threshold = self.params.get("safety_margin_rapid_decrease_threshold", self.target_initial_Theta_T * 0.03)
        self.F_crit_pulse_min_requirement = self.params.get("F_crit_pulse_min_requirement", 15.0)
        self.F_crit_conservative_threshold = self.params.get("F_crit_conservative_threshold", 10.0)
        self.F_crit_proactive_abs_threshold = self.params.get("F_crit_proactive_abs_threshold", 30.0)
        self.g_pulse_duration_max = self.params.get("g_pulse_duration_max", 10)
        self.g_pulse_duration_min = self.params.get("g_pulse_duration_min", 3)
        self.g_min = self.params.get("g_min", 0.1); self.g_max = self.params.get("g_max", 7.0)
        self.beta_min = self.params.get("beta_min", 0.1); self.beta_max = self.params.get("beta_max", 3.0)
        self.history_len_for_trends = self.params.get("history_len_for_trends", 5)
        self.log_detailed_rl_env_steps = self.params.get("log_detailed_rl_env_steps", False)
        self.reset() # Initialize state variables

    def reset(self):
        self.g_lever = self.initial_g; self.beta_lever = self.initial_beta
        self.F_crit = self.initial_F_crit_calculated
        self.is_pulsing_g = False; self.g_pulse_counter = 0
        self.current_g_baseline = self.g_baseline_target
        self.safety_margin_history = deque([self.target_initial_Theta_T * 0.7] * self.history_len_for_trends, maxlen=self.history_len_for_trends)
        self.F_crit_history = deque([self.F_crit] * self.history_len_for_trends, maxlen=self.history_len_for_trends)
        self.strain_avg_history = deque([self.target_initial_Theta_T * 0.3] * self.history_len_for_trends, maxlen=self.history_len_for_trends)
        self.current_time_step = 0
        # Init trends to 0
        self.delta_F_crit_trend = 0.0
        self.delta_safety_margin_trend = 0.0
        self.delta_strain_avg_trend = 0.0


    def get_tolerance(self, g=None, beta=None, F_crit_val=None):
        g_to_use = g if g is not None else self.g_lever
        b_to_use = beta if beta is not None else self.beta_lever
        F_to_use = F_crit_val if F_crit_val is not None else self.F_crit
        return calculate_tolerance_sheet(g_to_use, b_to_use, F_to_use,
                                         self.w_1, self.w_2, self.w_3, self.C_const)

    def update_F_crit(self, delta_t=1):
        cost_g = calculate_gain_cost(self.g_lever, self.phi_1, self.k_g)
        cost_beta = self.k_beta * (self.beta_lever ** self.phi_beta)
        total_cost = cost_g + cost_beta + self.baseline_cost_F_crit
        self.F_crit -= total_cost * delta_t
        self.F_crit = max(0, self.F_crit)

    def update_history_and_trends(self, current_strain_avg, current_tolerance_val):
        current_safety_margin = current_tolerance_val - current_strain_avg
        self.safety_margin_history.append(current_safety_margin)
        self.F_crit_history.append(self.F_crit) # F_crit *before* current step's cost
        self.strain_avg_history.append(current_strain_avg)
        # Calculate trends (simple diff for now, could be slope)
        if len(self.F_crit_history) >= 3: # Ensure enough data points
            self.delta_F_crit_trend = np.mean(np.diff(list(self.F_crit_history)[-3:]))
            self.delta_safety_margin_trend = np.mean(np.diff(list(self.safety_margin_history)[-3:]))
            self.delta_strain_avg_trend = np.mean(np.diff(list(self.strain_avg_history)[-3:]))
        else: # Not enough history for trend
            self.delta_F_crit_trend = 0.0
            self.delta_safety_margin_trend = 0.0
            self.delta_strain_avg_trend = 0.0


    def update_dynamic_baselines(self):
        avg_F_crit_trend = self.delta_F_crit_trend
        if self.F_crit < self.F_crit_proactive_abs_threshold or avg_F_crit_trend < -0.3 :
            reduction_factor = max(0, (self.F_crit - self.F_crit_conservative_threshold) /
                                   (self.F_crit_proactive_abs_threshold - self.F_crit_conservative_threshold + 1e-6))
            self.current_g_baseline = self.g_conservative_level + (self.g_baseline_target - self.g_conservative_level) * reduction_factor
            self.current_g_baseline = max(self.g_conservative_level, self.current_g_baseline)
        else:
            self.current_g_baseline = self.g_baseline_target
        current_safety_margin = self.safety_margin_history[-1] if self.safety_margin_history else self.target_initial_Theta_T
        if self.F_crit < self.F_crit_conservative_threshold * 0.9 or \
           current_safety_margin < self.safety_margin_critical_threshold * 0.8:
            self.beta_lever = self.beta_conservative_level
        else:
            self.beta_lever = self.beta_baseline_target
        self.beta_lever = np.clip(self.beta_lever, self.beta_min, self.beta_max)

# --- EnvironmentStrainShocks (Unchanged) ---
class EnvironmentStrainShocks:
    def __init__(self, params):
        self.params = params.copy()
        self.baseline_strain = self.params.get("baseline_strain", 0.5)
        self.shock_magnitude_factor = self.params.get("shock_magnitude_factor", 7.0)
        self.shock_duration = self.params.get("shock_duration", 15)
        self.shock_interval_mean = self.params.get("shock_interval_mean", 100)
        self.shock_interval_std = self.params.get("shock_interval_std", 20)
        self.strain_avg_tau = self.params.get("strain_avg_tau", 20)
        self.rng_seed_base = self.params.get("seed", None)
        self.reset() # Initialize RNG

    def reset(self, seed_offset=0):
        current_seed = self.rng_seed_base
        if current_seed is not None:
            current_seed += seed_offset
        self.rng = np.random.default_rng(current_seed)
        self.current_strain_val = self.baseline_strain
        self.time_since_last_shock = 0
        self.next_shock_at = self._get_next_shock_time()
        self.in_shock_counter = 0
        self.strain_history_for_avg = deque(maxlen=self.strain_avg_tau)

    def _get_next_shock_time(self):
        return max(self.shock_duration + 10, int(self.rng.normal(self.shock_interval_mean, self.shock_interval_std)))

    def get_instantaneous_strain(self, time_step):
        self.time_since_last_shock +=1
        current_val = self.baseline_strain
        if self.in_shock_counter > 0:
            current_val = self.baseline_strain * self.shock_magnitude_factor
            self.in_shock_counter -= 1
            if self.in_shock_counter == 0:
                self.time_since_last_shock = 0
                self.next_shock_at = self._get_next_shock_time()
        elif self.time_since_last_shock >= self.next_shock_at:
            self.in_shock_counter = self.shock_duration
            current_val = self.baseline_strain * self.shock_magnitude_factor
        self.current_strain_val = max(0, current_val + self.rng.normal(0, current_val * 0.02))
        return self.current_strain_val

    def get_avg_delta_P_tau(self, current_inst_strain):
        self.strain_history_for_avg.append(current_inst_strain)
        if not self.strain_history_for_avg: return 0.0
        return np.mean(self.strain_history_for_avg)

# --- RL Environment ---
class GainPulsingRLEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, sim_duration, agent_params, env_strain_params, render_mode=None, worker_id=0, is_eval_env=False, eval_log_filename=None):
        super().__init__()
        self.sim_duration = sim_duration
        self.agent_params_base = agent_params.copy()
        self.env_strain_params_base = env_strain_params.copy()
        self.render_mode = render_mode
        self.worker_id = worker_id # For unique seeding in SubprocVecEnv
        self.is_eval_env = is_eval_env
        self.eval_log_filename = eval_log_filename
        if self.is_eval_env and self.eval_log_filename:
            mode = 'a' if os.path.exists(self.eval_log_filename) else 'w'
            with open(self.eval_log_filename, mode) as f:
                if mode == 'w': f.write(f"--- Eval Log Start: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"--- New Eval Env Instance (worker_id {self.worker_id}) ---\n")

        # Create fresh instances for each env
        self.agent = AgentStateHolder(self.agent_params_base)
        current_env_strain_params = self.env_strain_params_base.copy()
        if "seed" in current_env_strain_params and current_env_strain_params["seed"] is not None:
            current_env_strain_params["seed"] += self.worker_id # Offset seed by worker_id
        self.env_strain = EnvironmentStrainShocks(current_env_strain_params)

        self.action_space = spaces.Discrete(4)
        self.num_obs_features = 14
        # Define typical min/max for observation normalization more explicitly
        # F_crit (0, init_F_crit), Theta_T (0, target_Theta_T*1.5), strain_avg (0, target_Theta_T*1.5)
        # SM (-target_Theta_T, target_Theta_T), g (g_min, g_max), beta (beta_min, beta_max)
        # is_pulsing (0,1), g_pulse_counter(0,1)
        # trends (-0.5*target, 0.5*target), time_since_shock(0,1), in_shock(0,1), g_base(g_min,g_max)
        # Using -5 to 5 as a general clipped range after normalization.
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.num_obs_features,), dtype=np.float32)
        
        # Initialize these in reset
        self.current_sim_step = 0
        self.current_avg_strain = 0.0
        self.current_tolerance_val = 0.0
        self.current_safety_margin = 0.0
        self.last_strain_avg = 0.0; self.last_tolerance = 0.0; self.last_F_crit = 0.0; self.last_safety_margin = 0.0
        self.history_for_render = []; self.action_taken_this_step = 0; self.was_pulsing_before_action = False


    def _log_eval_detail(self, message):
        if self.is_eval_env and self.eval_log_filename:
            with open(self.eval_log_filename, 'a') as f:
                f.write(f"[WID:{self.worker_id} T={self.current_sim_step}] {message}\n")

    def _get_obs(self):
        # Normalization factors
        norm_factor_F_crit = self.agent.max_initial_F_crit + 1e-6 # Use a fixed max for F_crit norm
        norm_factor_Theta_T = self.agent.target_initial_Theta_T * 1.5 + 1e-6 # Allow some overhead
        norm_factor_SM = self.agent.target_initial_Theta_T + 1e-6 # SM can be negative
        norm_factor_trend_F = self.agent.target_initial_Theta_T * 0.1 + 1e-6 # Expected F_crit change per step group
        norm_factor_trend_SM_Strain = self.agent.target_initial_Theta_T * 0.2 + 1e-6 # Expected SM/Strain change
        norm_factor_time_shock = self.env_strain.shock_interval_mean * 2.0 + 1e-6

        norm_F_crit = (self.agent.F_crit / norm_factor_F_crit) * 2 - 1 # Scale to approx [-1, 1] if F_crit can be > max_initial
        norm_Theta_T = self.current_tolerance_val / norm_factor_Theta_T
        norm_strain_avg = self.current_avg_strain / norm_factor_Theta_T # Same factor as Theta_T
        norm_SM = self.current_safety_margin / norm_factor_SM # Centered around 0

        norm_g_lever = (self.agent.g_lever - self.agent.g_min) / (self.agent.g_max - self.agent.g_min + 1e-6)
        norm_beta_lever = (self.agent.beta_lever - self.agent.beta_min) / (self.agent.beta_max - self.agent.beta_min + 1e-6)
        
        norm_is_pulsing_g = 1.0 if self.agent.is_pulsing_g else 0.0
        norm_g_pulse_counter = self.agent.g_pulse_counter / (self.agent.g_pulse_duration_max + 1e-6)

        norm_delta_F_crit = self.agent.delta_F_crit_trend / norm_factor_trend_F
        norm_delta_SM = self.agent.delta_safety_margin_trend / norm_factor_trend_SM_Strain
        norm_delta_strain_avg = self.agent.delta_strain_avg_trend / norm_factor_trend_SM_Strain

        norm_time_since_shock = self.env_strain.time_since_last_shock / norm_factor_time_shock
        norm_in_shock_counter = self.env_strain.in_shock_counter / (self.env_strain.shock_duration + 1e-6)
        norm_current_g_baseline = (self.agent.current_g_baseline - self.agent.g_min) / (self.agent.g_max - self.agent.g_min + 1e-6)

        obs_raw = np.array([
            norm_F_crit, norm_Theta_T, norm_strain_avg, norm_SM, norm_g_lever, norm_beta_lever,
            norm_is_pulsing_g, norm_g_pulse_counter, norm_delta_F_crit, norm_delta_SM,
            norm_delta_strain_avg, norm_time_since_shock, norm_in_shock_counter, norm_current_g_baseline
        ], dtype=np.float32)
        return np.clip(obs_raw, -5.0, 5.0) # Final clip, should ideally be within [-1,1] or [0,1] from normalization

    def _calculate_reward(self, breached, F_crit_zero, just_survived_step, safety_margin_now, sm_improvement):
        reward_components = {}
        reward = 0.0
        is_terminal = False

        if breached or F_crit_zero:
            reward = R_TERMINAL_FAILURE
            reward_components['terminal_failure'] = R_TERMINAL_FAILURE
            is_terminal = True
        elif self.current_sim_step >= self.sim_duration :
             reward = R_TERMINAL_SUCCESS
             reward_components['terminal_success'] = R_TERMINAL_SUCCESS
             is_terminal = True # Technically truncated, but we give success reward
        else: # Not terminal yet for this step
            if just_survived_step:
                reward += R_STEP_SURVIVAL
                reward_components['step_survival'] = R_STEP_SURVIVAL

            cost_g = calculate_gain_cost(self.agent.g_lever, self.agent.phi_1, self.agent.k_g)
            cost_beta = self.agent.k_beta * (self.agent.beta_lever ** self.agent.phi_beta)
            op_cost_penalty = PENALTY_OPERATING_COST_FACTOR * (cost_g + cost_beta)
            reward -= op_cost_penalty
            reward_components['op_cost_penalty'] = -op_cost_penalty

            sm_crit_thresh = self.agent.safety_margin_critical_threshold
            if safety_margin_now < sm_crit_thresh * PENALTY_LOW_SM_CRIT_THRESHOLD_FACTOR_VERY_LOW:
                reward -= PENALTY_LOW_SM_VERY_LOW
                reward_components['sm_very_low_penalty'] = -PENALTY_LOW_SM_VERY_LOW
            elif safety_margin_now < sm_crit_thresh * PENALTY_LOW_SM_CRIT_THRESHOLD_FACTOR_LOW:
                reward -= PENALTY_LOW_SM_LOW
                reward_components['sm_low_penalty'] = -PENALTY_LOW_SM_LOW
            
            if sm_improvement > 0 and self.last_safety_margin < sm_crit_thresh:
                sm_imp_reward = REWARD_SM_IMPROVEMENT_FACTOR_POSITIVE * sm_improvement
                reward += sm_imp_reward
                reward_components['sm_improvement_reward'] = sm_imp_reward
            
            if self.agent.is_pulsing_g and safety_margin_now > self.agent.target_initial_Theta_T * PENALTY_PULSING_HIGH_SM_TARGET_THETA_FACTOR:
                reward -= PENALTY_PULSING_HIGH_SM
                reward_components['pulsing_high_sm_penalty'] = -PENALTY_PULSING_HIGH_SM

            if self.action_taken_this_step > 0 and not self.was_pulsing_before_action:
                reward -= PENALTY_PULSE_START
                reward_components['pulse_start_penalty'] = -PENALTY_PULSE_START
                if self.last_safety_margin < sm_crit_thresh and \
                   sm_improvement < self.agent.target_initial_Theta_T * PENALTY_INEFFECTIVE_PULSE_START_TARGET_THETA_FACTOR:
                    reward -= PENALTY_INEFFECTIVE_PULSE_START
                    reward_components['ineffective_pulse_penalty'] = -PENALTY_INEFFECTIVE_PULSE_START
        
        if self.is_eval_env:
            self._log_eval_detail(f"RewardCalc: Term={is_terminal}, Total={reward:.3f}, Breached={breached}, F_Zero={F_crit_zero}, SM={safety_margin_now:.2f}, SM_improve={sm_improvement:.2f}, Fcrit={self.agent.F_crit:.2f}, Action={self.action_taken_this_step}, WasPulsing={self.was_pulsing_before_action}, Components={reward_components}")
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for SB3's seeding
        self.agent.reset() # Reset agent state
        # Use worker_id for env_strain seeding if available, else use gym's seed
        seed_offset = self.worker_id if hasattr(self, 'worker_id') and self.worker_id is not None else (seed if seed is not None else 0)
        self.env_strain.reset(seed_offset=seed_offset)

        self.current_sim_step = 0
        self.current_inst_strain = self.env_strain.get_instantaneous_strain(self.current_sim_step)
        self.current_avg_strain = self.env_strain.get_avg_delta_P_tau(self.current_inst_strain)
        self.current_tolerance_val = self.agent.get_tolerance()
        self.current_safety_margin = self.current_tolerance_val - self.current_avg_strain
        
        # Prime history for first observation
        for _ in range(self.agent.history_len_for_trends):
            self.agent.update_history_and_trends(self.current_avg_strain, self.current_tolerance_val)
        self.agent.update_dynamic_baselines()

        self.last_F_crit = self.agent.F_crit
        self.last_safety_margin = self.current_safety_margin
        self.last_strain_avg = self.current_avg_strain
        self.last_tolerance = self.current_tolerance_val

        self.history_for_render = [] # Crucial: Clear history on reset for plotting
        self.action_taken_this_step = 0
        self.was_pulsing_before_action = False

        if self.is_eval_env: self._log_eval_detail(f"Env Reset Complete. Initial SM={self.current_safety_margin:.2f}, Fcrit={self.agent.F_crit:.2f}")
        initial_obs = self._get_obs()
        info = self._get_info()
        return initial_obs, info

    def _apply_rl_action(self, action_idx):
        self.action_taken_this_step = action_idx # Store for reward calculation
        self.was_pulsing_before_action = self.agent.is_pulsing_g # Store for reward calculation
        
        log_prefix = f"[RL_ENV t={self.current_sim_step}] " if self.agent.log_detailed_rl_env_steps or (self.is_eval_env and self.eval_log_filename) else ""

        if not self.agent.is_pulsing_g: self.agent.g_lever = self.agent.current_g_baseline
        if self.agent.is_pulsing_g:
            self.agent.g_pulse_counter -= 1
            if action_idx == 0 or self.agent.g_pulse_counter <= 0 or self.agent.F_crit < self.agent.F_crit_pulse_min_requirement * 0.5:
                self.agent.is_pulsing_g = False; self.agent.g_pulse_counter = 0
                self.agent.g_lever = self.agent.current_g_baseline
                if log_prefix: self._log_eval_detail(f"{log_prefix}RL_ACTION: Stop/End Pulse. Set g_lever={self.agent.g_lever:.2f}")
            else:
                self.agent.g_lever = self.agent.g_pulse_target_val
                if log_prefix: self._log_eval_detail(f"{log_prefix}RL_ACTION: Continue Pulse. g_pulse_counter={self.agent.g_pulse_counter}")
        if not self.agent.is_pulsing_g and action_idx > 0: # Try to start a new pulse
            if self.agent.F_crit > self.agent.F_crit_pulse_min_requirement:
                self.agent.is_pulsing_g = True; self.agent.g_lever = self.agent.g_pulse_target_val
                durations = [0, self.agent.g_pulse_duration_min,
                             (self.agent.g_pulse_duration_min + self.agent.g_pulse_duration_max) // 2,
                             self.agent.g_pulse_duration_max]
                self.agent.g_pulse_counter = durations[action_idx]
                if log_prefix: self._log_eval_detail(f"{log_prefix}RL_ACTION: Start Pulse. Duration={self.agent.g_pulse_counter}, g_lever={self.agent.g_lever:.2f}")
            elif log_prefix:
                self._log_eval_detail(f"{log_prefix}RL_ACTION: Tried Start Pulse, F_crit too low (Fcrit={self.agent.F_crit:.2f}). g_lever={self.agent.current_g_baseline:.2f}")
                self.agent.g_lever = self.agent.current_g_baseline # Ensure it's on baseline if pulse failed
        elif not self.agent.is_pulsing_g and action_idx == 0 and log_prefix:
             self._log_eval_detail(f"{log_prefix}RL_ACTION: No Pulse. g_lever={self.agent.g_lever:.2f}")
        self.agent.g_lever = np.clip(self.agent.g_lever, self.agent.g_min, self.agent.g_max)

    def _is_decision_point(self):
        trigger_pulse_flag = False
        # Ensure history is populated before accessing agent.safety_margin_history[-1]
        current_safety_margin = self.agent.safety_margin_history[-1] if len(self.agent.safety_margin_history) == self.agent.history_len_for_trends else self.agent.target_initial_Theta_T
        avg_sm_trend = self.agent.delta_safety_margin_trend # This is now updated in update_history_and_trends

        if self.current_tolerance_val > 0 and self.current_avg_strain > self.agent.strain_threat_factor_for_pulse * self.current_tolerance_val:
            trigger_pulse_flag = True
        elif current_safety_margin < self.agent.safety_margin_critical_threshold:
            trigger_pulse_flag = True
        elif avg_sm_trend < -self.agent.safety_margin_rapid_decrease_threshold: # Negative trend is bad
            trigger_pulse_flag = True
        return self.agent.is_pulsing_g or trigger_pulse_flag

    def step(self, action_idx):
        self._apply_rl_action(action_idx) # Apply chosen action and update agent's pulse state
        
        # Store state *before* the simulation step for reward calculation context
        self.last_F_crit = self.agent.F_crit
        self.last_safety_margin = self.current_safety_margin # SM from *end* of previous full env step
        self.last_strain_avg = self.current_avg_strain
        self.last_tolerance = self.current_tolerance_val

        accumulated_reward_since_last_decision = 0; steps_taken_since_last_decision = 0
        breached_in_epoch = False; F_crit_zero_in_epoch = False

        while True:
            steps_taken_since_last_decision += 1; self.current_sim_step += 1
            self.agent.current_time_step = self.current_sim_step
            
            # Order of operations for a single simulation step:
            # 1. Agent internal state updates (baselines, etc., based on *previous* step's outcome)
            self.agent.update_dynamic_baselines()
            # 2. Environment evolves (strain changes)
            self.current_inst_strain = self.env_strain.get_instantaneous_strain(self.current_sim_step)
            self.current_avg_strain = self.env_strain.get_avg_delta_P_tau(self.current_inst_strain)
            # 3. Agent acts (tolerance calculated based on current levers, F_crit consumed)
            self.current_tolerance_val = self.agent.get_tolerance() # Uses current agent.g_lever, F_crit
            self.agent.update_F_crit() # Consumes F_crit
            # 4. Outcome assessment
            self.current_safety_margin = self.current_tolerance_val - self.current_avg_strain
            # 5. Update history for *next* step's trends and observation
            self.agent.update_history_and_trends(self.current_avg_strain, self.current_tolerance_val)
            
            sm_improvement = self.current_safety_margin - self.last_safety_margin # Improvement over the *full epoch* since last RL action

            breached_this_sim_step = self.current_avg_strain > self.current_tolerance_val
            F_crit_zero_this_sim_step = self.agent.F_crit <= 0
            breached_in_epoch = breached_in_epoch or breached_this_sim_step
            F_crit_zero_in_epoch = F_crit_zero_in_epoch or F_crit_zero_this_sim_step
            
            step_reward = self._calculate_reward(breached_this_sim_step, F_crit_zero_this_sim_step, True, self.current_safety_margin, sm_improvement)
            accumulated_reward_since_last_decision += step_reward
            
            self.history_for_render.append({ # This history is for plotting a single episode
                'time': self.current_sim_step, 'g_lever': self.agent.g_lever, 'beta_lever': self.agent.beta_lever,
                'F_crit': self.agent.F_crit, 'Theta_T': self.current_tolerance_val, 'strain_avg': self.current_avg_strain,
                'safety_margin': self.current_safety_margin, 'is_pulsing': 1 if self.agent.is_pulsing_g else 0,
                'reward_step': step_reward
            })

            terminated = breached_in_epoch or F_crit_zero_in_epoch
            truncated = self.current_sim_step >= self.sim_duration
            
            # If a pulse was active, and its counter ran out (and RL didn't re-initiate), ensure it's off
            if self.agent.is_pulsing_g and self.agent.g_pulse_counter <=0 :
                 self.agent.is_pulsing_g = False
                 self.agent.g_lever = self.agent.current_g_baseline # Revert to baseline

            if terminated or truncated or self._is_decision_point(): break # Exit internal loop
        
        observation = self._get_obs()
        info = self._get_info()
        info["steps_taken_since_last_decision"] = steps_taken_since_last_decision
        info["accumulated_reward_since_last_decision"] = accumulated_reward_since_last_decision
        return observation, accumulated_reward_since_last_decision, bool(terminated), bool(truncated), info

    def _get_info(self):
        return {"current_sim_step": self.current_sim_step, "F_crit": self.agent.F_crit,
                "safety_margin": self.current_safety_margin, "is_pulsing": self.agent.is_pulsing_g,
                "g_pulse_counter": self.agent.g_pulse_counter, "g_lever": self.agent.g_lever}
    def render(self): pass # Not implemented for this text-based example
    def close(self): pass
    def _render_frame(self): pass # Placeholder

# --- Plotting Function ---
def plot_rl_episode_results(df_history, title="RL Episode Results", save_dir="."):
    if df_history.empty:
        print(f"Warning: Empty history for {title}, cannot plot (history length: 0).")
        return
    print(f"Plotting for {title} with {len(df_history)} history points. Saving to {save_dir}")

    fig, axs = plt.subplots(6, 1, figsize=(18, 20), sharex=True)
    time_col = df_history['time']

    axs[0].plot(time_col, df_history['g_lever'], label='g_lever');
    pulsing_plot_scale_val = np.max(df_history['g_lever']) if not df_history['g_lever'].empty and np.any(df_history['g_lever'] > 0) else 1.0
    if pulsing_plot_scale_val == 0: pulsing_plot_scale_val = 1.0
    axs[0].plot(time_col, np.array(df_history['is_pulsing']) * pulsing_plot_scale_val * 0.5, label='Is Pulsing (scaled)', linestyle=':', alpha=0.7, color='red')
    axs[0].set_ylabel('g_lever'); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(time_col, df_history['beta_lever'], label='beta_lever', color='orange'); axs[1].set_ylabel('beta_lever'); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(time_col, df_history['F_crit'], label='F_crit', color='green'); axs[2].set_ylabel('F_crit'); axs[2].legend(); axs[2].grid(True)

    ax3_twin = axs[3].twinx()
    axs[3].plot(time_col, df_history['strain_avg'], label=r'Strain_avg ($\Delta P_{\tau}$)', color='red')
    axs[3].plot(time_col, df_history['Theta_T'], label=r'Tolerance ($\Theta_T$)', color='blue', linestyle='--')
    axs[3].set_ylabel('Strain / Tolerance'); axs[3].legend(loc='center left'); axs[3].grid(True)
    ax3_twin.plot(time_col, df_history['safety_margin'], label='Safety Margin', color='purple', linestyle=':')
    ax3_twin.set_ylabel(r'Safety Margin ($\Theta_T - Strain_{avg}$)', color='purple'); ax3_twin.tick_params(axis='y', labelcolor='purple'); ax3_twin.legend(loc='center right')
    ax3_twin.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    axs[4].plot(time_col, df_history['reward_step'], label='Step Reward', color='teal'); axs[4].set_ylabel('Reward'); axs[4].legend(); axs[4].grid(True)
    
    if 'reward_step' in df_history.columns:
        cumulative_reward = df_history['reward_step'].cumsum()
        axs[5].plot(time_col, cumulative_reward, label='Cumulative Reward', color='brown')
        axs[5].set_xlabel('Time Step'); axs[5].set_ylabel('Cumulative Reward'); axs[5].legend(); axs[5].grid(True)

    fig.suptitle(title, fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename_sanitized = title.replace(' ', '_').replace(':', '').replace('/', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    filepath = os.path.join(save_dir, f"{filename_sanitized}.png")
    try:
        plt.savefig(filepath)
        print(f"Saved episode plot: {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
    plt.close(fig)


# --- Main ---
if __name__ == "__main__":
    SIM_DURATION = 500
    GENERAL_TARGET_INITIAL_THETA_T = 8.0
    UNIQUE_RUN_ID = time.strftime("%Y%m%d-%H%M%S")
    LOG_DIR_BASE = "rl_gain_pulsing_logs"
    LOG_DIR = os.path.join(LOG_DIR_BASE, UNIQUE_RUN_ID)
    MODEL_PATH = os.path.join(LOG_DIR, "ppo_gain_pulsing_model")
    EVAL_LOG_FILE = os.path.join(LOG_DIR, "eval_details.log")
    PLOT_SAVE_DIR = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    # Use a different seed for the base configuration of each run
    # Individual components (env stochasticity, PPO) will then use offsets or their own seeds.
    main_run_seed = int(time.time() % 10000) 
    np.random.seed(main_run_seed)

    agent_params_template = {
        "initial_g": 1.0, "initial_beta": 1.0, "target_initial_Theta_T": GENERAL_TARGET_INITIAL_THETA_T,
        "w_1": 0.3, "w_2": 0.3, "w_3": 0.4, "C_const": 1.0, "k_g": 0.02, "k_beta": 0.001,
        "baseline_cost_F_crit": 0.01, "phi_1": 0.30, "phi_beta": 1.0, "g_baseline_target": 1.0,
        "g_pulse_target_val": 3.0, "g_conservative_level": 0.3, "g_pulse_duration_max": 15,
        "g_pulse_duration_min": 5, "beta_baseline_target": 1.0, "beta_conservative_level": 0.5,
        "strain_threat_factor_for_pulse": 0.9, "F_crit_pulse_min_requirement": 20.0,
        "F_crit_conservative_threshold": 12.0, "F_crit_proactive_abs_threshold": 25.0,
        "g_min": 0.1, "g_max": 6.0, "beta_min": 0.1, "beta_max": 2.5, "history_len_for_trends": 5,
        "max_initial_F_crit_for_norm": 250.0, # Added for normalization consistency
        "log_detailed_rl_env_steps": False # General step logging for non-eval callback runs
    }
    agent_params_template["safety_margin_critical_threshold"] = agent_params_template.get("safety_margin_critical_threshold", agent_params_template["target_initial_Theta_T"] * 0.05)
    agent_params_template["safety_margin_rapid_decrease_threshold"] = agent_params_template.get("safety_margin_rapid_decrease_threshold", agent_params_template["target_initial_Theta_T"] * 0.03)

    env_strain_params_template = {
        "baseline_strain": GENERAL_TARGET_INITIAL_THETA_T * 0.3, "shock_magnitude_factor": 3.2,
        "shock_duration": 20, "shock_interval_mean": 150, "shock_interval_std": 25,
        "strain_avg_tau": 15, "seed": main_run_seed + 1
    }
    
    TRAIN_MODEL = True
    TOTAL_TIMESTEPS = 300_000 # Can increase this further
    PPO_LEARNING_RATE = 3e-4    # e.g., 1e-4, 5e-4
    PPO_NET_ARCH_PI = [64, 64]  # Policy network
    PPO_NET_ARCH_VF = [64, 64]  # Value network
    # PPO_NET_ARCH_PI = [128, 128] 
    # PPO_NET_ARCH_VF = [128, 128]
    N_ENVS = 1 # Number of parallel environments for training, 1 for DummyVecEnv
               # For SubprocVecEnv, you can increase this (e.g., 4 or 8 if CPU cores allow)

    print(f"--- Run ID: {UNIQUE_RUN_ID} ---")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Plot Save Directory: {PLOT_SAVE_DIR}")


    def make_env(rank, seed=0, is_eval=False, log_file=None):
        """
        Utility function for multiprocessed env.
        :param rank: (int) index of the subprocess
        :param seed: (int) the initial seed for RNG
        :param is_eval: (bool) whether this is an evaluation environment
        :param log_file: (str) path to log file for eval env
        """
        def _init():
            # Ensure agent_params and env_strain_params are fresh copies for each env
            current_agent_params = agent_params_template.copy()
            current_env_strain_params = env_strain_params_template.copy()
            if "seed" in current_env_strain_params and current_env_strain_params["seed"] is not None:
                 current_env_strain_params["seed"] += rank # Make seed unique per environment process
            
            env = GainPulsingRLEnv(SIM_DURATION, current_agent_params, current_env_strain_params, 
                                   worker_id=rank, is_eval_env=is_eval, eval_log_filename=log_file)
            env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_rank{rank}.csv") if not is_eval else None) # Log training stats
            return env
        # Set a unique seed for each environment process.
        # This is now handled by passing rank to GainPulsingRLEnv for env_strain seeding.
        # The Monitor wrapper will handle its own seeding if necessary from SB3.
        return _init

    print("Creating Training RL Environment(s)...")
    # Use DummyVecEnv for simplicity if N_ENVS = 1, otherwise SubprocVecEnv for parallel
    if N_ENVS == 1:
        vec_train_env = DummyVecEnv([make_env(0, seed=main_run_seed+10)])
    else:
        vec_train_env = SubprocVecEnv([make_env(i, seed=main_run_seed+10+i) for i in range(N_ENVS)])
    
    # Check a raw instance of the environment
    raw_check_env = GainPulsingRLEnv(SIM_DURATION, agent_params_template, env_strain_params_template)
    check_env(raw_check_env, warn=True)
    del raw_check_env # Clean up
    print("Training environment check passed.")
    

    eval_vec_env = DummyVecEnv([make_env(0, seed=main_run_seed+100, is_eval=True, log_file=EVAL_LOG_FILE)])

    if TRAIN_MODEL:
        print(f"\n--- Starting Training for {TOTAL_TIMESTEPS} Timesteps ---")
        eval_callback = EvalCallback(eval_vec_env, best_model_save_path=os.path.join(LOG_DIR, 'best_model'),
                                     log_path=LOG_DIR, eval_freq=max(TOTAL_TIMESTEPS // (20 * N_ENVS), 5000 // N_ENVS),
                                     deterministic=True, render=False, n_eval_episodes=5, warn=False)
        
        model = PPO("MlpPolicy", vec_train_env, verbose=1, tensorboard_log=LOG_DIR,
                    learning_rate=PPO_LEARNING_RATE,
                    policy_kwargs=dict(net_arch=dict(pi=PPO_NET_ARCH_PI, vf=PPO_NET_ARCH_VF)),
                    seed=main_run_seed + 2, # Seed for PPO's operations
                    n_steps=2048 // N_ENVS, # SB3 default is 2048, adjust if using multiple envs
                    batch_size=64,       # SB3 default
                    n_epochs=10          # SB3 default
                    ) 
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback])
        model.save(MODEL_PATH)
        print(f"Training complete. Model saved to {MODEL_PATH}")
        del model
    
    # --- Evaluation ---
    print("\n--- Starting Evaluation ---")
    # Create a fresh VecEnv for final evaluation.
    # The make_env function already includes the Monitor wrapper if not is_eval.
    # For final evaluation, we might not strictly need Monitor unless we want its stats here too.
    # Let's make it explicit: for final eval, we might want the raw env for direct access.
    
    def make_raw_env(rank, seed=0): # Helper for creating raw env for final eval
        def _init():
            current_agent_params = agent_params_template.copy()
            current_env_strain_params = env_strain_params_template.copy()
            if "seed" in current_env_strain_params and current_env_strain_params["seed"] is not None:
                 current_env_strain_params["seed"] += rank
            env = GainPulsingRLEnv(SIM_DURATION, current_agent_params, current_env_strain_params, 
                                   worker_id=rank, is_eval_env=False, eval_log_filename=None) # Not an eval_callback env
            return env
        return _init

    final_eval_vec_env = DummyVecEnv([make_raw_env(0, seed=main_run_seed+200)])


    if os.path.exists(MODEL_PATH + ".zip"):
        model_to_eval_path = MODEL_PATH + ".zip"
    elif os.path.exists(os.path.join(LOG_DIR, 'best_model.zip')):
        model_to_eval_path = os.path.join(LOG_DIR, 'best_model.zip')
        print(f"Using best model from callback: {model_to_eval_path}")
    else:
        print("Error: No trained model found to load for evaluation.")
        if plt.get_fignums(): plt.show()
        exit()
        
    model = PPO.load(model_to_eval_path, env=final_eval_vec_env) # Pass the vec_env
    print(f"Model loaded from {model_to_eval_path}")

    num_eval_episodes = 10
    all_episode_rewards = []; all_survival_times = []; all_total_pulses = []; all_final_F_crit = []
    
    # Access the underlying GainPulsingRLEnv instance
    underlying_final_env = final_eval_vec_env.envs[0] 
    original_log_detail_param = underlying_final_env.agent.params["log_detailed_rl_env_steps"]

    for episode in range(num_eval_episodes):
        current_episode_history = []
        if episode == 0:
            # Modify the parameter in the underlying environment instance
            underlying_final_env.agent.params["log_detailed_rl_env_steps"] = True
            underlying_final_env.agent.log_detailed_rl_env_steps = True # Also update the direct attribute
            print(f"\n--- Detailed Log for Eval Episode {episode + 1} (Final Model) ---")
        
        obs = final_eval_vec_env.reset() # Reset the VecEnv
        done = [False] 
        total_episode_reward = 0; episode_pulse_starts = 0; was_pulsing_last_step = False

        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            next_obs, rewards, dones, infos = final_eval_vec_env.step(action) # Use the VecEnv
            current_reward = rewards[0]; done[0] = dones[0]; info = infos[0]
            obs = next_obs
            
            # Append to current episode's history from the *underlying env instance*
            current_episode_history.extend(underlying_final_env.history_for_render)
            underlying_final_env.history_for_render = []

            total_episode_reward += current_reward
            is_pulsing_now = info.get("is_pulsing", False)
            if is_pulsing_now and not was_pulsing_last_step: episode_pulse_starts += 1
            was_pulsing_last_step = is_pulsing_now

            if underlying_final_env.agent.log_detailed_rl_env_steps and episode == 0:
                 truncated = info.get("TimeLimit.truncated", False) or info.get("TimeLimitTruncation", False)
                 terminated = done[0] and not truncated
                 print(f"  SimStep: {info['current_sim_step']}, Action: {action[0]}, Reward: {current_reward:.3f}, Term: {terminated}, Trunc: {truncated}, Done: {done[0]}, F_crit: {info['F_crit']:.2f}, SM: {info['safety_margin']:.2f}")
        
        all_episode_rewards.append(total_episode_reward)
        all_survival_times.append(info.get("current_sim_step", 0))
        all_total_pulses.append(episode_pulse_starts)
        all_final_F_crit.append(info.get("F_crit",0))
        print(f"Eval Episode {episode + 1} (Final Model): Survival Time = {info.get('current_sim_step', 0)}, Reward = {total_episode_reward:.2f}, Pulses: {episode_pulse_starts}, F_crit End: {info.get('F_crit',0):.2f}")
        
        if episode == 0:
            df_ep_hist = pd.DataFrame(current_episode_history)
            plot_rl_episode_results(df_ep_hist, title=f"RL_Agent_Final_Eval_Episode_{episode+1}", save_dir=PLOT_SAVE_DIR)
            # Reset the parameter in the underlying environment instance
            underlying_final_env.agent.params["log_detailed_rl_env_steps"] = original_log_detail_param
            underlying_final_env.agent.log_detailed_rl_env_steps = original_log_detail_param
            print(f"--- End Detailed Log for Eval Episode {episode + 1} ---")


    print("\n--- Final Evaluation Summary (RL Agent) ---")
    print(f"Average Survival Time: {np.mean(all_survival_times):.2f} +/- {np.std(all_survival_times):.2f}")
    print(f"Average Total Reward: {np.mean(all_episode_rewards):.2f} +/- {np.std(all_episode_rewards):.2f}")
    print(f"Average Pulse Starts: {np.mean(all_total_pulses):.2f} +/- {np.std(all_total_pulses):.2f}")
    print(f"Average Final F_crit: {np.mean(all_final_F_crit):.2f} +/- {np.std(all_final_F_crit):.2f}")
    survival_rate = np.mean([1 if t >= SIM_DURATION else 0 for t in all_survival_times])
    print(f"Survival Rate (reached SIM_DURATION): {survival_rate*100:.1f}%")
    # For the baseline, also ensure you access the underlying env if it was wrapped
    print("\n--- Running Simplified 'HeurOFF' Baseline (RL Env with Action 0) ---")
    baseline_vec_env = DummyVecEnv([make_raw_env(0, seed=main_run_seed+300)]) # Use raw env for baseline too
    underlying_baseline_env = baseline_vec_env.envs[0]
    baseline_survival_times = []; baseline_pulse_counts = []; baseline_final_F_crit = []

    for episode in range(num_eval_episodes):
        current_episode_history_baseline = []
        obs = baseline_vec_env.reset(); done = [False]
        pulse_starts_baseline = 0; was_pulsing_baseline = False
        while not done[0]:
            action = [0]
            next_obs, rewards, dones, infos = baseline_vec_env.step(action)
            obs = next_obs; done[0] = dones[0]; info = infos[0]
            
            current_episode_history_baseline.extend(underlying_baseline_env.history_for_render)
            underlying_baseline_env.history_for_render = []
            
            is_pulsing_b_now = info.get("is_pulsing", False)
            if is_pulsing_b_now and not was_pulsing_baseline: pulse_starts_baseline +=1
            was_pulsing_baseline = is_pulsing_b_now
        baseline_survival_times.append(info.get("current_sim_step", 0))
        baseline_pulse_counts.append(pulse_starts_baseline)
        baseline_final_F_crit.append(info.get("F_crit",0))
        if episode == 0 :
             df_base_hist = pd.DataFrame(current_episode_history_baseline)
             plot_rl_episode_results(df_base_hist, title=f"Baseline_HeurOFF_Episode_{episode+1}", save_dir=PLOT_SAVE_DIR)

    print("\n--- Baseline 'HeurOFF' Summary (RL Env with Action 0) ---")
    print(f"Average Survival Time: {np.mean(baseline_survival_times):.2f} +/- {np.std(baseline_survival_times):.2f}")
    print(f"Average Final F_crit: {np.mean(baseline_final_F_crit):.2f} +/- {np.std(baseline_final_F_crit):.2f}")
    baseline_survival_rate = np.mean([1 if t >= SIM_DURATION else 0 for t in baseline_survival_times])
    print(f"Survival Rate (reached SIM_DURATION): {baseline_survival_rate*100:.1f}%")
    print(f"Average Pulse Starts: {np.mean(baseline_pulse_counts):.2f} +/- {np.std(baseline_pulse_counts):.2f}")

    print(f"\nAll logs and plots saved in: {LOG_DIR}")
    print("Script finished.")
    if plt.get_fignums(): plt.show(block=True)