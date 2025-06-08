# Adaptive Lever Management: Heuristic Pulsing Simulation

This repository contains the Python code for the **heuristic gain-pulsing agent** simulation, which serves as a key case study in Chapter 9, "Adaptive Lever Management: From Heuristic Limitations to Learned Resilience," of the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness*.

**Note:** This code implements the *heuristic* (rule-based) agent and its performance evaluation. The Reinforcement Learning (RL) agent discussed in the same chapter is part of a separate study and repository.

## About the Experiment

This simulation investigates the effectiveness and limitations of a simple, rule-based heuristic for managing a system's adaptive capacity. The core components are:

*   **An Agent:** An entity that must survive in a challenging environment. Its viability is determined by its **Tolerance Sheet** ($\Theta_T$), a concept from Threshold Dialectics representing its capacity to withstand stress. This capacity is a function of its adaptive levers: perception gain ("g"), policy precision ("beta"), and energetic slack ("F_crit").
*   **An Environment:** A system that generates periodic, high-magnitude **strain shocks**, which challenge the agent's viability.
*   **The Heuristic Intervention:** The agent's primary defense mechanism is to "pulse" its perception gain ("g_lever"). The decision to trigger this pulse is based on a fixed rule: if its safety margin ($G = \Theta_T - \text{strain}$) falls below a predefined **Safety Margin Critical Threshold (SMCT)**, it initiates a high-gain pulse to increase its tolerance.

The central question explored here is: **How robust is such a simple, threshold-based heuristic?** The experiment demonstrates the concept of "heuristic brittleness"—a phenomenon where a small change in a hand-tuned parameter (like SMCT) can lead to a catastrophic drop in performance.

## How to Run the Code

The simulation is contained in a single Python script.

### 1. Requirements

You will need Python 3 and the following libraries:

*   "numpy"
*   "pandas"
*   "matplotlib"

You can install them using pip:
"""bash
pip install numpy pandas matplotlib
"""

### 2. Execution

Simply run the Python script from your terminal:

"""bash
python adaptive_lever_management_heuristic.py
"""
*(Assuming you save the provided code as "adaptive_lever_management_heuristic.py")*

The script will run a series of Monte-Carlo simulations for different scenarios defined in the code. It will print a summary of survival rates for each scenario to the console and generate several output files.

## Outputs

Upon completion, the script will generate the following files in the same directory:

*   **"all_scenarios_summary.csv"**: A CSV file with detailed summary statistics for every scenario tested (survival rate, pulse counts, final "F_crit" levels, etc.).
*   **"all_scenarios_summary.txt"**: A human-readable text file mirroring the CSV data, including a header with the global simulation parameters.
*   **"smct_sweep_summary.txt"**: A focused summary of the SMCT sweep scenarios.
*   **"sweep_survival_vs_smct.png"**: A plot visualizing the "cliff-edge" effect, showing survival rate as a function of the SMCT.
*   **"sweep_pulse_count_vs_smct.png"**: A plot of the average number of pulses used vs. the SMCT.
*   **"sweep_fcrit_survivors_vs_smct.png"**: A plot of the average final energetic slack ("F_crit") for surviving runs vs. the SMCT.

## Key Findings & Results

The primary finding of this simulation is the **"cliff-edge" brittleness** of the heuristic policy. While the agent performs well for a certain range of SMCT values, a small increase beyond a critical point causes a catastrophic failure in survival rates.

This is demonstrated in the summary output, which shows the survival rate plummeting from **92-94%** to **34%** and then to **0%** as the SMCT multiplier increases from 0.45 to 0.55.

### Sample Output ("all_scenarios_summary.txt")

"""
# Global simulation parameters
{
  "duration": 500,
  "baseline_strain": 2.4,
  "shock_factor": 3.2,
  ...
}

Scenario                                   n  surv%  phi1  k_g  heuristic smct normG  pulses±SD  t_surv±SD  totPulse±SD  indivDur±SD  Fsurv±SD  Fbreach±SD  initF±SD
SMCTSweep_Baseline_HeurOFF                 50  94.00 0.30 0.020 OFF        nan       22.20±1.65 498.3±6.9  327.8±24.1 14.76±1.46 163.0±0.2  163.9±0.1  181.0±0.0
SMCTSweep_x0.45_AbsGainHeurON              50  92.00 0.30 0.020 AbsGain    3.6       22.20±1.65 498.0±7.0  327.5±23.8 14.75±1.51 163.0±0.2  163.7±0.3  181.0±0.0
SMCTSweep_x0.50_AbsGainHeurON              50  34.00 0.30 0.020 AbsGain    4.0       16.60±4.87 402.3±86.0 238.0±76.2 14.34±2.87 163.0±0.2  168.6±2.3  181.0±0.0
SMCTSweep_x0.55_AbsGainHeurON              50   0.00 0.30 0.020 AbsGain    4.4       2.00±0.00  165.3±25.8 16.0±0.0   8.00±7.00  nan±nan   175.7±0.8  181.0±0.0
...
"""
This result powerfully motivates the need for more sophisticated, adaptive control strategies (like Reinforcement Learning) that are less dependent on such precise, hand-tuned parameters.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.