# Project Structure

```text
.
├───data
├───experiments
├───report
│       analysis_of_results.ipynb
│
├───results
│   ├───plots
│   │   └───A-n38-k5
│   │           comparison.html
│   │           comparison.png
│   │
│   └───summaries
│           rl_ea_summary.csv
│           vanilla_ea_detailed.csv
│           vanilla_ea_summary.csv
│
├───scripts
│       cache_instances.py
│       run_exp01.py
│       run_exp02.py
│       save.py
│       train_rl.py
│
├───src
│   │   roots.py
│   │
│   ├───core
│   │       instance_loader.py
│   │       solution_representation.py
│   │
│   ├───ea
│   │       crossover.py
│   │       evolutionary_algorithm.py
│   │       mutation.py
│   │       population.py
│   │       selection.py
│   │
│   ├───frameworks
│   │   ├───rl_param_optimizer
│   │   │       agent.py
│   │   │       config_rl.yaml
│   │   │       env.py
│   │   │
│   │   └───vanilla_ea
│   │           config.yaml
│   │
│   └───utils
│           find_best.py
│           logging.py
│           metrics.py
│           seeding.py
│           visualization.py
│
└───tests
        test_instance_loader.ipynb
        test_metrics_and_solrep.ipynb
```
Key File Descriptions
Configuration & Data (.yaml, .csv)
config.yaml / config_rl.yaml: Define hyperparameters for the Vanilla EA and Reinforcement Learning agent respectively.

rl_ea_summary.csv: Aggregated performance metrics for the RL-tuned Evolutionary Algorithm.

vanilla_ea_summary.csv: Aggregated baseline performance metrics for the standard EA.

vanilla_ea_detailed.csv: Raw per-generation data for the baseline experiments.

Source Code (.py)
src/core/: Logic for loading VRP instances and representing solutions.

src/ea/: Genetic operators (crossover, mutation, selection) and the main EA loop.

src/frameworks/: Specific implementations of optimization strategies (RL vs. Vanilla).

src/utils/: Helper modules for distance metrics, seeding, and Plotly visualization.

Notebooks (.ipynb)
report/analysis_of_results.ipynb: Final statistical analysis and comparison of optimization runs.

tests/: Unit tests for verifying instance parsing and cost calculation logic.
