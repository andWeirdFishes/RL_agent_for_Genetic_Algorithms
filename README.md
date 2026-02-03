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
