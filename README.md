OEA_sem/

│

├── README.md

│

├── requirements.txt

│

├── data/

│   ├── raw/

│   │   └── A/
│   │       ├── problem_1.vrp
│   │       ├── problem_1.sol
│   │       └── ...
│   │
│   └── processed/
│       └── cached_instances/
│
├── src/
│   ├── core/
│   │   ├── instance_loader.py
│   │   ├── dvrp_simulator.py
│   │   ├── solution_representation.py
│   │   └── evaluation.py
│   │
│   ├── ea/
│   │   ├── population.py
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   ├── mutation.py
│   │   └── elitism.py
│   │
│   ├── frameworks/
│   │   ├── vanilla_ea/
│   │   │   ├── config.yaml
│   │   │   └── run.py
│   │   │
│   │   ├── rl_param_optimizer/
│   │   │   ├── env.py
│   │   │   ├── agent.py
│   │   │   ├── config.yaml
│   │   │   └── run.py
│   │   │
│   │   └── conv_param_optimizer/
│   │       ├── optimizer.py
│   │       ├── config.yaml
│   │       └── run.py
│   │
│   └── utils/
│       ├── logging.py
│       ├── seeding.py
│       └── metrics.py
│
├── experiments/
│   ├── exp_01_vanilla/
│   ├── exp_02_rl/
│   └── exp_03_conv/
│
├── results/
│   ├── tables/
│   ├── plots/
│   └── summaries/
│
└── report/
    ├── methodology.md
    ├── results.md
    └── discussion.md
