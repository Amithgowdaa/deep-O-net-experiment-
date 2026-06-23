# DeepONet — SciML Experiments

Personal research repo for learning and extending DeepONet (Lu et al. 2021).

## Structure
```
configs/          # YAML hyperparameter configs per experiment
data/generators/  # Data generation (GRF sampling, numerical integration)
models/           # Branch net, Trunk net, DeepONet assembly
training/         # Loss functions, training loop
experiments/      # One folder per experiment, each has run.py
utils/            # Checkpointing, GitHub push, visualization
notebooks/        # Thin Kaggle runner notebooks
```

## Experiments
| Experiment | Operator | Status |
|---|---|---|
| antiderivative | G(u)(x) = ∫₀ˣ u(t)dt | 🔄 in progress |



## References
- Lu et al. 2021 — Learning nonlinear operators via DeepONet
- Wang et al. 2022 — PI-DeepONet
