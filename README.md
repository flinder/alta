# alta
Active Learning Approaches for Labeling Text

# Simulation file naming conventions

For initial random/active simulations:

```
{mode}_simulation_data_{balance}.csv
```
- mode (active,random)
- balance (0.01, 0.05, 0.1, 0.3, 0.5)

For second iteration of active simulations:

```
{mode}_simulation_data_{algorithm}_{balance}_icr_{icr}_rand_{percent_random}.csv
```

- mode (active,random)
- algorithm (committee, margin)
- balance (0.01, 0.05, 0.1, 0.3, 0.5)
- percent_random, optional (0.0, 0.5, 0.75)
- icr, optional (0.7, 0.8, 0.85, .9, .95)