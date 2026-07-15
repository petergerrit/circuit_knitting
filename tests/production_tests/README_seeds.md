# Seed Parameters Matrix and Test Descriptions

> **Note**: This test scheme is implemented in both `test_step1_seeds.py` (128 shots, first Trotter step) and `test_step2_seeds.py` (256 shots, second Trotter step).

## Seed Parameters Matrix (runs 1-5)

```
+-----+----------------+-----------------+-----------------+
| Run | simulator_seed | transpiler_seed | bootstrap_seed  |
+-----+----------------+-----------------+-----------------+
|  1  |      42        |       42        |       42        |
+-----+----------------+-----------------+-----------------+
|  2  |      42        |       42        |       42        |
+-----+----------------+-----------------+-----------------+
|  3  |      42        |       42        |       123       |
+-----+----------------+-----------------+-----------------+
|  4  |      123       |       42        |       42        |
+-----+----------------+-----------------+-----------------+
|  5  |      42        |       123       |       42        |
+-----+----------------+-----------------+-----------------+
```

## Test Descriptions

- **1 == 2**: Identical seeds (all 42) -> fermion number and bootstrap error must match. Tests deterministic behavior.

- **1 != 3**: Only bootstrap_seed differs (42 to 123) -> fermion number must match, bootstrap error must differ. Tests that bootstrap seed only affects error estimation (resampling), not the underlying counts.

- **1 != 4**: Only simulator_seed differs (42 to 123) -> fermion number must differ. Tests that simulator randomness affects measurement outcomes.

- **1 != 5**: Only transpiler_seed differs (42 to 123) -> fermion number must differ. Tests that transpiler randomness (circuit compilation) affects final results.

- **4 != 5**: simulator_seed (123 vs 42) and transpiler_seed (42 vs 123) both differ -> fermion number must differ. Tests combined effect of simulator and transpiler seed changes without computing bootstrap error.
