# Validation against the official code

Official code: https://github.com/AI4LIFE-GROUP/fair-unlearning.git at commit `8d7a019de3e61180d1c881ceff21072422f4cd84`.
Result: **45 / 45 checks passed** (plus 33 informational rows).

Identical inputs (split, noise vector b, deleted rows, starting parameters) were given to the official functions and to ours. Adult and HSLS use a random subset of training rows (4000 and 2500) because the official fairness term needs ~10 GB of memory on the full data; COMPAS uses all rows. See `validation/compare_official.py` for details.

| Dataset | Check | Quantity | Value | Tolerance | Result | Note |
|---|---|---|---|---|---|---|
| compas | objective | vs. official (float32 fairness sum) | 1.536e-07 | 1e-03 | PASS |  |
| compas | objective | vs. official formula in float64 | 1.414e-16 | 1e-06 | PASS |  |
| compas | training | fair: (our objective - official) / |official| | -2.959e-07 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| compas | training | fair: gradient norm, ours | 6.086e-09 | — | info | info: 0 at the exact optimum |
| compas | training | fair: gradient norm, official | 9.349e-06 | — | info | info: 0 at the exact optimum |
| compas | training | fair: test prediction agreement | 9.991e-01 | — | info | info |
| compas | training | bce: (our objective - official) / |official| | -1.316e-04 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| compas | training | bce: gradient norm, ours | 1.257e-09 | — | info | info: 0 at the exact optimum |
| compas | training | bce: gradient norm, official | 1.282e-03 | — | info | info: 0 at the exact optimum |
| compas | training | bce: test prediction agreement | 9.867e-01 | — | info | info |
| compas | fair step | transcription (as released) vs. official fn | 6.987e-04 | 1e-03 | PASS | official sums in float32 |
| compas | fair step | transcription ((n-k) lambda) vs. ours | 8.771e-06 | 1e-04 | PASS |  |
| compas | fair step | distance to retrained model, ours | 7.814e-03 | — | info | info: lower is better |
| compas | fair step | distance to retrained model, official | 1.426e-02 | — | info | info: lower is better |
| compas | newton step | transcription (as released) vs. official fn | 0.000e+00 | 1e-06 | PASS |  |
| compas | newton step | transcription (fixed) vs. ours | 8.349e-07 | 1e-04 | PASS |  |
| compas | newton step | distance to retrained model, ours | 6.234e-03 | — | info | info: lower is better |
| compas | newton step | distance to retrained model, official | 3.583e-02 | — | info | info: lower is better |
| compas | PRU | relative difference of update (bug fixed) | 7.538e-15 | 1e-06 | PASS |  |
| compas | PRU | official as released vs. fixed | 2.569e-01 | — | info | info: effect of the official row-index bug |
| compas | SISA vote | share of differing predictions | 0.000e+00 | 0e+00 | PASS |  |
| compas | metrics | accuracy: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| compas | metrics | aeod: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| compas | metrics | dp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| compas | metrics | eopp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| compas | certificate | relative difference of eps' (data) | 4.057e-16 | 1e-06 | PASS |  |
| adult | objective | vs. official (float32 fairness sum) | 1.664e-06 | 1e-03 | PASS |  |
| adult | objective | vs. official formula in float64 | 0.000e+00 | 1e-06 | PASS |  |
| adult | training | fair: (our objective - official) / |official| | -1.372e-02 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| adult | training | fair: gradient norm, ours | 1.250e-08 | — | info | info: 0 at the exact optimum |
| adult | training | fair: gradient norm, official | 5.885e-03 | — | info | info: 0 at the exact optimum |
| adult | training | fair: test prediction agreement | 9.778e-01 | — | info | info |
| adult | training | bce: (our objective - official) / |official| | -3.385e-04 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| adult | training | bce: gradient norm, ours | 5.788e-09 | — | info | info: 0 at the exact optimum |
| adult | training | bce: gradient norm, official | 3.151e-04 | — | info | info: 0 at the exact optimum |
| adult | training | bce: test prediction agreement | 9.938e-01 | — | info | info |
| adult | fair step | transcription (as released) vs. official fn | 1.419e-04 | 1e-03 | PASS | official sums in float32 |
| adult | fair step | transcription ((n-k) lambda) vs. ours | 3.122e-06 | 1e-04 | PASS |  |
| adult | fair step | distance to retrained model, ours | 3.048e-03 | — | info | info: lower is better |
| adult | fair step | distance to retrained model, official | 1.692e-02 | — | info | info: lower is better |
| adult | newton step | transcription (as released) vs. official fn | 0.000e+00 | 1e-06 | PASS |  |
| adult | newton step | transcription (fixed) vs. ours | 1.180e-06 | 1e-04 | PASS |  |
| adult | newton step | distance to retrained model, ours | 3.803e-03 | — | info | info: lower is better |
| adult | newton step | distance to retrained model, official | 2.142e-01 | — | info | info: lower is better |
| adult | PRU | relative difference of update (bug fixed) | 1.646e-12 | 1e-06 | PASS |  |
| adult | PRU | official as released vs. fixed | 1.228e+00 | — | info | info: effect of the official row-index bug |
| adult | SISA vote | share of differing predictions | 0.000e+00 | 0e+00 | PASS |  |
| adult | metrics | accuracy: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| adult | metrics | aeod: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| adult | metrics | dp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| adult | metrics | eopp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| adult | certificate | relative difference of eps' (data) | 3.243e-16 | 1e-06 | PASS |  |
| hsls | objective | vs. official (float32 fairness sum) | 6.211e-04 | 1e-03 | PASS |  |
| hsls | objective | vs. official formula in float64 | 2.144e-15 | 1e-06 | PASS |  |
| hsls | training | fair: (our objective - official) / |official| | -7.949e-08 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| hsls | training | fair: gradient norm, ours | 3.450e-08 | — | info | info: 0 at the exact optimum |
| hsls | training | fair: gradient norm, official | 4.828e-05 | — | info | info: 0 at the exact optimum |
| hsls | training | fair: test prediction agreement | 9.997e-01 | — | info | info |
| hsls | training | bce: (our objective - official) / |official| | -7.074e-12 | 1e-09 | PASS | <= 0: ours reaches an equal or lower loss |
| hsls | training | bce: gradient norm, ours | 7.931e-09 | — | info | info: 0 at the exact optimum |
| hsls | training | bce: gradient norm, official | 1.371e-06 | — | info | info: 0 at the exact optimum |
| hsls | training | bce: test prediction agreement | 1.000e+00 | — | info | info |
| hsls | fair step | transcription (as released) vs. official fn | 2.418e-05 | 1e-03 | PASS | official sums in float32 |
| hsls | fair step | transcription ((n-k) lambda) vs. ours | 1.576e-06 | 1e-04 | PASS |  |
| hsls | fair step | distance to retrained model, ours | 4.122e-06 | — | info | info: lower is better |
| hsls | fair step | distance to retrained model, official | 3.724e-03 | — | info | info: lower is better |
| hsls | newton step | transcription (as released) vs. official fn | 0.000e+00 | 1e-06 | PASS |  |
| hsls | newton step | transcription (fixed) vs. ours | 3.265e-07 | 1e-04 | PASS |  |
| hsls | newton step | distance to retrained model, ours | 3.782e-06 | — | info | info: lower is better |
| hsls | newton step | distance to retrained model, official | 3.431e-03 | — | info | info: lower is better |
| hsls | PRU | relative difference of update (bug fixed) | 5.520e-14 | 1e-06 | PASS |  |
| hsls | PRU | official as released vs. fixed | 4.360e-01 | — | info | info: effect of the official row-index bug |
| hsls | SISA vote | share of differing predictions | 0.000e+00 | 0e+00 | PASS |  |
| hsls | metrics | accuracy: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| hsls | metrics | aeod: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| hsls | metrics | dp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| hsls | metrics | eopp: |ours - aif360| | 0.000e+00 | 1e-12 | PASS |  |
| hsls | certificate | relative difference of eps' (data) | 0.000e+00 | 1e-06 | PASS |  |

## Differences found in the official code

1. **Hessian L2 term.** `fair_unlearning` and `log_newton` add `n * reg * I`; the Hessian of the loss on the remaining data (paper Eq. 8 and Lemma 1) has `(n - k) * reg * I`.
2. **Intercept counted twice.** `log_newton` computes the Hessian weights from `X_inflated @ theta + theta[-1]`, but `X_inflated` already has the intercept column.
3. **PRU rows.** `lin_res` builds its gradient from `X[i]` (the first k training rows) instead of `X[ind[i]]` (the deleted rows).
4. **Precision and convergence.** The fairness sum is accumulated in float32, and training stops after 20 L-BFGS steps without a line search, which does not always reach the optimum (see the gradient norms above).
5. **Certificate.** `eps_delta_experiments.py` divides by sigma^2; Lemma 2 gives epsilon proportional to 1/sigma (same at sigma = 1).

Our implementation follows the paper's equations; the distance rows show it lands closer to the retrained model than the official updates.
