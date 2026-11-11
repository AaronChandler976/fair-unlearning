# Consolidated Fair Machine Unlearning Results

Means are calculated across the tested deletion scenarios. Group deletion averages include both protected groups. Runtime statistics summarize single-run measurements and should not be interpreted as repeated benchmark estimates.

| Dataset | Deletion strategy | Scenarios | Baseline accuracy | Baseline AEOD | Newton accuracy (mean) | Retrained accuracy (mean) | Newton AEOD (mean) | Retrained AEOD (mean) | Parameter distance (mean) | Newton runtime (median s) | Retraining runtime (median s) | Speedup (median x) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adult | Random deletion | 3 | 0.8446 | 0.0602 | 0.8442 | 0.8442 | 0.0600 | 0.0601 | 0.1255 | 0.6235 | 6.5463 | 10.4997 |
| COMPAS | Random deletion | 3 | 0.6525 | 0.1469 | 0.6544 | 0.6544 | 0.1413 | 0.1413 | 0.0001 | 0.0225 | 0.0731 | 4.1708 |
| HSLS | Random deletion | 3 | 0.8034 | 0.0389 | 0.8034 | 0.8034 | 0.0389 | 0.0389 | 0.0149 | 0.0812 | 1.0815 | 16.1056 |
| Adult | Group deletion | 6 | 0.8446 | 0.0602 | 0.8444 | 0.8443 | 0.0593 | 0.0593 | 0.1057 | 0.6813 | 5.9842 | 8.4034 |
| COMPAS | Group deletion | 6 | 0.6525 | 0.1469 | 0.6534 | 0.6534 | 0.1461 | 0.1461 | 0.0002 | 0.0316 | 0.1472 | 4.9555 |
| HSLS | Group deletion | 6 | 0.8034 | 0.0389 | 0.8033 | 0.8033 | 0.0391 | 0.0391 | 0.0181 | 0.0634 | 0.7314 | 12.2139 |
