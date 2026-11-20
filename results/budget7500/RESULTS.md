# Five-condition results

Generated 2026-11-19 by `analyze_conditions.py` from `results/<condition>.jsonl`. Answerer and judge models are recorded per row.

Completed rows per condition: C0 0/100, A-ret 100/100, A-full 0/100, B-nav 100/100, B-full 0/100


## Question class: all

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Flagged ungrounded | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | not run | | | | | | | | | | |
| A-ret | 100 | 99% | 9.37 | 9.29 | 83% | 1% | 16% | 1% | 0% | 63% | 9,596 (1,973) [4,949-14,028] | 5.0 |
| A-full | 0 | not run | | | | | | | | | | |
| B-nav | 100 | 98% | 9.52 | 9.35 | 92% | 0% | 8% | 1% | 1% | 76% | 10,264 (1,751) [3,775-11,665] | 2.9 |
| B-full | 0 | not run | | | | | | | | | | |

## Question class: topic-page

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Flagged ungrounded | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | not run | | | | | | | | | | |
| A-ret | 45 | 100% | 9.69 | 9.69 | 91% | 0% | 9% | 0% | 0% | 96% | 9,161 (1,854) [4,949-13,664] | 5.0 |
| A-full | 0 | not run | | | | | | | | | | |
| B-nav | 45 | 96% | 9.63 | 9.24 | 88% | 0% | 11% | 2% | 2% | 93% | 9,625 (2,238) [3,775-11,665] | 2.6 |
| B-full | 0 | not run | | | | | | | | | | |

## Question class: cross-page

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Flagged ungrounded | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | not run | | | | | | | | | | |
| A-ret | 45 | 100% | 9.00 | 9.00 | 73% | 2% | 24% | 0% | 0% | 44% | 10,097 (2,121) [5,662-14,028] | 5.0 |
| A-full | 0 | not run | | | | | | | | | | |
| B-nav | 45 | 100% | 9.33 | 9.33 | 93% | 0% | 7% | 0% | 0% | 76% | 10,970 (490) [9,173-11,584] | 3.1 |
| B-full | 0 | not run | | | | | | | | | | |

## Question class: unanswerable

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Flagged ungrounded | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | not run | | | | | | | | | | |
| A-ret | 10 | 90% | 9.67 | 8.80 | 89% | 0% | 10% | 10% | 0% | 0% | 9,302 (1,066) [7,850-11,501] | 5.0 |
| A-full | 0 | not run | | | | | | | | | | |
| B-nav | 10 | 100% | 9.90 | 9.90 | 100% | 0% | 0% | 0% | 0% | 0% | 9,954 (1,687) [6,111-11,376] | 2.8 |
| B-full | 0 | not run | | | | | | | | | | |

## Paired contrasts (95% bootstrap CI, questions resampled)

| Contrast | Class | n | Score diff (refusal=1) | Fabrication diff | Answer-rate diff |
|---|---|---|---|---|---|
| B-nav - A-ret | all | 100 | +0.06 [-0.33, +0.43] | -0.01 [-0.03, +0.00] | -0.01 [-0.05, +0.02] |
| B-nav - A-ret | topic-page | 45 | -0.44 [-1.11, +0.09] | +0.00 [+0.00, +0.00] | -0.04 [-0.11, +0.00] |
| B-nav - A-ret | cross-page | 45 | +0.33 [-0.09, +0.76] | -0.02 [-0.07, +0.00] | +0.00 [+0.00, +0.00] |
| B-nav - A-ret | unanswerable | 10 | +1.10 [-0.10, +3.00] | +0.00 [+0.00, +0.00] | +0.10 [+0.00, +0.30] |
