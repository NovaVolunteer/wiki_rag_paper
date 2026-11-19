# Five-condition results

Generated 2026-11-18 by `analyze_conditions.py` from `results/<condition>.jsonl`. Answerer and judge models are recorded per row.

Completed rows per condition: C0 30/59, A-ret 0/59, A-full 0/59, B-nav 0/59, B-full 0/59


## Question class: all

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 30 | 90% | 8.93 | 8.13 | 26% | 67% | 0% | 0% | 0% | 121 (11) [102-151] | 0.0 |
| A-ret | 0 | not run | | | | | | | | | |
| A-full | 0 | not run | | | | | | | | | |
| B-nav | 0 | not run | | | | | | | | | |
| B-full | 0 | not run | | | | | | | | | |

## Question class: topic-page

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 30 | 90% | 8.93 | 8.13 | 26% | 67% | 0% | 0% | 0% | 121 (11) [102-151] | 0.0 |
| A-ret | 0 | not run | | | | | | | | | |
| A-full | 0 | not run | | | | | | | | | |
| B-nav | 0 | not run | | | | | | | | | |
| B-full | 0 | not run | | | | | | | | | |

## Question class: cross-page

| Condition | n | Answer rate | Score (answered) | Score (refusal=1) | Grounded (answered) | Fabrication | Justified ref. | Unjustified ref. | Gold fully in ctx | Input tok mean (sd) [min-max] | Units seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | not run | | | | | | | | | |
| A-ret | 0 | not run | | | | | | | | | |
| A-full | 0 | not run | | | | | | | | | |
| B-nav | 0 | not run | | | | | | | | | |
| B-full | 0 | not run | | | | | | | | | |

## Paired contrasts (95% bootstrap CI, questions resampled)

| Contrast | Class | n | Score diff (refusal=1) | Fabrication diff | Answer-rate diff |
|---|---|---|---|---|---|
