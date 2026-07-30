# ML Bias and Fairness

Week 14 of DS-3021. Two linked threads: (1) the technical definition of "bias" in ML and how it manifests unfairly when it reflects human/societal bias baked into training data, and (2) the evaluation machinery (confusion matrices, ROC/AUC) needed to actually measure fairness rather than just assert it.

## Bias is not inherently bad — until it is

The course is careful to start from Tom Mitchell's 1980 definition: bias is "any basis for choosing one generalization over another, other than strict consistency with the observed training instances." Every learning algorithm needs *some* bias to generalize beyond its training data — bias in this sense is a requirement, not a flaw. The problem is a specific case: when a model's bias reflects **human-generated societal bias** encoded into the training data (a link to *cognitive bias* / behavioral economics — humans don't decide rationally, and that irrationality shows up in the data we generate).

## Case studies used in lecture

- **Google Cloud Natural Language API (2017)** — sentiment analysis scored "I'm Christian" positively but "I'm a Jew" and "I'm a gay black woman" negatively, because it was trained on news/social media data. Google's public response is quoted directly in the slides: they don't always get it right.
- **Amazon's recruiting engine** — scored resumes 1-5 stars, trained on 10 years of hiring data that was mostly men, and systematically down-scored women. Amazon abandoned it once it couldn't be shown reliable.
- **YouTube recommendations (2011-12 shift)** — switched optimization from clicks to watch-time and brought in Google Brain for the recommender; optimizing purely for watch-time nudged viewers toward progressively more extreme content (jogging → ultramarathon, or political → extremist), not because the model was told to radicalize anyone but because that's what it found maximized the objective it was actually given.

The throughline across all three: the model did exactly what it was optimized to do — the failure was in the objective and the data, not a "bug."

## Evaluation machinery (prerequisite to fairness metrics)

Before the fairness formulas, the slides re-derive confusion matrices and ROC/AUC using a fraud-detection toy example (135 emails, 50% classification threshold):

- **True Positive Rate / Sensitivity** = TP/(TP+FN)
- **False Positive Rate** (1 − Specificity) = FP/(FP+TN)
- Threshold to 0 → everything flagged as fraud (maximum FPR, no FNs); threshold to 100 → nothing flagged (zero TPR) — the slides use these extremes to make the threshold/tradeoff concrete before introducing ROC curves (TPR on y-axis, FPR on x-axis) and AUC as the area under that curve.

This is the same confusion-matrix/sensitivity/specificity machinery introduced in [[course/knn]] for the imbalanced bank-marketing dataset — worth pointing students there-and-back for the worked numeric example if this page alone doesn't land it.

## Fairness metrics (built on the evaluation machinery above)

- **Demographic (proportional) Parity** — do protected subgroups get the same *proportion* of positive outcomes? Simple to compute but crude: it can hide the fact that false-positive/true-positive rates differ by group even when the raw proportions match.
- **Equality of Odds** — are TPR and FPR the same across protected groups, at every threshold? Effectively "run separate ROC curves per subgroup and compare."
- **Equal Opportunity** — a narrower version of Equality of Odds focused only on the positive class: is TPR (not FPR) equal across groups?

The practical note from lecture: satisfying any of these exactly by adjusting the decision threshold is often difficult or infeasible given real data, which is why some approaches instead reweight variables *before* model training rather than adjusting the threshold after.

## The broader ethics layer

A companion deck (`Value_1.pptx`) frames this within a "Data Science Value Domain": tensions between enterprise incentives and ethics, private interest and public good, accuracy and efficiency. It organizes interventions by *where in the pipeline* they happen — pre-processing (reweighting data before training), in-processing (favoring interpretable models over pure accuracy), post-processing (auditing how errors distribute after the fact) — and walks the same Amazon and YouTube cases as discussion exercises ("what went wrong, where in the DS lifecycle, how could it have been avoided").

## Related topics

- [[course/knn]] and [[course/decision-trees]] — both are classifiers whose confusion-matrix output is exactly what these fairness metrics operate on; class imbalance (flagged on both those pages) is a direct fairness concern, not just an accuracy one
- *(not yet ingested)* `fairlearn` library usage (`Fairness_Example.py` is currently just a two-line import stub — worth flagging to the instructor that the runnable fairness-metric example may be incomplete or was never filled in)

## Sources

- `raw/course/DS-3021/ml-bias/2024-10-29-ml-bias-and-fairness-slides.pdf` — main lecture slides
- `raw/course/DS-3021/ml-bias/2024-10-29-value-ethics-slides.pptx` — companion ethics/values deck
- `raw/course/DS-3021/ml-bias/2024-10-29-fairness-example.py` — fairlearn code stub (incomplete — see note above)
- `raw/course/DS-3021/ml-bias/2024-10-29-japan-credit-dataset.csv` — dataset referenced for a fairness exercise
