# Model Evaluation Metrics

Week 7 of DS-3021. Once a classifier exists — a [[course/knn]] model, say — the next question is whether it's actually any good, and "accuracy" alone is not a reliable answer. This page covers the confusion-matrix-derived metrics (accuracy, precision, recall, specificity, F1), ROC/AUC, and log loss, and how they get used together rather than in isolation.

## Why accuracy alone can mislead

The lecture makes this concrete with a plain two-class scatter: 28 of 30 points are class `a`, 2 are class `b`. A model that predicts `a` for everything scores 93% accuracy while having learned nothing about class `b` at all. This isn't a contrived edge case for this course specifically — the in-class KNN example (bank marketing data, target "signed up") has only ~11.6% positive cases, the same imbalance flagged on [[course/knn]], which is exactly the setting where accuracy is most likely to lie. The fix isn't a different single number; it's looking at where the errors are concentrated, which is what the confusion matrix is for.

## The confusion matrix

Four outcomes for a binary classifier, illustrated in lecture with a shepherd-and-wolf framing: **True Positive** (wolf real, shepherd says wolf — hero), **False Positive** (no wolf, shepherd says wolf — false alarm), **False Negative** (wolf real, shepherd says no wolf — the costly miss), **True Negative** (no wolf, shepherd says no wolf — quiet night). Accuracy is `(TP+TN)/(TP+TN+FP+FN)` — i.e. accuracy is the confusion matrix collapsed into one number, and unpacking that number back into its four parts is how you catch a model that's only good at the majority class.

The slides work a running fraud/spam-detection example (135 emails, 50% classification threshold) that's worth knowing because [[course/ml-bias]] later re-derives this exact same example when introducing fairness metrics — it's the connective tissue between "how do I evaluate a model" (week 7) and "how do I evaluate a model *fairly*" (week 14). At the default threshold: TP=10, FN=22, FP=7, TN=96. Pushing the threshold to the extremes makes the tradeoff concrete: threshold 0 flags every email as fraud (TP=32, FN=0, FP=103, TN=0 — no missed fraud, but every legitimate email also blocked); threshold 100 flags nothing (TP=0, FN=32, FP=0, TN=103 — no false alarms, but every actual fraud gets through).

## Precision, recall, specificity, and F1

Four more metrics fall directly out of the confusion matrix, and which one matters depends on which error type is expensive:

- **Precision** = TP/(TP+FP) — of everything flagged positive, how much was actually positive? Low precision means the model cries wolf too often.
- **Recall / Sensitivity** = TP/(TP+FN) — of everything actually positive, how much did the model catch? Low recall means real positives are slipping through. This is the same "sensitivity" referenced on [[course/knn]]'s R→Python table.
- **Specificity** = TN/(TN+FP) — the negative-class mirror of recall.
- **F1** = harmonic mean of precision and recall (`2 × P×R / (P+R)`). The lecture is explicit about when to prefer it over accuracy: accuracy treats TP and TN as equally important, F1 is for when false positives and false negatives are the actual concern — which in practice means F1 is the better default on imbalanced data, exactly the bank-marketing situation above.
- **Balanced accuracy** = (sensitivity + specificity)/2 is offered as a middle ground when you want one number but the classes are imbalanced.

For more than two classes, the slides also flag **Cohen's Kappa** — how much better the classifier does than random guessing weighted by class frequency — as the metric that generalizes better to multi-class problems, since precision/recall/F1 are fundamentally binary constructs.

## ROC and AUC: performance across every threshold

Precision, recall, and F1 all depend on wherever the classification threshold happens to sit (0.5 by default in sklearn). The **ROC curve** (Receiver Operating Characteristic) removes that dependency by plotting **True Positive Rate** (TPR = sensitivity) against **False Positive Rate** (FPR = 1 − specificity) at every possible threshold — sweeping the threshold from 1 down to 0 traces the curve. In the 135-email example above, the default-threshold point sits at TPR=10/32=.31, FPR=7/103=.06; the two extreme thresholds worked out earlier are the curve's endpoints, (0,0) and (1,1). The diagonal line is what a model with no discriminative power (random guessing) traces out — a useful curve bows toward the top-left corner, away from that diagonal.

**AUC** (Area Under the [ROC] Curve) collapses the whole curve into one number: the probability the model ranks a random positive example above a random negative one. The course gives a concrete scale for reading it: 0.9–1.0 excellent, 0.8–0.9 good, 0.7–0.8 fair, 0.6–0.7 poor, 0.5–0.6 fail (0.5 = the random-guess diagonal). Because AUC is threshold-independent, it's the metric to reach for when comparing two models overall, before deciding where to set the operating threshold for either one.

## Log loss: penalizing confident wrong answers

Every metric above is computed from *hard* class labels (predicted 0 or 1), which throws away how confident the model was. **Log loss** (cross-entropy loss) scores the predicted *probabilities* directly against the true labels: `logLoss = -1/N × Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]`. The shape of that formula is the whole point — it's not linear, it's punishing. A worked table in the source material makes this concrete: a correct prediction made with high confidence (actual=1, predicted p=0.99) costs almost nothing (loss ≈ 0.01), but a *wrong* prediction made with equally high confidence (actual=0, predicted p=0.9) costs 230× more (loss ≈ 2.30) — far worse than simply being unsure (actual=0, predicted p=0.5, loss ≈ 0.69). The course's rule of thumb: a log loss near 1 roughly corresponds to a model that's typically giving less than ~40% probability to the correct class, and the *prevalence* (baseline positive rate) matters for judging whether a given log loss is actually good, the same way accuracy needs a baseline to be meaningful.

## Worked example: the bank-marketing KNN model, again

The course's eval-metrics notebook runs the exact same KNN pipeline as [[course/knn]] (bank marketing data, MinMax-scaled numeric features, one-hot encoded categoricals, `k=5`) and reports every metric above on the tune set, which makes it the cleanest single illustration of why none of these numbers should be read alone:

| Metric | Value | Reading it |
|---|---|---|
| Confusion matrix | TN=5678, FP=105, FN=630, TP=131 | 761 actual positives out of 6544 (11.6% prevalence — matches the imbalance noted on [[course/knn]]) |
| Accuracy | 88.8% | Looks strong, but... |
| Precision (positive class) | 0.56 | Of the customers flagged as likely sign-ups, just over half actually signed up |
| Recall (positive class) | 0.17 | ...the model only *catches* 17% of the customers who actually signed up |
| F1 | 0.26 | The precision/recall imbalance shows up clearly once combined |
| AUC | 0.79 | "Fair" on the course's own scale — genuine discriminative ability, just not being exploited at the default threshold |
| Log loss | 0.84 | Consistent with a model that's often not very confident even when it's right |

88.8% accuracy and 0.79 AUC describe a model with real signal; 17% recall describes a model that, at the default 0.5 threshold, misses five out of six people who would actually sign up. That gap is the whole argument for reporting the confusion matrix and multiple metrics rather than a single accuracy figure, and it's why the lab code includes a threshold-adjustment helper (`adjust_thres()`) to recompute the confusion matrix at thresholds other than 0.5 — precision and recall trade off against each other as the threshold moves, and which tradeoff is worth making is a business question, not a modeling one.

## Evaluating within cross-validation, not just a single split

The week 7 lab computes all of the above on a single train/tune split, which is the right first step pedagogically but understates the point made later on [[course/knn]]: the current KNN code was rewritten around **nested cross-validation** specifically so that performance estimates (and the metrics used to produce them) aren't inflated by tuning and evaluating on the same data. These evaluation metrics are what gets computed inside each fold of that outer loop — cross-validation doesn't replace the confusion matrix, F1, or AUC, it's the harness that makes a single reported value trustworthy.

## R → Python differences

Same conversion-in-progress pattern as [[course/knn]] and [[course/decision-trees]], scoped here to the evaluation-specific calls:

| R | Python | Note |
|---|---|---|
| `confusionMatrix()` | `metrics.confusion_matrix()` | R's version gives much more detail by default; Python needs several separate calls to match it |
| (same) | `metrics.ConfusionMatrixDisplay.from_predictions()` | Adds the visual, human-readable version |
| `confusionMatrix()` | `metrics.classification_report()` | Closest Python equivalent to R's fuller default output (precision/recall/F1 per class) |
| `prediction()` + `performance()` (ROCR package) | `metrics.RocCurveDisplay.from_predictions()` | R needs two functions — one to build the prediction object, one to extract TPR/FPR/AUC from it; Python does it in one call |
| (same) | `metrics.roc_auc_score()` | Extracts just the AUC value |
| `LogLoss()` (MLmetrics package) | `metrics.log_loss()` | |
| `F1_Score()` (MLmetrics package) | `metrics.f1_score()` | |

## Related topics

- [[course/knn]] — the classifier these metrics are demonstrated on; same dataset, same imbalance, and the destination for the nested cross-validation rewrite referenced above
- [[course/decision-trees]] — a different classifier that these same metrics apply to unchanged; also imbalance-sensitive (recommends balancing the dataset before fitting)
- [[course/ml-bias]] — reuses this page's exact 135-email confusion-matrix/ROC/AUC example as the prerequisite machinery for its fairness metrics (Demographic Parity, Equality of Odds, Equal Opportunity); read this page first if that one doesn't fully land

## Sources

- `raw/course/DS-3021/ml-eval-metrics/2025-06-09-perfmet-slides.pdf` — main lecture slides ("Practice of Data Science" deck; the confusion matrix / ROC / AUC / log loss / Kappa content is on slides 5-23, also reused in week 14's ml-bias lecture — see [[course/ml-bias]])
- `raw/course/DS-3021/ml-eval-metrics/2025-06-09-model-metrics-lab.md` — lab assignment (mushroom dataset, KNN classification, requires accuracy/TPR/FPR/F1/cross-entropy/ROC-AUC)
- `raw/course/DS-3021/ml-eval-metrics/2025-06-09-knn-eval-example-notebook.ipynb` — in-class worked example notebook with executed outputs (bank-marketing KNN model; source of the worked-example table above)
- `raw/course/DS-3021/ml-eval-metrics/2025-06-09-logloss-worked-example.xlsx` — small worked table of per-row log loss contributions (source of the confident-right vs. confident-wrong comparison above)
- `raw/course/DS-3021/ml-eval-metrics/2022-07-22-eval-metrics-r-to-python-conversion.docx` — R→Python conversion notes for evaluation-metric function calls
