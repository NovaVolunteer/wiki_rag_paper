# Ensemble Methods: Bagging & Random Forest

Week 12 of DS-3021. A single [[course/decision-trees]] is unstable and greedy: small changes in the training data can produce a completely different tree, and an unconstrained tree overfits by splitting until it memorizes the training set. Ensemble methods address this not by fixing the single tree, but by building *many* models and aggregating their predictions. This page covers Random Forest, the ensemble method the course centers on — bagging applied specifically to decision trees, plus one extra trick (random feature subsets) layered on top.

## Core intuition: power in numbers

The lecture motivates ensembling before it motivates Random Forest specifically. Suppose you already have several different classifiers trained on the same problem — say a KNN model, logistic regression, an SVM, and a decision tree ([[course/knn]] and [[course/decision-trees]] both make a cameo here). Combining their predictions by majority vote (**hard voting**) or by averaging their predicted class probabilities (**soft voting**, used when a model can output probabilities) is often more reliable than trusting any single one of them — this works because it turns several "weak learners" (models only slightly better than random guessing) into one stronger predictor.

The slides justify this with a biased-coin analogy: a coin with a 51% chance of heads, flipped once, is barely more informative than a fair coin. Flipped 1,000 times, the *majority* outcome is heads with ~75% probability; flipped 10,000 times, ~97%. Aggregating many independent, weakly-informative signals converges on the right answer far more reliably than any one signal does — but only **if the underlying models' errors are independent of each other**. That independence condition is the whole design problem for the rest of this page: a single dataset and a single greedy algorithm (a decision tree) doesn't naturally produce independent models, so bagging and Random Forest are specifically engineered to manufacture that independence.

## From one tree to many: bagging

**Bagging** (bootstrap **agg**regat**ing**) is the fix for "if we keep building decision trees on the same dataset, we get the same tree every time." Instead of fitting one tree to the full training set, bagging:

1. Draws many **bootstrap samples** — random samples of the same size as the original dataset, drawn *with replacement* (as opposed to "pasting," sampling without replacement), so some rows appear multiple times in a given sample and others don't appear at all.
2. Fits an unconstrained decision tree to each resampled dataset independently.
3. Aggregates the trees' predictions — majority vote for classification, average for regression.

Because each tree sees a slightly different (resampled) dataset, the trees end up different from each other even though they're all the same algorithm on the same underlying data. Averaging across them reduces variance: any one tree's overfit quirks tend to cancel out rather than compound, which is exactly the instability problem called out on [[course/decision-trees]].

## What Random Forest adds on top of bagging

Random Forest is bagging with one more randomization step: at **each split**, instead of considering every available feature, it only considers a random subset of features (`max_features`). The slides frame Random Forest's two randomization sources as:

- **Random samples of rows** (the bagging part) → mostly reduces **variance**.
- **Random subsets of features per split** → the slides call this "kinda boosting" and frame it as reducing **bias**, but the more standard way to think about it is **decorrelation**: if one or two features are strongly predictive, bagging alone still tends to put those same features at the top of most trees, making the trees correlated with each other and limiting how much aggregation can help. Forcing each split to ignore most of the available features breaks that correlation — different trees end up built around different features — so averaging many *decorrelated* trees reduces variance further than bagging alone would.

The three main tuning parameters follow directly from this: the number of trees grown (`n_estimators`), the number of features sampled per split (`max_features` — a common default/rule of thumb is √p for classification, where p is the number of predictors), and the size of each bootstrap sample (`max_samples`). Random Forest works for both classification and regression.

**Aggregating predictions:** for classification, each tree casts a vote and the forest reports either the majority class (hard) or the averaged class probabilities across trees (soft — this is what `oob_decision_function_`/`predict_proba` expose); for regression, predictions are simply averaged across trees.

### Brief contrast: boosting

The slides also introduce **boosting** (e.g. AdaBoost) as a different ensemble strategy worth distinguishing from bagging/Random Forest: boosting fits weak learners (e.g. shallow trees) *sequentially*, with each new tree upweighting the points the previous trees got wrong, so the ensemble progressively corrects its own errors. This reduces **bias**/underfitting rather than variance, and the trees are dependent on each other (fit in sequence) rather than independent (fit in parallel, as in bagging/Random Forest). Boosting isn't this week's focus, but it's the natural next ensemble family once Random Forest is understood.

## Out-of-Bag (OOB) error: a built-in validation set

Because each bootstrap sample leaves out roughly a third of the training rows (sampling with replacement from n rows tends to leave ~1/e ≈ 37% unselected), every row has some trees that never saw it during training. Predicting each row only from the trees that excluded it (its "out-of-bag" trees) gives an error estimate — `oob_score_` in sklearn — without needing a separate validation split or cross-validation fold. `1 - oob_score_` is the OOB error rate.

The lecture notebook's example (`RandomForestExample.ipynb`, a shopper "pregnancy" dataset) sweeps `n_estimators` from 100 to 1000 and shows OOB error isn't monotonically improving — it's lowest around 100–300 trees and creeps back up by 800–1000 on that particular dataset. This is a useful contrast with the single-tree overfitting story on [[course/decision-trees]]: more trees in a forest doesn't overfit the way more depth in a single tree does (each tree is still fit independently on its own resample), but it also doesn't help indefinitely — there's a point of diminishing/slightly negative returns, and OOB error is exactly the tool used to find it.

## Feature importance: two methods

Random Forest carries over the idea of variable importance from single trees, but computes it two ways:

- **Mean Decrease in Impurity (MDI)** — `estimator.feature_importances_`, the same attribute name as a single `DecisionTreeClassifier` (see the R↔Python table on [[course/decision-trees]]), but here it's averaged across every tree in the forest: for each feature, average how much it reduces Gini impurity at every node where it's used, weighted by how many samples reach that node. Computed automatically as part of training, at scale, across all trees. A standard deviation across trees can be computed too (`np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)`) to see how consistently a feature's importance holds across trees.
- **Permutation importance** — shuffle one feature's values (breaking its relationship to the target), re-score the model, and measure how much performance drops; repeat per feature. Unlike MDI, permutation importance isn't biased toward high-cardinality features and can be computed on held-out test data rather than only training data, using whatever metric you care about (the lab example uses precision). The lecture notebook computes both on the same dataset and notes they can disagree on relative ranking even when they agree on which features matter most overall.

## Advantages / limitations (as taught) — the interpretability trade

Advantages emphasized: substantially more stable and accurate than a single tree (bagging reduces variance, feature subsetting further decorrelates the trees), works for both classification and regression, gives a free built-in validation estimate (OOB error) without sacrificing training data to a holdout split, and still produces feature importance the way a single tree does.

The cost is exactly what [[course/decision-trees]] lists as a single tree's biggest advantage: a single tree is a "white box" — a human-readable set of boolean rules. A forest of hundreds of trees is not; you can no longer read off the decision logic directly. Random Forest trades that per-prediction interpretability for accuracy and stability, and the course's tooling reflects the trade: instead of reading rules, you lean on aggregate feature importance (MDI or permutation) to understand the model *in general*, and on tools like LIME (`lime_tabular.LimeTabularExplainer`, used in both notebooks) to explain *individual* predictions after the fact by locally approximating the forest around one data point. There's also more compute cost (training hundreds to thousands of trees instead of one) and more hyperparameters to tune (`n_estimators`, `max_features`, `max_samples`, `min_samples_leaf`, etc.) than a single tree's `max_depth`/pruning knobs. Whether that trade is worth it depends on the use case: pick a single decision tree when a stakeholder needs to see the exact rule that drove a decision; pick Random Forest when predictive accuracy/stability matters more than being able to read the model directly, and you're willing to lean on feature importance and LIME instead.

## R → Python differences

Same conversion-in-progress pattern as [[course/knn]] and [[course/decision-trees]]. The biggest structural difference: R's `randomForest()` retains information across all trees as they're added (so you get an error-rate trajectory for free), while sklearn's `RandomForestClassifier` only gives you the final model's error — getting an error-vs-number-of-trees curve in Python means manually fitting a list of forests at different `n_estimators` values in a loop.

| R | Python | Note |
|---|---|---|
| `randomForest()` | `RandomForestClassifier()` | Python doesn't retain error info across tree counts like R does — R gives an error trajectory as trees are added; Python needs a for-loop over `n_estimators` values to reconstruct that curve. R's output also bundles OOB error rate and a confusion matrix by default; Python requires extracting both manually. |
| `pregnancy_RF$err.rate` | `1 - pregnancy_RF.oob_score_` | Get `oob_score_` from the fitted model, then subtract from 1. Python only gives one error rate per fitted forest (same "no trajectory" limitation as above). |
| `pregnancy_RF$votes` | `pregnancy_RF.oob_decision_function_` | Per-class vote/probability percentages for each data point, extracted directly from the fitted model in both languages. |
| `pregnancy_RF$predicted` | `np.argmax(pregnancy_RF.oob_decision_function_, axis=1)` | No direct sklearn equivalent; reconstructed via `argmax` over the OOB decision function. |
| `pregnancy_RF$importance` | `permutation_importance()` | R's version also reports importance by individual class and mean-decrease-in-Gini; Python needs a separate call (`feature_importances_`) for the Gini/MDI version. |
| (not built into R workflow shown) | `lime_tabular.LimeTabularExplainer` | Pass training data (as an array), feature names, and class names to construct an explainer object. |
| — | `explainer.explain_instance()` | Pass a single row of test data plus the model's `predict_proba` to get a per-row explanation. |
| `treesize(pregnancy_RF, terminal = FALSE)` | `[tree.tree_.node_count for tree in pregnancy_RF_tune.estimators_]` | Python requires pulling `.estimators_` off the fitted forest and looping to get each individual tree's node count — R gives forest-wide tree sizes in one call. |

## Labs

- **Lecture notebook** (`RandomForestExample.ipynb`) — walks a shopper "pregnancy" dataset (2,000 rows, 15 binary purchase-history features, ~28% positive class) through the full workflow: setting `RandomForestClassifier` hyperparameters (`n_estimators`, `max_features`, `bootstrap`, `max_samples`, `min_samples_leaf`, `oob_score`, `warm_start`), fitting, reading OOB error and the confusion matrix, extracting per-point vote probabilities, computing both MDI and permutation feature importance, tuning `n_estimators` (sweeping 100–1000) and `max_features` (using the √p rule of thumb) against OOB error, building an ROC curve and picking an optimal classification threshold, and using LIME to explain one individual prediction.
- **RF_Lab.ipynb** — deliberately open-ended (less prescriptive than earlier labs, "use the skills you've gained over the semester"). Uses the UCI Adult/Census Income dataset (48,000+ individuals, predict income above/below $50K). Students build a Random Forest classifier from scratch: preprocess, fit a baseline model using in-class defaults, report precision/recall/accuracy/OOB error and a confusion matrix, identify the highest-probability positive and negative cases, plot the density of positive-class probabilities, compute and compare MDI vs. permutation feature importance, tune the number of trees and number of features and compare against the baseline, evaluate on a held-out test set with ROC/AUC, and (bonus) use LIME to explain why one specific test row was classified positive.

## Related topics

- [[course/decision-trees]] — Random Forest is bagging + feature-subsetting applied directly to this model; contrast a single greedy, unstable, but fully interpretable tree against a forest that trades that interpretability for variance reduction and stability.
- [[course/knn]] — appears as one of the models in the hard/soft voting ensemble example that opens this week's material; distance-based and requires scaling, unlike the tree-based methods on this page.
- [[course/ml-bias]] — feature importance and evaluation-metric considerations (precision, class imbalance) from earlier weeks carry over directly to how Random Forest models are evaluated here.

## Sources

- `raw/course/DS-3021/ensemble-methods/2020-11-09-ensemble-rf-slides.pptx` — canonical lecture slides (a PDF export of the same 11/9/20 deck also exists in the original course folder but was not carried forward, since it's a redundant format of this same file, not a different version)
- `raw/course/DS-3021/ensemble-methods/2022-09-13-week12-r-to-python-conversion.docx` — R→Python conversion notes
- `raw/course/DS-3021/ensemble-methods/2026-08-31-rf-lecture-notebook.ipynb` — in-class notebook (pregnancy/shopper dataset)
- `raw/course/DS-3021/ensemble-methods/2026-08-31-rf-lab.ipynb` — lab assignment (census income dataset)
