# Decision Tree Regression

Week 9 of DS-3021 — the direct extension of [[course/decision-trees]] (week 8's CART classification trees) to a continuous target. Same algorithm skeleton: hierarchical, greedy recursive binary splitting, root/internal/leaf structure, same overfitting risk and pruning hyperparameters. What changes is the arithmetic inside each split and what a leaf hands back as a prediction. This page assumes the CART fundamentals from [[course/decision-trees]] and only covers what's different here — read that page first if the vocabulary (root/internal/leaf nodes, recursive binary splitting, greedy vs. globally optimal) isn't already familiar.

⚠️ **Possible conflict — see below.** This page originally described `2025-09-25-dt-regression-lab.ipynb` as the *current* lab (per the instructor's public reference repo) and treated the version with the normality/transform and "what-if" steps as an older, locally stored copy. The newly ingested `2026-03-10-dt-regression-lab-v2.ipynb` is that longer version. It carries the later date and a "v2" label, which suggests it may now be the current one. The sources don't say which version is taught. Both are described in the [Lab](#lab) section, so check which one a given semester actually used.

**Note on sources:** the lecture slides for this week (`Decision_Trees_3.24.22.pptx`) are literally the same deck used for week 8 — it's one continuous lecture that covers classification first (slides 1–31, entropy/Gini/information-gain worked example, CART/ID3/C4.5/C5.0 lineage) and then regression trees second (slides 32–44). This page draws only from the regression portion; the classification portion is already synthesized on [[course/decision-trees]] and isn't repeated here.

## What carries over unchanged

- Recursive binary splitting, greedy/locally-optimal (not globally optimal) split selection
- Root / internal / leaf node structure and terminology
- No scaling required, handles numeric and categorical features natively
- Overfitting risk and instability under small data changes (leaf count still roughly doubles per added depth level)
- The same pruning/stopping hyperparameters: `max_depth`, minimum samples per leaf, `min_impurity_decrease`, cost-complexity pruning (`ccp_alpha`)

That last point is worth calling out explicitly: scikit-learn's `DecisionTreeRegressor` exposes the *same* hyperparameter surface as `DecisionTreeClassifier` — including `min_impurity_decrease` and `feature_importances_` — because "impurity" is still the internal bookkeeping concept the tree minimizes at every split. What changed is only how impurity is computed (below), not the API built on top of it.

## What changes: the split criterion is variance reduction (MSE), not Gini/entropy

Gini index and entropy both require a class *probability* at a node, which doesn't exist for a continuous target. Regression trees instead search for the split that minimizes **sum of squared error (SSE)** — equivalently, mean squared error (MSE) once normalized by the count of points in a group. The slides work a small numeric example: for one side of a candidate split with values 75, 85, and 80, the group's average is 80; squared errors are (75−80)² + (85−80)² + (80−80)² = 25 + 25 + 0 = 50, giving MSE = 50/3 ≈ 16.67. Do that for every candidate split point on every feature and pick whichever split gives the lowest (weighted) MSE across the two resulting groups — the direct regression analog of picking the split with the highest information gain in a classification tree.

In scikit-learn this is the `criterion` parameter: the lecture notebook leaves it at its default, while the companion `.py` script sets it explicitly — `DecisionTreeRegressor(random_state=30, criterion='squared_error')`. Both are grid-searching *some* hyperparameter to control tree growth, but they don't search the same one (see the worked example below).

## What changes: a leaf predicts a mean, not a vote

A classification leaf outputs a class (by majority vote among training points that landed there, or a class probability). A regression leaf outputs the **arithmetic mean** of the training target values that landed there. This isn't an arbitrary convention — it falls directly out of the split criterion above: the slides note explicitly that "the average of each of the groups is the term that minimizes the mean squared error," so the mean is the leaf's prediction precisely because it's the value that minimizes squared error against everything in that leaf. Predicting on new data just means walking the tree to a leaf and reporting that leaf's stored mean.

## What changes: evaluation metrics

Classification trees are evaluated with a confusion matrix, accuracy, sensitivity/specificity, ROC. None of those are defined for a continuous prediction. The course instead scores regression trees with three metrics in parallel: **R²**, **RMSE**, and **MAE**. The lecture notebook passes all three to `GridSearchCV` at once (`scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']`), which surfaces two Python-specific gotchas not present in the classification workflow:

- **Sign flip.** scikit-learn's cross-validation scorers are written so that "higher is always better" (for consistency with things like accuracy), so MSE and MAE are stored as *negative* values internally — `neg_mean_squared_error`. Getting an actual RMSE back requires `np.sqrt(abs(...))`; MAE just needs `abs(...)`.
- **`refit` becomes required.** When you pass a *list* of scoring metrics instead of one, `GridSearchCV` no longer knows which metric should determine the "best" estimator — you have to tell it explicitly (`refit='r2'` in this course's example).

The R→Python conversion doc for this week frames the same contrast from the R side: R's `caret::postResample()` bundles RMSE, R², and MAE into one call given predictions and actuals; Python needs three separate `sklearn.metrics` calls (`r2_score()`, `mean_squared_error()`, `mean_absolute_error()`) to get the same three numbers — one R function maps to three Python ones. (The conversion doc for this week is intentionally thin beyond this table — it explicitly tells students to "reference week 08 conversion doc as well, since many functions are similar and not repeated on here," so most of the general R→Python patterns already logged on [[course/decision-trees]] carry over unchanged.)

| R | Python | Note |
|---|---|---|
| `postResample()` | `r2_score()` | predictions must be generated from the fitted model first, in both languages |
| `postResample()` | `mean_squared_error()` | Python requires `np.sqrt()` on top to convert MSE to RMSE; R's version returns RMSE directly |
| `postResample()` | `mean_absolute_error()` | same "get predictions first" requirement |
| (not used) | `Pipeline()` | bundles preprocessing steps and a final estimator into one object; each step must be explicitly named |
| (not used) | `ColumnTransformer()` | applies different preprocessing to numeric vs. categorical columns within one step |
| (not used) | `make_pipeline()` | shorthand for `Pipeline()` that auto-names steps instead of requiring names |

## Target variable considerations (no classification analog)

Because the target is now continuous rather than a fixed set of classes, the lab explicitly adds a step with no equivalent in the classification lab: check the target variable's range and variance before modeling. The locally-stored version of the lab goes further, asking whether the target is roughly normally distributed and, if not, whether it's worth transforming — "why might this be important?" The version of this lab currently published in the instructor's own reference repo simplifies this to just "determine the range and variance of the target variable," dropping the normality/transform framing, but the underlying point is the same either way: a skewed continuous target can bias splits in ways that a balanced/imbalanced *class* discussion (relevant on the classification side, see [[course/ml-bias]]) doesn't capture.

## Worked example: the lecture notebook

`DT_Reg_Example.ipynb` opens by noting it reuses "the same wine dataset from last week" (`winequality-red-ddl.csv`, the same file used in the week 8 lecture notebook) — but now predicting a continuous target (**alcohol** content) instead of a class label. Walkthrough:

1. Drop NA rows, then ordinal-encode the one text column (`text_rank`, a categorical wine-quality label) to numeric — same idea as the categorical encoding used on the classification side, just necessary here because *every* feature has to be numeric for `DecisionTreeRegressor`.
2. Split 70/15/15 (train / tune / test) via two chained `train_test_split()` calls.
3. Cross-validate with `RepeatedKFold(n_splits=10, n_repeats=5)` — 10-fold CV repeated 5 times and averaged, rather than the single k-fold pass used in the classification lab.
4. Grid-search `max_depth` (the notebook) or `min_impurity_decrease` (the `.py` script — a different hyperparameter, deliberately, so students see both levers) via `GridSearchCV`, scoring on all three regression metrics, `refit='r2'`.
5. Best model: `DecisionTreeRegressor(max_depth=7, random_state=30)`. Variable importance is dominated by `density` (importance ≈ 0.53 — "density had a huge impact!" per the notebook's own comment), a long way ahead of the next-highest features (`quality` ≈ 0.12, `fixed acidity` ≈ 0.10).
6. Test-set scores: R² ≈ 0.58, RMSE ≈ 0.67, MAE ≈ 0.49.
7. A second pass rebuilds the whole thing as a `Pipeline` + `ColumnTransformer` that **imputes** median values for missing data (via `SimpleImputer`) instead of dropping NA rows — so imputation reruns fresh on every CV fold rather than leaking information by imputing once globally before splitting. This pipeline version lands on a different best depth (6 instead of 7) and does *worse* on the held-out test set (R² ≈ 0.45) while doing *better* on tune (R² ≈ 0.52) — the notebook flags this explicitly as a discussion point about the trade-off between simplicity and a more realistic (fold-safe) preprocessing setup, not a bug.

The `.py` script version of this same example (`DT_Reg_Example.py`) is a rougher, less-edited companion to the notebook and contains two source-level quirks worth knowing about if a student opens it directly rather than the notebook: an early cell computes a Big-O complexity estimate (`O(n · m · log m)`, n = features, m = samples) using a variable (`X`) that isn't actually defined until much later in the script; and a later cell (`print(sorted(SCORERS.keys()))`) is explicitly flagged in its own comment with "why doesn't this work?" — it fails because `SCORERS` was never imported directly (only `from sklearn import metrics` was), where the notebook's equivalent line correctly calls `metrics.SCORERS.keys()`. Both look like intentional in-class debugging exercises rather than accidental errors, consistent with this course's general pattern of having students spot and fix small Python-specific mistakes.

## Lab

`DT_Reg_Lab.ipynb` continues the exact same fictional scenario as the week 8 classification lab, one rung up the org chart: "Congrats....again! Based off the amazing work you did on the classification model, you've been promoted to chief of all data scientists in Hollywood... your boss... has now asked you to build a model to predict gross revenue." Same `movie_metadata.csv` dataset, same "you don't need to scale the data, because DTs don't require this... they make local greedy decisions" callback as the classification lab — but the task swaps from predicting IMDb-score class to predicting continuous gross revenue, and the step list swaps out confusion-matrix/evaluation questions for the target-range/variance check and RMSE/R²/MAE evaluation questions described above. (This page uses the lab's current wording from the instructor's public reference repo, which tightens several of the step prompts and trims a couple of "what-if experimentation" steps present in the locally-stored copy — the underlying assignment is the same.)

### Lab v2 (`2026-03-10-dt-regression-lab-v2.ipynb`)

Same scenario, intro text, dataset, and "don't scale" step 4. v2 has 22 steps instead of 19 and is more open-ended, with a fuller train → tune → test workflow. Differences from the 2025-09-25 version:

- **Step 2** drops "including the target variable" from the variable-typing instruction.
- **Step 5** asks for the target's **variance and whether it is normally distributed**, and if not, whether to **transform** it ("Why might this be important?"). The 2025 version asks for "range and variance." This is the normality/transform framing discussed under *Target variable considerations* above.
- **Step 8** lets students "choose a hyperparameter to optimize" instead of prescribing `max_depth`. The separate "grid search for max depth" and "what is the best depth?" steps collapse into one: "Print the output of the grid search. What is the best parameter and score?"
- **Steps 14–16 evaluate on the tune set** (predict, compare tune vs. training, and list the five movies with the lowest predicted gross revenue from the **tune** set). The 2025 version does all of this on the test set.
- **New "WHAT IF?!" steps (17–18):** brainstorm ways to improve the model, try two, and report whether it improved and by how much. If it didn't improve, report what you learned. Code for every experiment isn't required.
- **New final-evaluation steps (19–21):** predict on the test set with the best model, compare test performance against training *and* tune ("Why or why not?"), and do **error analysis**: "Where are the errors occurring in your predictions? Are they systematic or random?" plus the implications.
- Step 22 keeps the closing "summarize… recommendations… careful not to over promise."

## Related topics

- [[course/decision-trees]] — shared CART fundamentals: node vocabulary, greedy recursive splitting, overfitting/pruning hyperparameters, "white box" interpretability
- [[course/ensemble-methods]] — Random Forest / Ensembles (week 12): `DecisionTreeRegressor` plugs into `RandomForestRegressor` the same way `DecisionTreeClassifier` extends to `RandomForestClassifier`; the instability argument for ensembling made on the classification page applies identically here

## Sources

- `raw/course/DS-3021/dt-regression/2022-03-24-decision-trees-regression-slides.pptx` — full lecture deck; slides 1–31 duplicate the classification-tree material already covered on [[course/decision-trees]], slides 32–44 (MSE/SSE splitting, hyperparameter pruning options, dataset-imbalance discussion) are the regression-specific portion this page draws from
- `raw/course/DS-3021/dt-regression/2022-08-25-week09-r-to-python-conversion.docx` — R→Python conversion notes (evaluation-metric table); explicitly points back to the week 8 conversion doc for everything not repeated here
- `raw/course/DS-3021/dt-regression/2024-10-29-dt-regression-lecture-notebook.ipynb` — in-class notebook, wine-quality dataset, predicting alcohol content
- `raw/course/DS-3021/dt-regression/2026-03-10-dt-regression-lecture-script.py` — companion `.py` version of the same example, grid-searches a different hyperparameter (`min_impurity_decrease` vs. the notebook's `max_depth`) and contains two source-level debugging quirks noted above
- `raw/course/DS-3021/dt-regression/2025-09-25-dt-regression-lab.ipynb` — lab assignment (movie gross-revenue prediction), current version per the instructor's public reference repo
- `raw/course/DS-3021/dt-regression/2026-03-10-dt-regression-lab-v2.ipynb` — lab v2: 22 steps, open hyperparameter choice, normality/transform check, tune-set evaluation, "what-if" experiments, error analysis (see the ⚠️ note at top on which version is current)
