# Decision Trees (CART)

Week 8 of DS-3021 (classification trees; week 9 covers the regression-tree variant — see [[course/dt-regression]]). A hierarchical model: ask a sequence of yes/no questions about the data, each one chosen to maximize information gain, until a stopping criterion is met.

⚠️ **Version note:** this folder had three different years' slide decks for the same lecture (2021, 2022, 2024-modified). This page is built from the most recently taught version (`2022-10-20-decision-trees-slides-current.pptx`, last edited 2026). The older decks weren't ingested — if something here looks inconsistent with what was actually taught in a given semester, check which deck was used that year.

## Core intuition

The slides build the idea through a dating-app analogy before the math: which question gives you the most information (e.g. "are you married?") gets asked first; each subsequent question is chosen *conditional on* the answer to the previous one; you stop asking questions once an answer no longer provides additional information (no information gain). That four-step process — pick the most informative question, branch, recurse, stop — is the algorithm.

Key structural vocabulary: **root node** (no incoming edges), **internal nodes** (test conditions), **leaf nodes** (final predictions), **edges** (the branches connecting them).

## Why "greedy" and what it costs

Decision trees use **recursive binary splitting**: considering every possible partition of the data is computationally infeasible, so at each step the tree picks the locally best split without looking ahead to whether a different split now would lead to a better tree later. This greediness is the source of two of the model's main weaknesses (both noted explicitly in the slides):

- **Instability** — small changes in the training data can produce a completely different tree. This is the direct motivation for ensembles like Random Forest, which train many trees on resampled data and average them out — see [[course/ensemble-methods]].
- **No global optimality guarantee** — same fix (ensembling).

The other major failure mode is **overfitting**: an unconstrained tree keeps splitting until nodes contain very few points (leaf nodes roughly double with each added depth level — 6 levels ≈ 64 terminal nodes), which is controlled with `max_depth`, minimum leaf size, or pruning.

## Splitting criteria: entropy and information gain

For classification, splits are chosen by **Gini index** (most common in practice) or **entropy**, both of which measure impurity at a node; **information gain** is the reduction in (weighted average) entropy from parent to children. The slides work a full numeric example (a 4-row "should you play outside?" dataset) showing the calculation: entropy of the parent node, entropy of each candidate child split weighted by how many points fall into it, and the resulting information gain used to pick which attribute to split on first.

Historical lineage worth knowing: CART (Breiman et al., 1984, binary trees) → ID3 (information gain) → C4.5 (gain ratio, added pruning, handles continuous + discrete) → C5.0 (added boosting, more efficient) — these names show up in the broader ML literature and it's useful for students to recognize sklearn's `DecisionTreeClassifier` as a CART implementation specifically.

## Advantages / limitations (as taught)

Advantages emphasized: minimal data prep required (no scaling needed, unlike [[course/knn]]), handles numeric and categorical data natively, and is a "white box" model — the prediction logic is human-readable boolean rules, unlike black-box approaches.

Limitations emphasized: overfitting risk, instability under small data perturbations, only a locally (not globally) optimal solution, and bias toward dominant classes in imbalanced data (recommend balancing the dataset before fitting) — worth cross-referencing with [[course/ml-bias]] on why class imbalance also matters for fairness evaluation, not just accuracy.

## R → Python differences

Same conversion-in-progress pattern as [[course/knn]]:

| R | Python | Note |
|---|---|---|
| `mice::md.pattern(df)` | `df.isna().sum()` | R gives a visual; Python gives counts only |
| `fct_collapse()` | `.replace()` | |
| (not needed in R) | `OrdinalEncoder()` | only for naturally-ordered categorical data |
| `train(method="rpart2", trControl=..., tuneGrid=...)` | `DecisionTreeClassifier()` + `GridSearchCV()` + `.fit()` | R's `train()` bundles hyperparameter search and fitting; Python needs the pieces assembled explicitly |
| `model$results` | `model.cv_results_` | dictionary access in Python vs. indexing in R |
| `rpart.plot()` | `export_graphviz()` | Python requires passing feature names/labels explicitly |
| `varImp(model)` | `estimator.feature_importances_` | Python needs the *best* estimator object specifically |
| `confusionMatrix()` | `ConfusionMatrixDisplay.from_estimator()` | R gives more stats by default |
| `rpart(control = rpart.control(cp=.001))` | `cost_complexity_pruning_path` | Python returns raw alphas/impurities to build pruned models manually, rather than one call |

## Lab

`DT_Class_Lab.ipynb` — a movie-studio scenario: predict whether a movie will get a high IMDb score. Walks through the full pipeline (type-checking variables, missing data function, 80/10/10 train/tune/test split, k-fold grid search over `max_depth`, variable importance, confusion matrix on tune data, then a second hyperparameter pass to compare). Notably, step 4 explicitly tells students *not* to scale the data ("DTs don't require this... they make local greedy decisions") — a direct, gradable contrast with the KNN lab's insistence on scaling.

### Newer (MLBook) version of the lab

The MLBook copy (`2026-05-21-dt-class-lab.ipynb`) has the same movie-studio framing, the same `movie_metadata.csv` load, and the same steps 1–15 (including the "don't scale" step 4, the base-rate step 5, the 80/10/10 split, the k-fold object, and grid search over `max_depth`). It is a **shorter, test-set-focused** version. What changed from the 2026-03-10 DS-3021 version:

- **Step 3** no longer asks students to wrap their missing-data cleaning in a reusable function (it's just "check for missing variables and correct as needed").
- **Step 8** drops the explicit "(grid search)" label, and **step 14** asks students to comment on performance "using several evaluation metrics" rather than "the metrics you selected."
- **Steps 16–19 move from the tune set to the test set**: predict on test data, judge test performance, print the test-set confusion matrix, and find the top 3 movies in the test set.
- **Steps 20–23 are gone**: the second hyperparameter pass, the "did the model improve?" comparison, predicting test data with the better model, and the closing summary/recommendations to the boss ("being careful not to over promise"). The lab now ends at step 19.

One loose end is that step 6 still asks for a train/tune/test (80/10/10) split, but no later step uses the tune set.

## Related topics

- [[course/knn]] — contrasting non-distance-based vs. distance-based approach to classification; scaling required there, not here
- [[course/ml-bias]] — class imbalance and evaluation metrics apply to trees the same way they do to any classifier
- [[course/ml-eval-metrics]] — the metrics used to evaluate a tree's predictions (confusion matrix, precision/recall/F1, ROC/AUC)
- [[course/dt-regression]] — the same CART algorithm applied to a continuous target (week 9); covers only what's different from this page
- [[course/ensemble-methods]] — Random Forest, the direct fix for the instability/no-global-optimality weaknesses noted above (week 12)

## Sources

- `raw/course/DS-3021/decision-trees/2022-10-20-decision-trees-slides-current.pptx` — canonical lecture slides (most recently taught version)
- `raw/course/DS-3021/decision-trees/2024-10-29-week08-r-to-python-conversion.docx` — R→Python conversion notes
- `raw/course/DS-3021/decision-trees/2025-02-13-decision-trees-lecture-notebook.ipynb` — in-class notebook
- `raw/course/DS-3021/decision-trees/2026-03-10-dt-class-lab.ipynb` — lab assignment
- `raw/course/MLBook/decision-trees/2026-05-21-dt-class-lab.ipynb` — newer MLBook version of the same lab (shorter; evaluates on the test set, drops the second hyperparameter pass)
- `raw/course/DS-3021/decision-trees/2024-10-29-cart-loh-academic-reference.pdf` — Loh, "Classification and Regression Trees" (academic reference)
