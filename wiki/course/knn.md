# KNN (K-Nearest Neighbors)

Week 6 of DS-3021. A distance-based classifier: to predict a new point's class, look at its `k` nearest neighbors in feature space and take a majority vote. The MLBook overview chapter adds that the same idea also does **regression**: predict the average (or weighted average) of the k neighbors' values instead of a vote.

## How the algorithm works

The lecture slides ("KNN and Probability") present KNN as **memory-based learning**, also called "case-based" or "example-based" learning. The intuition is that similar inputs map to similar outputs, so the work is in defining "similar." Training just indexes the training rows for fast lookup (building a "database"). At prediction time you find the closest stored rows to the new point and copy their label. With k=1 you copy the single nearest neighbor's class. With larger k the neighbors vote.

- **Distance.** The default is Euclidean distance between rows. The slides work one by hand over 8 features: for a new row q = (3,2,2,5,6,2,1,5) and a training row p = (4,3,3,6,4,4,5,6), the squared differences are 1,1,1,1,4,4,16,1, summing to 26, so d = √26 ≈ **5.39**. A second training row z identical to q has distance 0 and would be q's nearest neighbor.
- **Why vote over k > 1.** Polling several neighbors is more reliable than trusting one when there's noise in the features, noise in the labels, or overlapping classes. A graphical red/green example shows the same query point classified **Red with k=3** but **Green with k=5**, which is the slides' setup for "so how do we choose k?"
- **Key characteristics** (slides and MLBook overview agree): **non-parametric** (no assumptions about the data's distribution), **instance-based / lazy** (it stores the whole training set and does no real work until prediction time), and able to learn arbitrarily complicated decision boundaries. The slides' upside of lazy learning is that it adapts quickly to new data (the "Netflix recommendations" example, where new options appear constantly). The downside is very fast training but **slow prediction**, since every prediction is a nearest-neighbor search.

## Why it needs the steps it needs

KNN's behavior follows directly from being distance-based, which is why the lecture spends so much time on preprocessing before ever fitting a model:

- **Feature scaling is mandatory, not optional.** Because classification depends on distance, a variable measured in the thousands (income) will dominate a variable measured in single digits (age) unless everything is scaled to the same range first. The course uses MinMax scaling (`sklearn.preprocessing.MinMaxScaler`) to bring numeric features to [0, 1].
- **Categorical variables must be one-hot encoded** (`pd.get_dummies()`) — distance isn't defined on category labels.
- **Class imbalance matters for k selection.** The example dataset (bank marketing) has only ~11.6% positive cases, so accuracy alone is misleading — see [[course/ml-bias]] for why sensitivity/specificity and confusion matrices matter more than raw accuracy in exactly this kind of imbalanced setting.

The slides motivate scaling with a units example: if one input is weight recorded in **milligrams** rather than pounds, tiny physical differences become huge distances and overwhelm every other feature. The slides' fix is "Standard Scaling" (rescale so standard deviation = 1). The lecture notebook's bank example uses MinMax instead, and the iris example uses `sklearn.preprocessing.scale()` (standardization). The newer data-prep module (below) supports both.

## Advantages and disadvantages (as taught)

From the slides. **Advantages:** simple and fast to deploy, little to no training time, easy to interpret and explain (often used as a **baseline** for other models), handles multiclass data naturally, non-parametric.

**Disadvantages:** the stored "model" is the entire training set, so it takes a lot of disk space. It suffers from the **curse of dimensionality** ("often works best with 25 or fewer dimensions"; in high dimensions there's little difference between the nearest and farthest neighbor). Predictions are computationally expensive (a large search problem), which can make it impractical in industry settings. It needs normalization. It is biased by **skewed class distributions**: a dominant class is more likely to show up among any point's neighbors and so dominates the vote.

The slides also list classifier-selection considerations beyond accuracy: **speed** (an 85%-accurate classifier can beat a 95%-accurate one that is 100× slower at test time), **comprehensibility** to the human operators who apply it, and **training time** in fast-changing environments.

## Choosing k

The lecture builds a `chooseK()` function that tests odd k values (to avoid tie votes) from 1 to 21 and plots accuracy against k — odd values only, and the upper bound has to be extended to 22 in the `range()` call since Python's range is end-exclusive. In the bank-marketing example, k=1 came out best, which the notebook explicitly flags as a teaching moment: k=1 is the most overfit choice possible (every point is its own nearest neighbor), so a follow-up question worth asking is whether best-on-this-sample-accuracy is the same thing as best-for-generalization — see the nested cross-validation section below.

The slides' rules of thumb for k:
- **Odd k** (often 1, 3, or 5) avoids ties in a binary classifier.
- **Large k** is less sensitive to noise (especially class noise), gives better probability estimates, and becomes affordable with larger training sets.
- **Small k** captures fine structure better and may be necessary with small training sets.
- In a three-class (R/G/B) example with heavy overlap, the **k=1** decision boundary looks complex ("Overfitting?"), while **k=15** gives smoother boundaries that "generalize better on unseen data."

## k and the bias-variance tradeoff

The MLBook KNN overview chapter (`knn-overview.ipynb`) uses k as the entry point to the **bias-variance tradeoff**, which it calls "core to machine learning." Its demo fits `KNeighborsClassifier` on a synthetic 2-D dataset (`make_classification(n_samples=200, n_features=2, ...)`, 70/30 split). It starts with k=5, then draws decision boundaries for **k = 1, 7, and 50** and plots training vs. test accuracy for k = 1–50, marking the best test k.

- **Small k (e.g. k=1):** highly flexible, with jagged boundaries that fit noise. The result is **overfitting**: high training accuracy, poor generalization. This is **low bias, high variance**: predictions change a lot with small changes in the training data.
- **Large k (e.g. k=n):** rigid, with overly smooth boundaries that respond only to global trends. The result is **underfitting**: training and test accuracy both low. This is **high bias, low variance**.
- **Sweet spot:** an intermediate k gives the best generalization.

The chapter defines **bias** as error from approximating a complex problem with a too-simple model, and **variance** as how much predictions would change if the model were trained on a different dataset. It gives the decomposition **Total Error = Bias² + Variance + Irreducible Noise**. Its bias²/variance/total-error plot is explicitly a *simulated, conceptual* curve, not computed from data.

The chapter then generalizes beyond KNN with a table of modeling decisions as "bias-variance decisions in disguise": deeper decision trees mean lower bias and higher variance (see [[course/decision-trees]]); L1/L2 regularization raises bias to cut variance; adding features may cut bias but raise variance; bagging reduces variance and boosting reduces bias (see [[course/ensemble-methods]]); more training data reduces variance without raising bias. It frames cross-validation, regularization, train/validation/test splits, grid/random search, and learning/validation curves as tools for finding the sweet spot. "Perfect training performance is not the goal." The chapter ends with practice questions, for example what to do with k when training accuracy is 100% but test accuracy is 60%, or how to respond to a colleague who always uses k=1 "because it gives the best training accuracy."

## Lecture notebook walkthrough (bank marketing)

The in-class notebook predicts `signed up` on a bank-marketing dataset (`bank.csv` from the UVADS DS-3001 repo):

1. **Inspect categorical levels** with `value_counts()`. `job` has 12 levels ("What should we do?"), so it is collapsed into `Employed` (admin, blue-collar, entrepreneur, housemaid, management, self-employed, services, technician) vs. `Unemployed` with an `apply(lambda ...)`. Then convert the categorical columns to `category` dtype.
2. Check missing data with the seaborn plot ("No missing data!!"), MinMax-scale the `int64` columns, and one-hot encode with `pd.get_dummies()`, which makes the target `signed up_1`.
3. **Prevalence**: 11.6% signed up. "At random, we have an 11.6% chance of correctly picking a subscribed individual."
4. Split 60/20/20 train/test/validation with two stratified `train_test_split` calls (`test_size=0.4`, then `0.5`). Fit `KNeighborsClassifier(n_neighbors=9)`.
5. **Evaluate**: about **99.0% accuracy**, which the notebook immediately tempers against the roughly 89/11 base rate. The model's value is in identifying the sign-ups. It shows a confusion matrix and `classification_report`, then computes sensitivity and specificity by hand.
6. `chooseK()` sweep (see above), then **threshold adjustment**. Build a table of `predict_proba` outputs and the positive-class probability, plot its density (`sns.displot(kind="kde")`), and use an `adjust_thres()` helper to print confusion matrices at a raised (0.90) and lowered (0.3) threshold. "Is a 60% certainty positive? How about 40%?"
7. A "more for next week" preview of ROC/AUC (`metrics.roc_curve`, `roc_auc_score`, and the `plot_metric` package), F1, and log loss. These are covered on [[course/ml-eval-metrics]].

A second, shorter **iris** example standardizes with `scale()`, fits k=3, and introduces **repeated k-fold CV**: `RepeatedKFold(n_splits=10, n_repeats=3)` plus `cross_val_score(..., scoring='accuracy')`, and a training-vs-testing accuracy plot over k = 1–8. It then builds **permutation variable importance** by hand: a `featureImportance()` function that shuffles one column at a time (`np.random.permutation`) and records baseline accuracy minus permuted accuracy.

## Reusable data-preparation module

`2026-03-10-knn-data-preparation-script.py` is a PEP 8–styled, heavily docstringed module that packages the lecture notebook's bank-data cleaning into reusable functions. It matches the `data_preparation.py` described in the folder's `README.md`.
- `collapse_job_categories()`: the same Employed/Unemployed collapse as the notebook
- `convert_to_categorical()`: `.astype('category')` on a list of columns
- `normalize_numeric_columns(df, method='minmax')`: scales `int64`/`float64` columns with `MinMaxScaler` (default) or `StandardScaler` (`method='standard'`), and returns the fitted scaler too, "for potential inverse transformation" or use on new data. The README only mentions MinMax/[0, 1].
- `one_hot_encode_categorical()`: `pd.get_dummies()` on auto-detected `category` columns
- `prepare_bank_data(df, target_column='signed up')`: the full pipeline **in order**: collapse jobs, convert to categorical, normalize, one-hot encode
- `split_train_val_test(df, target_column, test_size=0.4, val_size=0.5, random_state=1984)`: two stratified splits giving 60/20/20. Note it returns `(train, test, val)` in that order, and the target is the post-encoding name (`'signed up_1'`).
- `prepare_features_and_target()`: splits the frame into X and y for scikit-learn

## Nested cross-validation (the current version of this material)

The `README.md` in this folder documents a more recent rewrite of the KNN code (`data_preparation.py` + `knn.py`) that replaces the single train/test split with **nested cross-validation**:
- **Outer loop** (5-fold): estimates overall model performance on data the hyperparameter search never touched.
- **Inner loop** (3-fold): tunes k for each outer fold.

Per the README, `knn.py` tests k = 1, 3, 5, …, 21, reports the best k for each outer fold, then trains a final model and evaluates it on validation and test sets. Reporting covers a confusion matrix, classification report, ROC/AUC, and sensitivity/specificity. Fold counts and the k range are parameters of `nested_cross_validation()` (e.g. `k_range=range(1, 31, 2)`, `outer_cv_folds=10`, `inner_cv_folds=5`). The README also explains why train and validation are combined for nested CV (to maximize tuning data while keeping test fully independent), and says the code shows both nested CV and a traditional train/val/test split "for educational purposes."

This matters pedagogically because it's the fix for the exact overfitting risk the k=1 result above hints at — tuning k on the same data you evaluate on overstates performance. This is also a good candidate for a general "cross-validation" concept page once more topics are ingested, since nested CV comes up again in decision tree hyperparameter search (see [[course/decision-trees]]).

## R → Python differences

A recurring pattern across this course: labs were originally written in R and are being converted to Python. For KNN specifically:

| R | Python | Note |
|---|---|---|
| `md.pattern()` (mice) | `seaborn` displot on `df.isna().melt()` | Python version has no built-in value counts |
| built-in normalize | `sklearn.preprocessing.MinMaxScaler` | |
| `one_hot()` | `pd.get_dummies()` | |
| `createDataPartition()` | `train_test_split()` | |
| `knn()` | `KNeighborsClassifier()` | |
| model print-out | `plot_confusion_matrix()` + `classification_report()` | Python needs two calls where R gives one; sensitivity/specificity aren't built in and must be computed manually as TP/(TP+FN) and TN/(TN+FP) |
| variable importance | not built into sklearn for KNN | Computed manually via permutation: shuffle one variable, measure accuracy drop, repeat |

Indexing is the other trap called out repeatedly: R is 1-indexed, Python is 0-indexed.

## Labs

- **Graduation Lab** (`grad_lab.ipynb`) — reuses the college-completion dataset from the prior week, has students pick their own target variable and build/tune a KNN model against it, including experimenting with the classification threshold.
  - The data (from data.world, read with `encoding="cp1252"`) is described as messy with "a degrees of freedom problem, as in, we have too many features." The instructor's example cleaning drops many columns by index, replaces `'NULL'` strings with `NaN`, recodes `hbcu` ('X' → 1), converts `level`/`control` to categories, drops `med_sat_value`, then drops remaining NA rows.
  - Tasks: (1) use the target/question from last week, (2) tune k on the **tune set**, (3) experiment with higher/lower thresholds and document what happens, (4) evaluate with a confusion matrix at the default threshold and discuss concerns. **Bonus:** change the k-selection function to optimize **specificity** instead of accuracy.

## Probability primer (second half of the slides)

The "KNN and Probability" deck pivots after the KNN exercise to the probability background needed for probability-based classifiers. It covers random variables (discrete vs. continuous), Kolmogorov's three probability axioms, discrete distributions (a three-coin-toss example: P(X = 0, 1, 2, 3 heads) = 1/8, 3/8, 3/8, 1/8), continuous distributions and **probability density** (why a single exact height has probability 0, and why density can exceed 1), joint/marginal/conditional probability, and statistical independence. Classification is introduced at the start of the deck as one of several techniques (k-NN, Naïve Bayes, logistic regression, tree-based methods). The deck also distinguishes classification's unsupervised sense (discovering clusters) from its supervised sense; KNN is supervised.

## Related topics

- [[course/decision-trees]] — a non-distance-based alternative; contrasts usefully with KNN (no scaling required, handles mixed data types natively)
- [[course/ml-bias]] — confusion matrix / sensitivity / specificity concepts introduced here are foundational to the fairness metrics covered there
- [[course/ml-eval-metrics]] — ROC/AUC, F1, and log loss, previewed at the end of the KNN lecture notebook
- [[course/ensemble-methods]] — the bias-variance table in the MLBook overview frames bagging as variance reduction and boosting as bias reduction
- [[course/kmeans-clustering]] — also distance-based and scale-sensitive, but its `k` is the number of clusters, not neighbors
- [[course/pca]] — dimensionality reduction, a response to the curse of dimensionality that the KNN slides flag

## Sources

- `raw/course/DS-3021/knn/README.md` — nested CV rewrite documentation
- `raw/course/DS-3021/knn/2024-10-29-knn-r-to-python-changes.docx` — R→Python conversion notes
- `raw/course/DS-3021/knn/2024-10-29-knn-lecture-notebook.ipynb` — in-class lecture notebook
- `raw/course/DS-3021/knn/2024-10-29-knn-graduation-lab.ipynb` — lab assignment
- `raw/course/DS-3021/knn/2024-10-29-knn-and-probability-slides.pdf` — lecture slides
- `raw/course/DS-3021/knn/2024-10-29-machine-learning-III-reading.pdf` — supplementary reading
- `raw/course/DS-3021/knn/2026-03-10-knn-data-preparation-script.py` — reusable data-preparation module (the `data_preparation.py` described in the README)
- `raw/course/MLBook/knn/2026-05-21-knn-overview.ipynb` — MLBook KNN overview chapter (k vs. overfitting/underfitting, bias-variance tradeoff)
