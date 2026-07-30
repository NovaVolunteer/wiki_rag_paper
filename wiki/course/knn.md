# KNN (K-Nearest Neighbors)

Week 6 of DS-3021. A distance-based classifier: to predict a new point's class, look at its `k` nearest neighbors in feature space and take a majority vote.

## Why it needs the steps it needs

KNN's behavior follows directly from being distance-based, which is why the lecture spends so much time on preprocessing before ever fitting a model:

- **Feature scaling is mandatory, not optional.** Because classification depends on distance, a variable measured in the thousands (income) will dominate a variable measured in single digits (age) unless everything is scaled to the same range first. The course uses MinMax scaling (`sklearn.preprocessing.MinMaxScaler`) to bring numeric features to [0, 1].
- **Categorical variables must be one-hot encoded** (`pd.get_dummies()`) — distance isn't defined on category labels.
- **Class imbalance matters for k selection.** The example dataset (bank marketing) has only ~11.6% positive cases, so accuracy alone is misleading — see [[course/ml-bias]] for why sensitivity/specificity and confusion matrices matter more than raw accuracy in exactly this kind of imbalanced setting.

## Choosing k

The lecture builds a `chooseK()` function that tests odd k values (to avoid tie votes) from 1 to 21 and plots accuracy against k — odd values only, and the upper bound has to be extended to 22 in the `range()` call since Python's range is end-exclusive. In the bank-marketing example, k=1 came out best, which the notebook explicitly flags as a teaching moment: k=1 is the most overfit choice possible (every point is its own nearest neighbor), so a follow-up question worth asking is whether best-on-this-sample-accuracy is the same thing as best-for-generalization — see the nested cross-validation section below.

## Nested cross-validation (the current version of this material)

The `README.md` in this folder documents a more recent rewrite of the KNN code (`data_preparation.py` + `knn.py`) that replaces the single train/test split with **nested cross-validation**:
- **Outer loop** (5-fold): estimates overall model performance on data the hyperparameter search never touched.
- **Inner loop** (3-fold): tunes k for each outer fold.

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

## Related topics

- [[course/decision-trees]] — a non-distance-based alternative; contrasts usefully with KNN (no scaling required, handles mixed data types natively)
- [[course/ml-bias]] — confusion matrix / sensitivity / specificity concepts introduced here are foundational to the fairness metrics covered there

## Sources

- `raw/course/DS-3021/knn/README.md` — nested CV rewrite documentation
- `raw/course/DS-3021/knn/2024-10-29-knn-r-to-python-changes.docx` — R→Python conversion notes
- `raw/course/DS-3021/knn/2024-10-29-knn-lecture-notebook.ipynb` — in-class lecture notebook
- `raw/course/DS-3021/knn/2024-10-29-knn-graduation-lab.ipynb` — lab assignment
- `raw/course/DS-3021/knn/2024-10-29-knn-and-probability-slides.pdf` — lecture slides
- `raw/course/DS-3021/knn/2024-10-29-machine-learning-III-reading.pdf` — supplementary reading
