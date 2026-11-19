# Data Preparation & the ML Pipeline ("ML Bootcamp")

Week 5 of DS-3021 (the R→Python conversion doc is titled "Week 05: R to Python", and the bootcamp notebook's Colab link points at the course folder `05_ML_Concepts_II_Data_Prep`). This is the "ML Bootcamp" week: before any specific algorithm is taught, the course lays out Brian Wright's version of the data science lifecycle and the vocabulary of machine learning, then walks a full data-prep pipeline end to end on the **cereal** dataset: fix variable types, collapse factor levels, scale/normalize, one-hot encode, compute prevalence, drop unneeded columns, and partition into Train/Tune/Test. Every later model page ([[course/knn]], [[course/decision-trees]], [[course/ensemble-methods]]) assumes this pipeline has already been run.

> ⚠️ Possible conflict — see "Missing data" and "Partitioning" below. The Python slide deck and the older R-based "ML Bootcamp II" reading give different rules of thumb for missing data (5–10% vs. 5%) and name different partitioning functions (`train_test_split` vs. caret's `createDataPartition`).

## What machine learning is (as framed in the lecture)

The deck ("Machine Learning Overview", Brian Wright, PhD) opens with three definitions: Arthur Samuel (coined the term in 1959 at IBM: "the ability to learn without being explicitly programmed"), Goodfellow et al.'s *Deep Learning* ("acquire their own knowledge, by extracting patterns from raw data"), and Tom Mitchell's experience E / tasks T / performance measure P definition. A machine-vs-human table assigns understanding context, thinking through the problem, asking the right questions, selecting tools, and interpreting results to **humans**, and fast calculation, repetitive tasks, and following pre-defined rules to **machines**.

- **Supervised** learning: inputs (x) and outputs (y) are known (example: voter → political affiliation). Classification and regression are both supervised.
- **Unsupervised** learning: no target y; the goal is to impose structure on x to say something meaningful (the speaker notes use a wine list clustered by region, year, price, type). See [[course/kmeans-clustering]].
- ML is described as a **general-purpose technology (GPT)**, like electricity or the internet. Examples given: ImageNet error falling from 30% in 2010 to under 3% ("below 5% is important why?"), and Deep Blue searching ~200 million chess positions per second versus Kasparov's 5–10.
- **Suitable for Machine Learning (SML)** tasks: learning a function from well-defined inputs to well-defined outputs (prediction is not causal effect); large data with input-output pairs; clear feedback with definable goals and metrics; no need for reasoning or diverse background knowledge; no need to explain *why* a decision was made; tolerance for error (outputs are probabilistic); and a function that doesn't change rapidly over time (otherwise systems are needed to refresh algorithms).
- "How do machines learn?": **Data input → Abstraction → Generalization** (citing Chandramouli, Dutt & Das, *Introduction to ML*).

## Brian's Data Science Lifecycle (four phases, three gates)

The lifecycle slide groups the steps into four phases separated by gate reviews G1–G3:

1. **Phase 1: Idea Development.** Question ID, business understanding, data acquisition (ETL / representation).
2. **Phase 2: Data Prep and Problem Exploration.** Data understanding (EDA), initial model(s) building, initial model evaluation, evaluation criteria / value metric.
3. **Phase 3: Solution Development.** Model creation & training, feature engineering and evaluation, optimization (hyperparameters) / final model development.
4. **Phase 4: Solution Deployment.** Model deployment, data drift analysis, model performance against the evaluation value metric, model drift analysis, and reports, dashboards, and products.

The key terms are organized by phase:
- **Phase 1:** prediction vs. inference, independent metric for business value, target variable and features, classification vs. regression, probabilistic interpretation, data acquisition.
- **Phase 2:** variable and data types, baseline/prevalence, scaling/normalizing/one-hot encoding, missing data, data partitioning/sampling, EDA, cross-validation.
- **Phase 3:** parameters vs. hyperparameters, thresholding, feature engineering, the bias-variance tradeoff, model evaluation, and non-parametric modelling (random state).

The bootcamp notebook adds that the lifecycle is not always linear; you may iterate through the steps multiple times.

## Phase 1 concepts

- **Prediction vs. inference.** "Almost all ML methods are focused on prediction not causation or inference", so model performance is judged on how well a model predicts, not on how much individual variables contribute.
- **Independent metric for business value.** This is a metric *independent of the model* that tells you whether the model delivers value. Examples: Netflix recommender → number of user clicks; spam blocker → number of viruses on the network; market clustering → did sales increase.
- **Target vs. features.** The target is "the variable we are trying to predict". The features are the variables used to predict it, and feature engineering can create more of them.
- **Classification vs. regression.** Classification has a binary or multi-level categorical target (examples: winning season or not, loan default or not). Regression has a continuous target (game score, amount of rain, a person's weight). A key point: classifier outputs are **percentages, not 0/1**, with one probability per category for each row.
- **Probabilistic interpretation.** Classification is "much more common" than regression and is the focus of most of the class. Think of outputs as risk measures, or as a quantification of uncertainty. A 75% probability may be enough for "will a customer open a marketing email" but not for "is this drug effective".
- **Data brainstorming.** *Data to Concept* asks whether the data supports the target: how hard it is to gather, whether it is large enough, its rate of change, and whether it is the right source. *Learning Difficulty* asks how complex or vague the target is, whether the classes are imbalanced, whether the data links clearly to the problem, whether it was used before, and what risk level is acceptable.

## Phase 2: the data-prep pipeline (cereal worked example)

The bootcamp notebook frames a Phase I question: *build a model that predicts cereal quality rating*. The independent business metric is: "Assuming that higher ratings results in higher sales, can we predict which new cereals that enter the market over the next year will perform the best?" The data is read with `pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3001/main/data/cereal.csv")`. Imports: `train_test_split` from `sklearn.model_selection`, and `MinMaxScaler` and `StandardScaler` from `sklearn.preprocessing`.

### 1. Correct variable classes
Check structure with `cereal.info()`. The slides also list `.dtypes`, `.shape`, and `.describe`. Columns at index 1, 2, 11, and 12 are converted to categorical in one step: `cereal.iloc[:, [1,2,11,12]].astype('category')` (`iloc` avoids typing column names). `cereal.dtypes` is "simpler, but does not give an index". The notebook stresses that wrong types cause errors in training (for example, math on strings) and that categorical levels should be checked for balance.

### 2. Collapse factor levels
`cereal.mfr.value_counts()` shows many small manufacturer groups. The rule of thumb is "usually don't want more than 5 groups". The large groups `['K','G']` are kept and everything else becomes `"Other"`: `cereal.mfr.apply(lambda x: x if x in top else "Other").astype('category')`. The notebook explains lambda syntax with the example `lambda a: a+10` (passing 5 returns 15); in a conditional lambda the True value comes first, then `if`, then `else` and the False value. `type` and `vitamins` look fine. `weight` is not a categorical grouping, so it doesn't matter for this step.

### 3. Scale / center vs. normalize
- **Standardize (center + z-score):** `StandardScaler().fit_transform(cereal[['sodium']])`. The double brackets pass a DataFrame (2-D), which `fit_transform` requires.
- **Normalize (min-max to [0, 1]):** `MinMaxScaler().fit_transform(cereal[['sodium']])`. The density plots of raw and normalized sodium have the same shape ("Checks out!"), which shows that the relationships are preserved.
- Applied to all numeric columns: `abc = list(cereal.select_dtypes('number'))`, then `cereal[abc] = MinMaxScaler().fit_transform(cereal[abc])`.

The slides say the course will "mostly use a min max scaler that will maintain the variance of the values but re-calculate them to be between 1 and 0". The rationale: unscaled features with larger ranges dominate learning. This matters most for distance-based methods such as k-NN and SVMs, and normalization also speeds gradient-descent convergence (see [[course/knn]]).

### 4. One-hot encoding
`category_list = list(cereal.select_dtypes('category'))`, then `cereal_1h = pd.get_dummies(cereal, columns=category_list)`. This creates one 0/1 indicator column per category level, so the model does not assume an ordering among categories.

### 5. Baseline / prevalence
Prevalence is "the proportion of a particular population found to be in the positive class", the percentage split across the target's classes. It is the **baseline the model has to beat**, and it warns that accuracy can mislead on imbalanced data (see [[course/ml-eval-metrics]]). The continuous `rating` is turned into a binary target by taking the top quartile. `cereal_1h.rating.describe()` shows the upper quartile (after normalization) starts above 0.43, so `cereal_1h['rating_f'] = pd.cut(cereal_1h.rating, bins=[-1, 0.43, 1], labels=[0, 1])` creates it. Two segments need three numbers: start, cut, and stop. The prevalence is `value_counts()[1] / len(...)`, which equals **21/(21+56) ≈ 27%**.

### 6. Drop variables and partition into Train / Tune / Test
Drop columns the model can't use: `cereal_dt = cereal_1h.drop(['name','rating'], axis=1)`. A new DataFrame is created so the working copy keeps those columns. Then call `train_test_split` **twice**:

```python
Train, Test = train_test_split(cereal_dt, train_size=55, stratify=cereal_dt.rating_f)
Tune, Test  = train_test_split(Test, train_size=.5, stratify=Test.rating_f)
```

`stratify` preserves class proportions and reduces sampling error. The notebook checks that prevalence is the same in every partition: Train 15/(40+15), Tune 3/(8+3), Test 3/(8+3), all ≈ 27%. The slides define the three partitions this way:
- **Training** builds the original model.
- **Tune** evaluates outputs after the model is modified (for example, changing k in kNN, or feature engineering).
- **Test** is the very last step, used to evaluate quality after training and tuning.

The notebook ends with a `DecisionTreeClassifier` fit and a `precision_score` cell (see [[course/decision-trees]]). As written, that cell references `X_train`, `y_train`, `X_test`, and `y_test`, which are never created in the notebook. Separating features from the target is left implicit.

## Missing data

The slides call missing data a "large area of study"; here students only need to know how to check for it and some quick fixes. First, look for a *pattern*: **MCAR** (missing completely at random) vs. **MNAR** (missing not at random). If a pattern exists, you need a strategy for it. Start with `info()` on the DataFrame; the exercise uses the **beaches** dataset. Pandas tools: `x.isna()`, `x.notna()`, combined with `loc`/`iloc`. For example, `x.loc[x.notna().all(axis='columns')]` keeps only rows with no missing values in any column.

The standalone script `example-dropna.py` (data fetched with `requests` from a data.world URL and read through `StringIO`) shows a column-level fix: `c.dropna(thresh=c.shape[0]*0.9, axis=1, inplace=True)` keeps only columns that are at least 90% non-null, which **drops columns with more than 10% missing values**. The bootcamp notebook's "Now you try!" section loads the same data.world dataset.

⚠️ Possible conflict: the Python slide deck says variables with more than "about 5%–10%" missing values should be deleted or imputed. The R-based Bootcamp II reading says "about 5%". The reading also uses R's MICE package (`md.pattern()` to look for patterns) and `complete.cases()` to drop NA rows (`df <- df[complete.cases(df), ]`), and notes that MICE can also impute. The dropna script uses a 10% cutoff.

## Phase 3 terms introduced here

- **Thresholding.** This is the probability cutoff at which a prediction becomes 1 vs. 0. The default is 50%, but you may raise it to 75% or 90% "to be extra careful".
- **Feature engineering.** "Combining or exploring different levels of variable that best work in your model." The reading says a week may be dedicated to it.
- **Model evaluation.** "Everything is evaluation." A full week is devoted to metrics later; see [[course/ml-eval-metrics]].
- Parameters vs. hyperparameters, bias vs. variance, and cross-validation appear only as slide titles in the extracted text, with no body content.

## Train/test split with logistic regression (notebook)

The `train-test-logreg` notebook shows statsmodels logistic regression and an early, manual train/test split (see [[course/logistic-regression]]):
- **Admissions data** (`LogReg.csv` from the UVADS/DS-3001 repo). `rank` is converted with `.astype('category')`. The model is fit with `sm.formula.glm("admit ~ rank+gpa+gre", family=sm.families.Binomial(), data=data).fit()`. The coefficients are in log terms, so `np.exp()` converts them. The notebook builds an ROC curve from `fittedvalues` with `metrics.roc_curve` and notes that the x-axis is FPR, not inverted specificity. A Hosmer-Lemeshow goodness-of-fit test is left as an open "How to do hl test???".
- **mtcars** practice: `vs ~ wt + disp`, with exponentiated coefficients and an ROC curve.
- **Default data** (`Default.xlsx`, read from a local path). The notebook drops `Unnamed: 0`, maps `default` Yes/No to 1/0, and makes an **80/20 split by row position**: `train = default.iloc[:8000]`, `test = default.iloc[8000:10000]`. This is unshuffled and unstratified, unlike the `train_test_split` approach above.
  - It fits `default ~ balance + income + student` (flagged "WRONG SIGN???"), then drops income (`default ~ balance + student`). Out of curiosity it also fits `income + student`.
  - It compares models with a likelihood-ratio test, `LR = 2*(llmax - llmin)` and `chi2.sf(LR, 1)`.
  - Test-set probabilities are thresholded at 0.5 with `pd.cut(pred, bins=[-1, .5, 1], labels=[0, 1])`. The misses are counted, and accuracy is `(2000 - x)/2000*100`. ROC and `roc_auc_score` follow.
  - Finally it predicts for new data: a student with a $5,000 balance has a high default probability, while a $2,000 balance "drops to 52%". It plots likelihood of default against balance from 1 to 10,000.

## R → Python conversion notes (Week 05)

| R | Python | Note |
|---|---|---|
| `read_csv` | `pd.read_csv` | Basically the same |
| `str()` | `.info()` or `.dtypes` | `.dtypes` gives only column name and type. `.info()` also gives the index. Neither previews the data the way `str()` does. |
| `[]` | `.iloc` | Index-based row/column access with many indexing options |
| `as.factor` | `.astype('category')` | Python has no factor type; categorical is the closest equivalent |
| `table()` | `.value_counts()` | Basically the same |
| `fct_collapse()` | `.apply(lambda x: x if x in top else "Other")` | No direct equivalent; uses apply plus a lambda |
| `scale()` | `StandardScaler().fit_transform(df[['column']])` | z-scores; the argument must be DataFrame-like (double brackets) |
| (minmax) | `MinMaxScaler().fit_transform(df[['column']])` | 0–1 normalization; pass a list of column names to do several at once |
| `select_if(df, is.numeric)` | `.select_dtypes('number')` | Can also pass `'category'` and other types |
| `one_hot()` | `pd.get_dummies()` | Columns must be specified in Python |
| `fivenum()` | `.describe()` | `describe()` also gives count, mean, and std |
| `cut()` | `pd.cut()` | Bins a continuous variable (here, to make a binary target for prevalence) |
| indexing | `.drop()` | Must give name(s) and axis: rows = 0, columns = 1 |
| `caret::createDataPartition` | `train_test_split()` | Use it twice to get three sets; `stratify=` preserves proportions |

The older R reading did the same exercises on `mtcars` (e.g., "what is the prevalence of `vs`?"), using `str()`, `class()`, `names()`, `length()`, `dim()`, the gradDescent package's minmax function, and `createDataPartition()` "twice to create a sample, then a sample of a sample". It also lists R's atomic types: numeric, integer, complex, character, factor, and logical.

## Extra material: Booking.com lessons

The deck closes with Booking.com case lessons:
- **Swiss Army Knife:** models are highly adaptable.
- **Offline Health Check:** model outputs are tested with randomized controlled trials against business metrics such as customer conversion. "Increase model performance doesn't necessarily translate to better gain in value."
- **Make a Target Before you Shoot:** for example, what does "date flexibility" mean? This connects to learning difficulty, data to concept, and selection bias.
- **Speed Kills:** latency.
- **Keep a watchful eye:** monitoring.
- RCT experimentation is "ingrained into Booking.com culture".

A final slide works a 1-D k-means example: D = {2,4,10,12,3,11,20,25,30}, k = 2, starting means 3 and 4, which converges to {2,3,4,10,11,12} / {20,25,30} with means 7 and 25. See [[course/kmeans-clustering]].

## Acquiring data from APIs (MLBook chapter)

The MLBook file labeled as the data-prep pipeline chapter is actually **"Acquiring Data from APIs"** by Jon Kropko. It covers data acquisition, the Phase 1 step that comes before prep:
- **What an API is.** A web application has a frontend, a backend, and an API between them; the chapter uses a restaurant analogy, with the API as the waiter. APIs usually exchange JSON. *Encapsulation* lets providers control what data is shared, and Taina Bucher argues that "APIs have politics".
- **REST.** Client-server separation, stateless, cacheable, uniform interface, layered. The HTTP verbs are GET, POST, PUT, PATCH, and DELETE (CRUD). Status code 200 means success and 404 means not found.
- **Building a request URL.** A URL is *root/endpoint?param=value&key=value*. The NASA Image Library example: `https://images-api.nasa.gov/search?q=moon&media_type=image` returns 8626 hits, the first being PIA12235 "Nearside of the Moon". It is retrieved with `/asset/PIA12235`, a path parameter rather than `?`.
- **`requests`.** The Wikipedia example calls `requests.get(url, params=p_dict)` (with `action=query`, `prop=revisions`, `titles=University_of_Virginia`, `rvslots=*`, `rvprop=content`, `formatversion=2`, and `format=json`), parses the response with `json.loads(r.text)`, and drills into `['query']['pages'][0]['revisions'][0]['slots']['main']['content']`.
- **API keys.** The Census API allows up to 50 variables per query and 500 queries per IP per day without a key. Keep keys secret by putting them in a `.env` file (e.g., `CensusKey=...` with no spaces or quotes) and loading them with `dotenv.load_dotenv()` and `os.getenv('CensusKey')`. The example pulls `GEO_ID,POP` for `state:*` from `.../data/2019/pep/charagegroups` and builds a DataFrame with `pd.DataFrame(statepop[1:], columns=statepop[0])`.
- **Wrapper libraries.** `tweepy` for Twitter uses OAuthHandler with four keys and `tweepy.Cursor(api.search, q='#uva').items(30)`. `googlemaps` uses `gmaps.geocode(...)` on the Dell's address, 60 Bonnycastle Dr, and reads `[0]['geometry']['location']`.

## Labs / assignments

- **ML Bootcamp Lab** (`ml-bootcamp-lab.md`). Build **two data-prep pipelines** in a new GitHub repo. Suggested files: the assignment file, a Python file for questions 1–3, and a Python file for question 4. The datasets are **College Completion** (`cc_institution_details.csv`, with a Kaggle data dictionary) and **Job Placement** (`Placement_Data_Full_Class.csv`, where the data dictionary must be inferred from the column names and Kaggle comments).
  - *Steps 1–2:* For each dataset, write a question and an independent business metric. Then prepare the data: correct the variable types, collapse factor levels, one-hot encode, normalize the continuous variables, drop unneeded variables, create the target if needed, calculate prevalence, and create the Train/Tune/Test partitions.
  - *Step 3:* What do your instincts say about the data, and what are you worried about?
  - *Step 4:* Write reusable **functions** that produce the train and test sets for each problem. These should be a series of smaller functions called in sequence, "essentially creating a DAG for your data prep steps", not one monolithic function.
- **"Now you try!"** at the end of the bootcamp notebook loads the Job Placement data and the data.world dataset for practice.
- The College Completion data returns in the KNN graduation lab ([[course/knn]]).

## Related topics

- [[course/ml-foundations]]: broader ML concepts that this week's lifecycle and terminology introduce
- [[course/python-basics]] and [[course/dev-environment]]: the pandas and VS Code skills this pipeline relies on
- [[course/knn]]: the first model built on this pipeline; scaling and one-hot encoding are mandatory there
- [[course/logistic-regression]]: the statsmodels GLM workflow previewed in the train-test notebook
- [[course/ml-eval-metrics]]: prevalence, thresholds, ROC/AUC, and why accuracy misleads under imbalance
- [[course/decision-trees]]: the classifier stubbed at the end of the bootcamp notebook
- [[course/kmeans-clustering]]: unsupervised learning, plus the k-means example in the slide appendix
- [[course/ml-bias]]: selection bias and class imbalance raised in the data-brainstorming and Booking.com slides
- [[course/final-project]]: applies the full lifecycle end to end

## Sources

- `raw/course/DS-3021/data-prep/2024-10-29-example-dropna.py`: dropna example script (drop columns with more than 10% missing)
- `raw/course/DS-3021/data-prep/2024-10-29-ml-bootcamp-data-prep-slides.pptx`: lecture slides ("Machine Learning Overview", Python version)
- `raw/course/DS-3021/data-prep/2024-10-29-ml-bootcamp-II-reading.pdf`: ML Bootcamp II reading (PDF of the older, R-based version of the same deck)
- `raw/course/DS-3021/data-prep/2024-10-29-ml-bootcamp-r-to-python-conversion.docx`: Week 05 R→Python conversion notes
- `raw/course/DS-3021/data-prep/2024-10-29-train-test-logreg-notebook.ipynb`: train/test split and logistic regression notebook
- `raw/course/MLBook/data-prep/2026-05-21-acquiring-data-from-apis.ipynb`: MLBook chapter (contents are "Acquiring Data from APIs" by Jon Kropko)
- `raw/course/MLBook/data-prep/2026-05-21-ml-bootcamp-lab.md`: ML Bootcamp lab assignment
- `raw/course/MLBook/data-prep/2026-05-21-ml-bootcamp-notebook.ipynb`: ML Bootcamp notebook (cereal data-prep walkthrough)
