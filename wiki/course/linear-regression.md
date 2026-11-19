# Linear Regression

The course's first *parametric* supervised model, taught in mid-March (materials dated 3/11–3/17) right before [[course/logistic-regression]]. The lecture notebook ("Linear Models — Foundations of Machine Learning") frames the transition explicitly: [[course/knn]] and [[course/kmeans-clustering]] illustrated regression vs. classification and supervised vs. unsupervised learning, but each has essentially one tuning parameter (`k`) and "almost no explanation of their results." Linear models instead *optimally weight* the explanatory variables to predict the outcome, which makes them "extremely powerful and easily interpreted tools." The instructor stresses this is an entry-level treatment focused on **prediction** and the intuition/mechanics of **Ordinary Least Squares (OLS)**. The MLBook chapters add a regression overview, evaluation metrics, an `sklearn` walkthrough, a `statsmodels` (OLS) walkthrough, multiple features, and polynomial features.

> ⚠️ Possible conflict — see "Assumptions" and "Dummy variable trap" below (sources disagree on whether OLS assumptions are required, and on the `get_dummies` argument name).

## What regression is

**Regression** is a supervised learning task where the target is **continuous** (examples given: house prices in dollars, life expectancy in years, employee salary, distance to the nearest galaxy in light years; in finance, stock prices, risk, revenue). The general form is

$$y = f(X) + \epsilon$$

where `f(X)` may be linear or non-linear and ε is the error/residual. The regression-types chapter names three models:

- **Linear regression (OLS)** — straight-line relationship; minimizes the difference between actual and predicted values. One feature: `y = β0 + β1·X + ε` (β0 = value of y when X = 0; β1 = change in y for a one-unit change in X). Multiple features: `y = β0 + β1X1 + … + βnXn + ε`.
- **Lasso regression** — adds an **L1** penalty on the absolute sum of coefficients; shrinks some coefficients to exactly zero, so it performs **feature selection**.
- **Ridge regression** — adds an **L2** penalty on squared coefficients; forces coefficients small but not zero; useful when **multicollinearity** is present.

Python classes: `LinearRegression` (sklearn) and `OLS` (statsmodels) for OLS; `Ridge` and `Lasso` from sklearn. Lasso/Ridge are only introduced here; the lecture says data-driven model selection via cross-validation (e.g. LASSO) comes "later on in the course."

## Linear algebra foundations (lecture notebook)

- **Dot product** `x·b = Σ x_k b_k` (example `y=(3,-5,7)`, `x=(2,4,-6)` via `np.inner`). **Covariance is the statistical dot product**, `(x−x̄)·(y−ȳ)/N`.
- **Angle and correlation**: `cos θ = x·b / (√(x·x)√(b·b))`; zero dot product = **orthogonal**. For de-meaned variables the cosine *is* the correlation — "the sample correlation between two variables is really the angle between them." Regression coefficients are roughly `cov(x,y)/σx²`.
- **Matrix multiplication** `X·b` stacks row dot products (`np.matmul(X,b)` or `X @ b`); NAs break it, one reason tidy data matters.
- **Setup**: N×K matrix X, outcome y, model `y = X·b`; predicting is just `x̂·b` — "picking the weights b is the hard part." Neural networks are non-linear compositions of linear models.

## OLS: minimizing the sum of squared errors

- **Residual**: `e_i = y_i − ŷ_i = y_i − x_i·b`.
- **SSE** = Σ e_i² (squaring counts all errors as bad and large ones as worse). **MSE** = SSE/N; **RMSE** = √MSE. From a model-fitting view these are "fundamentally the same thing."
- Other possible targets: **MAD** (mean absolute deviation, more robust to outliers) and **WAD** (worst absolute deviation, `max|y−ŷ|`). Why SSE? "Probably because we can easily use calculus to minimize it."
- **First-order conditions** for `SSE(b0,b1) = Σ(y_i − b0 − b1x_i)²` (set derivatives to zero) give:
  - `b0* = ȳ − b1*·x̄`
  - `b1* = Σ(y_i − ȳ)x_i / Σ(x_i − x̄)x_i`
- Two key properties at the optimum: **the average residual is zero**, and **the residual is orthogonal to (uncorrelated with) x** (`e·x = 0`).
- The notebook implements this as `slr(x, y)`, returning `b0`, `b1`, `y_hat`, `residuals`. Simulated demo: `np.random.seed(500)`, N = 30, `x = 5 + 2·N(0,1)`, noise N(0,3), true b0 = −1, b1 = 3; residuals are drawn as red vertical lines ("SSE: sum of squared red lines").
- **Partialing out / projection**: `y = e + x·b` — the prediction holds the signal, the residual holds the noise, and they are uncorrelated.

### Multiple linear regression

The SSE generalizes to `Σ(y_i − X_i·b)²`. It can be minimized computationally (gradient descent) or with linear algebra: **`b* = (X'X)⁻¹X'y`** (the normal equations). The notebook's `mlr(X, y)` solves `np.linalg.solve(X.T@X, X.T@y)` and returns coefficients, predictions, residuals, SSE, and R²; an `(Intercept)` column of 1s is added manually.

**How MLR works ("the deep idea in regression")** — the partialing-out procedure: to get the coefficient on x_k, (1) regress y and x_k on all other variables to get residuals r_y and r_xk; (2) regress r_y on r_xk. The slope equals the MLR coefficient: `b_k = cov(r_y, r_xk)/σ²(r_xk)`. MLR coefficients reflect only the variation in y that x_k explains *after* removing what the other features explain. The notebook verifies this "laboriously" with `slr()` for age and mileage. The makeup script states the same idea for interpretation: each coefficient is the change in y for a one-unit change in that predictor **holding all others constant (ceteris paribus)**. The assignment's Q1.6 tests this ("what does the coefficient on `Review Scores Rating` mean if it changes from part 4 to 5?").

### R² (coefficient of determination)

`R² = 1 − SSE/SST`, where SST = Σ(y_i − ȳ)² is the error from predicting with just the mean. It is the reduction in SSE from using the features rather than the mean. Caveats from the lecture: **adding variables always raises in-sample R²** but doesn't necessarily improve out-of-sample prediction — "R² isn't an end in itself." The MLBook chapter adds that R² ranges 0–1 in usual use (1 = perfect fit, 0 = explains nothing) but **can be negative**, meaning the model does worse than predicting the mean. (The intro notebook's metric table describes R² as "0 to 1"; the negative case shows up on test data, e.g. the GDP example below.)

## Regression evaluation metrics

| Metric | Formula / meaning | Notes from sources |
|---|---|---|
| R² | 1 − SSE/SST | higher (closer to 1) is better |
| MAE | mean of \|y − ŷ\| | same units as target (e.g. dollars) |
| MSE | mean of (y − ŷ)² | penalizes large errors more; squared units |
| RMSE | √MSE | back in target units; usually differs from MAE |

Except for R², the goal is to minimize. sklearn functions: `r2_score`, `mean_absolute_error`, `mean_squared_error` (RMSE via `mse ** 0.5` or `np.sqrt`), called as `(y_true, y_pred)` on training data for training metrics or test data for test metrics. The notebooks also compute MAE/MSE/R² by hand from an `Error` column. Helper functions appear in two notebooks: `print_metrics()` (intro notebook) and `regression_report()` (multiple-features chapter).

## What makes a model "linear"?

A model is linear because the **coefficients** enter linearly: `ŷ = b1x̂1 + … + bJx̂J`. Transforming regressors (`log(x)`, `x²`) just creates new variables — **"transformations of the regressors don't make the model nonlinear."** Even a weirdly written coefficient like `√asinh(b_j)` can be relabeled. (Assignment Q0.1: "What makes a model 'linear'? 'Linear' in what?")

## Worked examples as taught

### Used cars (`USA_cars_datasets.csv`)

- **Intro notebook (3/11)** — pipeline "train/test split → fit → predict → evaluate." Cleaning: drop `price == 0` rows, drop prices above the 99th percentile. Three models with `train_test_split(test_size=0.2, random_state=42)`: simple LR `price ~ year`; multiple LR `price ~ year + mileage` (correlation heatmap first); and `year + mileage + brand` restricted to the **top 10 brands**, using a `Pipeline` with a `ColumnTransformer` (`'passthrough'` for numerics, `OneHotEncoder(drop='first', handle_unknown='ignore')` for brand). The Pipeline **prevents data leakage** because the encoder learns categories only from training data. Plots: regression line, residual plot, predicted-vs-actual with a 45° line; final bar charts compare R² and RMSE. Note that sklearn needs a **2-D X** (`df[['year']]`).
- **Lecture notebook (3/12)** — car **age** (`max(year) − year`) predicting price. The instructor's rough advice: regression approximates the **conditional expectation function** E[y|x]; it is "working" if the line tracks the average y as x varies, and fails mainly due to (1) outliers or (2) non-linearity of the CEF. Steps: apply the **inverse hyperbolic sine** (`np.arcsinh`) to price and age, use boxplots to drop outliers (`price_ihs > 9`, `age_ihs < 4`), fit, and map back to original units with `np.sinh`. The same regression on the raw, uncleaned data is titled "Looks Bad." A second case study does price vs. mileage (keep `9 < mileage_ihs < 13`, `price_ihs > 9`). MLR combines `age_ihs` and `mileage_ihs`, then adds squared terms (`mileage_ihs_sq`, `age_ihs_sq`) to capture non-linear aging; results are compared with `LinearRegression().fit(X, y)` (`intercept_`, `coef_`, `score()` = R²). Finally, "Cars, Again" regresses price on brand dummies with `fit_intercept=False` and then adds winsorized `mileage_ihs`.

Other example datasets listed: `airbnb_hw.csv`, heart failure clinical records, and `pierce_county_house_sales.csv`; other applications include bond/sentence prediction and student loan debt.

### Facebook metrics — the makeup regression script

`2026-03-12-makeup-regression-script.py` is an in-class live-coding script on the UCI **Facebook Performance Metrics** dataset (`fetch_ucirepo(id=368)`), target **`Total Interactions`** (rows with a missing target dropped). Eight sections:
1. **Kernel density plot** (`.plot.kde()`), a smoothed histogram — Total Interactions and Page total likes are heavily right-skewed.
2. **Dummy variables** with `pd.get_dummies(df['Type'], prefix='type')` (Photo, Status, Link, Video).
3. **No-intercept regression**: `fit_intercept=False` forces the line through the origin; "unless your domain knowledge justifies it, always keep the intercept," since forcing it biases the slope.
4. **Multivariate regression** on `Page total likes`, `Post Month`, `Post Weekday`, `Post Hour`, `Paid`; a low R² suggests these numeric features alone aren't very informative.
5. **Log and arcsinh transforms**: `log(x)` needs x > 0 (script uses `np.log(x + 1)`); `arcsinh` handles zeros and negatives and ≈ `log(2x)` for large x.
6. **Dummy variable trap**: `drop_first=True` makes **Link** (first alphabetically) the reference; `type_Photo` is then the expected difference vs. a Link post.
7. **PolynomialFeatures** degree 1, 2, 3 on `log_page_likes`, compared by test R² — "higher degree is not always better."
8. **True vs. predicted plot** with an 80/20 split (`random_state=42`), test RMSE and R²; points should hug the 45° line, while a fan shape (heteroskedasticity) or curve signals a limitation. Extreme values are hard to predict, "common with social media data."

### Student grades (MLBook, sklearn and statsmodels)

Both chapters use `grades.csv` (from the `prof-rossetti/python-for-finance` repo) to predict `Grade` from `StudyHours`. Rows with nulls are dropped (e.g. "Ali" has neither; "Juan" has hours but no grade, so he can't be used for training but could be predicted at 8 hours), and study hours below 2 are removed as outliers. Split: `train_test_split(x, y, random_state=99)`. Teaching points: sklearn wants `x = df[["StudyHours"]]` (passing a 1-D Series gives `ValueError: Expected 2D array`); attributes ending in `_` (`coef_`, `intercept_`) exist only after fitting; wrap coefficients in `Series(model.coef_, index=model.feature_names_in_)`; using the trained model on new data (`StudyHours` 0–20) is called **inference**. In-class extension: add a synthetic correlated `gpa` feature (clipped to [2.0, 4.0]) and see whether RMSE and the study-hours coefficient change.

**statsmodels OLS**: you must add the intercept column with `sm.add_constant(x)`; fit with `sm.OLS(y_train, x_train, missing="drop").fit()`; `results.summary()` shows coefficients, t-statistics and p-values (p < 0.05 → reject the null that the coefficient is zero); `results.params`, `fittedvalues`, `resid`. Training-summary R² describes *training* fit only. `results.get_prediction(x_test).summary_frame(alpha=0.05)` returns `mean`, `mean_se`, **confidence interval** (`mean_ci_lower/upper`, for the *average* value, "average apple weight between 140 and 160 g") and **prediction interval** (`obs_ci_lower/upper`, for an *individual* observation, "an apple might weigh 120–180 g"). Test results match the sklearn implementation.

### California housing — multiple features and feature selection

`fetch_california_housing(as_frame=True)` (1990 census, one row per block group of roughly 600–3,000 people); columns renamed to `income`, `age`, `rooms`, `bedrooms`, `population`, `occupants`, `latitude`, `longitude`, target `value` (median house value in $100K). Considerations: trade **performance vs. complexity** (fewer features means faster, cheaper, more interpretable); scaling is "not as important" for linear regression; beware **collinearity** (unstable coefficients, hard interpretation). EDA: violin/histogram (expensive outliers), income-vs-value scatter with OLS trendline, a map showing coastal homes are priciest, a pairplot on a 100-row sample, and a **Spearman** correlation heatmap — income is most correlated with value; rooms/bedrooms/income are inter-correlated. Features z-scored, `random_state=99`. The all-features baseline gets **test R² ≈ 0.61**; the largest coefficients are income (≈ +0.8) and latitude/longitude (≈ −0.9 each). A `train_eval(df, features=...)` helper compares subsets; **income + latitude + longitude** gives about the same performance as all features, so it is the suggested final set.

## Feature engineering

The lecture's options for expanding the feature space: **interaction terms** (`x1·x2`, e.g. mileage × age), **polynomial families** (x², x³, …), **log / inverse hyperbolic sine** transforms, **max-min or z-score normalization**, and PCA or splines. The more complex the feature space, the greater the **overfitting** risk; the "risk is that your model turns to mush (multicolinearity)."

- **`PolynomialFeatures`** (`sklearn.preprocessing`): `expander = PolynomialFeatures(degree=2, include_bias=False)`, `Z = expander.fit_transform(X)`, `names = expander.get_feature_names_out()`, then `pd.DataFrame(Z, columns=names)`. `interaction_only=True` gives only cross-products. Demo columns: `apples`, `joules`, `stocks`. "We rarely put one-hot variables into the polynomial expander" unless deliberately.
- **One-hot encoding**: `pd.get_dummies(var)` (add `dtype='int'` if you don't want booleans) or sklearn's `LabelBinarizer` (`OneHotEncoder` "can be more complex to use"). Demo: a `bird` column (Cardinal, Oriole, Robin).
- **Combining**: `pd.concat([df1, df2, ...], axis=1)`.
- **Outliers**: detect, create an outlier dummy, and **winsorize** (replace values beyond the boxplot whiskers, Q1 − 1.5·IQR and Q3 + 1.5·IQR, with the whisker values); removing them is **trimming**. The notebook defines `winsorize(x)` and applies it with `.apply`.

### Polynomial regression for time series (GDP)

The MLBook chapter fetches quarterly US **GDP** from FRED (`get_data_fred("GDP", start="1900-01-01")`, billions of dollars, seasonally adjusted annual rate), creates a `time_step` index, and uses a **sequential 80/20 split** (first 80% train, last 20% test, not random) because it is time-series data. A plain linear trend gets **training R² ≈ 0.85 but a negative test R²**: worse than predicting the mean, and it fails to generalize. A Lowess trendline suggests curvature, so `PolynomialFeatures(degree=2)` (columns `const`, `time_step`, `time_step_squared`) is used. The quadratic model "does a lot better generalizing to the test data."

## Log transformations and interpreting coefficients (reading)

The assigned reading (Kenneth Benoit, LSE, "Linear Regression Models with Logarithmic Transformations," 2011) uses natural logs. Logging makes the effective relationship non-linear while keeping a linear model, and turns right-skewed variables (e.g. expenses) closer to normal. Logs are **monotone**, unlike adding X², which can reverse direction.

| Model | Form | Interpretation of β̂ |
|---|---|---|
| Linear | Y = α + βX | change in Y per one-unit change in X |
| Linear-log | Y = α + β log X | β̂ = change in Y when X is multiplied by e (≈ +172%); p% increase in X → β̂·log((100+p)/100); 1% → ≈ β̂/100 |
| Log-linear | log Y = α + βX | one-unit increase in X multiplies Y by e^β̂; c units → e^(cβ̂); small β̂ → ≈ 100·β̂ % change |
| Log-log | log Y = α + β log X | **elasticity**: p% change in X multiplies Y by e^(β̂·log((100+p)/100)) |

Worked examples (Stata output):
- **Linear-log**: % urban (1995) on log GDP per capita, N = 132, R² = 0.5498. β̂ = 10.43, so multiplying GDP/cap by e adds 10.43 points, a 1% increase adds ≈ 0.104, and a 10% increase adds 10.43 × 0.09531 ≈ 0.994 points.
- **Log-linear**: log GDP/cap on % urban: β̂ = 0.052709. Each point of urbanization multiplies income by e^0.0527 = 1.054 (+5.4%), close to the 5.3% quick rule.
- **Log-log**: log infant mortality on log GDP/cap, N = 194, R² = 0.6781. β̂ = −0.498, so multiplying income by e multiplies IMR by 0.607 (−39.3%), a 1% rise cuts IMR by 0.5%, and a 10% rise cuts IMR by 4.6%.

The assignment's tip box adds that a **log** transform suits strictly positive, right-skewed data. **arcsinh** handles positive, negative, and zero values. **Normalization** rescales to a common range (typically 0–1) for scale-sensitive algorithms and does *not* make data normally distributed.

## Dummy variable trap

With an intercept plus a dummy for every category, one regressor is a linear combination of the others (`type_Photo + type_Status + type_Link + type_Video = 1`). This is **perfect multicollinearity**: the matrix is singular and some coefficients aren't defined. There are two fixes: drop one level (the **reference category**, absorbed into the intercept, with remaining coefficients interpreted *relative* to it) or keep all dummies and use `LinearRegression(fit_intercept=False)`. With no intercept and one categorical regressor, the coefficients equal the **group means** (assignment Q1.2–1.3). The assignment's note says not to use `fit_intercept=False` if you have already dropped a column. "Most stats packages hide this… you have to handle it with pandas/sklearn more explicitly."

⚠️ Possible conflict: the lecture notebook says to use `pd.get_dummies(..., drop_constant=True)`, while the makeup script and the assignment both use **`drop_first=True`** (and the intro notebook uses `OneHotEncoder(drop='first')`). The two `get_dummies` argument names disagree. The other sources' code all uses `drop_first`.

## Model selection, assumptions, and generalizations

- **Underfit** = too few variables to capture the relationships; **overfit** = too many variables, exploiting idiosyncrasies of the training data. Statistics offers AIC, BIC, and Mallows' Cp, but assumes the analyst knows the true model; ML uses data-driven cross-validation (e.g. LASSO) later in the course. The hedonic-pricing and assignment questions both ask what was learned about overfitting, underfitting, and model selection. High training but low test accuracy (assignment Q0.4) is the overfitting symptom.
- **Classical OLS assumptions** (lecture's framing): linearity, homoskedasticity, conditional independence of errors, and normality of errors are what make OLS **BLUE** for a "true" b and make the z-test valid. They are *not* needed to find the **best linear predictor**. "You don't need permission to run OLS and make predictions, you need permission to run OLS and interpret the coefficients."
  - ⚠️ Possible conflict: the logistic-regression slides (see [[course/logistic-regression]]) list "Model Assumptions" for simple linear regression (Y normally distributed, linearity, homoscedasticity, no multicollinearity) as part of the model-selection process, without this prediction-vs-inference distinction.
- **Generalized linear models**: for a binary 0/1 target, OLS can predict outside [0,1], so use **logit or probit**; for count data, use **Poisson or negative binomial**. These are fit by maximum likelihood / GLM. This is the bridge to [[course/logistic-regression]].

## Labs and assignments

- **Hedonic Pricing** (`2025-02-13-hedonic-pricing-notebook.ipynb`): predict an asset's price from its characteristics using `pierce_county_house_sales.csv`, `cars_hw.csv`, or `airbnb_hw.csv` (or your own data). Steps: EDA, transform categoricals, ~80/20 split, simple models with no transformations (RMSE and R² on train and test), then a more complex model with transformations and interactions, interpret the intercept, slopes, and dummy coefficients, and summarize lessons on over- and underfitting.
- **Lab IV: Linear Models** (`regression-assignment.ipynb`, "from Prof. Terry Johnson"):
  - **Q0**: conceptual questions (what is linear; dummy interpretation as a "trick question" about the intercept; can linear regression classify; train-high/test-low; two ways to add non-linearity, citing the inria scikit-learn MOOC page; interpreting intercept, slope, and dummy coefficients).
  - **Q1**: NYC Airbnb `Q1_clean.csv` (`Price`, `Review Scores Rating`, `Neighborhood ` *with a trailing space*, `Property Type`, `Room Type`). Group means by borough and KDE plots of price and log price, dummy regressions with and without an intercept, an 80/20 split adding rating then property type, and an optional neighborhood × property-type interaction.
  - **Q2**: `cars_hw.csv` (`Price`, `Make`, `Make_Year`, `Mileage_Run`, `Body_Type`, `Fuel_Type`, `Transmission`, etc.). Clean and transform, KDE by `Make`, numeric-only vs. dummy-only vs. combined models, `PolynomialFeatures` until test R² goes negative, a predicted-vs-true plot, and a residual KDE.
  - **Q3**: your own dataset (data.gov, kaggle, data.world).

## Related topics

- [[course/logistic-regression]] — the GLM extension for binary targets, taught immediately after.
- [[course/ml-foundations]] and [[course/data-prep]] — train/test splitting, scaling, one-hot encoding, and outlier handling used throughout.
- [[course/dt-regression]] — a non-linear, tree-based alternative for continuous targets, evaluated with the same RMSE/R² metrics.
- [[course/knn]] / [[course/kmeans-clustering]] — the low-parameter models the lecture contrasts linear models against.
- [[course/pca]] — listed among feature-space expansion tools.
- [[course/ml-eval-metrics]] — classification metrics, which replace regression metrics once the target becomes binary.

## Sources

- `raw/course/DS-3021/linear-regression/2025-02-13-hedonic-pricing-notebook.ipynb` — hedonic pricing assignment
- `raw/course/DS-3021/linear-regression/2026-03-11-linear-regression-intro-notebook.ipynb` — intro notebook: simple to multiple regression on used cars with sklearn
- `raw/course/DS-3021/linear-regression/2026-03-12-linear-regression-lecture-notebook.ipynb` — main lecture notebook (linear algebra, OLS derivation, MLR, feature engineering, assumptions)
- `raw/course/DS-3021/linear-regression/2026-03-12-makeup-regression-script.py` — in-class live-coding script (Facebook metrics)
- `raw/course/DS-3021/linear-regression/2026-03-17-log-conversion-reading.pdf` — Benoit reading on log transformations
- `raw/course/MLBook/linear-regression/2026-05-21-regression-types-and-evaluation.ipynb` — regression types (OLS/Lasso/Ridge), metrics, GLM/logistic preview
- `raw/course/MLBook/linear-regression/2026-05-21-linear-regression-overview.ipynb` — sklearn walkthrough (student grades)
- `raw/course/MLBook/linear-regression/2026-05-21-ols.ipynb` — statsmodels OLS walkthrough (student grades)
- `raw/course/MLBook/linear-regression/2026-05-21-multiple-features.ipynb` — multiple features and feature selection (California housing)
- `raw/course/MLBook/linear-regression/2026-05-21-polynomial-regression.ipynb` — polynomial features for GDP time series
- `raw/course/MLBook/linear-regression/2026-05-21-regression-assignment.ipynb` — Lab IV: Linear Models
