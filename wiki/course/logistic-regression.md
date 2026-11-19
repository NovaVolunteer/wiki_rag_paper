# Logistic Regression

Taught right after [[course/linear-regression]] (slides dated 3/19). The deck is titled "Generalized Linear Models: Logistic Regression." Logistic regression is the course's bridge from regression to classification. It keeps the linear predictor `β0 + β1X1 + … + βnXn` from linear regression, but passes it through a **link** so the output becomes a probability for a **binary** outcome. Despite the name, it is a **classification** algorithm. Two sources cover it: the lecture slides (model-selection framing, GLM theory, odds ratios, and an R-based in-class exercise) and the MLBook "Logistic Regression Overview" chapter (sigmoid, log-odds, loss, gradient descent, and an sklearn walkthrough with evaluation metrics).

> ⚠️ Possible conflict — terminology: the slides distinguish the **General** Linear Model (GLM) from the **Generalized** Linear Model (GLiM) and place logistic regression in the GLiM family. The MLBook chapter (and the GLM preview at the end of the regression-types chapter) calls logistic regression a "Generalized Linear Model (**GLM**)". Same idea, different abbreviations: on the slides, "GLM" means the *general* (normal/identity) linear model.

## Where it fits: the model-selection process (slides)

The slides open with a six-step process for choosing a model:
1. Objective of the model
2. Model structure
3. Model assumptions
4. Parameter estimates and interpretation
5. Model fit (goodness-of-fit tests)
6. Model selection

As a reference point they apply it to **simple linear regression**:
- **Objective**: model the expected value of a continuous Y.
- **Structure**: ŷ = b0 + b1x + ….
- **Assumptions**: Y normally distributed, a linear relationship between the dependent and independent variables, homoscedasticity (the slide adds "errors normal"), and no multicollinearity among the independent variables.
- **Estimates**: b0 = intercept and b1 = slope.
- **Fit**: R², residual analysis, F statistic.
- **Selection**: decide which variables to include based on fit.

(The linear-regression lecture argues these assumptions are only needed to *interpret* coefficients, not to predict. See the conflict note on [[course/linear-regression]].)

## General vs. Generalized Linear Models

A **Generalized Linear Model (GLiM)** is less restrictive than the general linear model. It adds a **link function** that connects the linear predictor Xβ to the mean of Y's distribution. This allows non-linear responses while still producing a linear equation with consistent variances. Which model you choose depends on the **distribution of the dependent variable**.

GLiM assumptions (as listed on the slides):
- Y need **not** be normal, but must come from the **exponential family** (e.g. binomial, Poisson).
- Linearity is assumed between the *link-transformed* response and the explanatory variables, not between Y itself and X.
- Homogeneity of variance does not need to hold.
- Parameters are estimated by **maximum likelihood estimation (MLE)** rather than OLS, so the model relies on large-sample approximations.
- Goodness-of-fit tests need sufficiently large samples.

Each GLiM has three components:
- **Random**: the probability distribution of Y.
- **Systematic**: the explanatory variables.
- **Link**: connects the random component to the systematic one.

| Model | Random | Link | Systematic |
|---|---|---|---|
| Linear regression | Normal | Identity | Continuous |
| ANOVA | Normal | Identity | Categorical |
| ANCOVA | Normal | Identity | Mixed |
| **Logistic regression** | **Binomial** | **Logit** | **Mixed** |
| Loglinear | Poisson | Log | Categorical |
| Poisson regression | Poisson | Log | Mixed |
| Multinomial response | Multinomial | Generalized logit | Mixed |

The linear-regression materials motivate the same move. OLS on a 0/1 target can predict values below 0 or above 1, so you use logit or probit. For count data you use Poisson or negative binomial.

## Definition and uses

The slides define logistic regression as "a type of generalized linear model … used to predict an event based on known factors when using a **dichotomous** dependent variable." It predicts a binary outcome (1/0, Yes/No, True/False) by estimating the probability of the event with a **logit** function. The slides add that it "has grown in popularity significantly and could in the future become the most commonly used form of regression."

Example questions:
- From the slides: will a customer buy a product (based on age, gender, geography, and other demographics), loan default, and political success or failure.
- From MLBook: disease diagnosis, credit risk/default, customer churn, spam filtering, and university admissions.

How it differs from earlier regressions (slides): it revolves around the **odds ratio**, and the dependent variable is categorical. It is usually dichotomous, but it "can have multiple categories making it **multinomial** (contrary to the popular belief)."

## The sigmoid function

Linear regression outputs any real number, but classification needs a probability between 0 and 1. The **sigmoid (logistic) function** maps any real z into (0, 1):

$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad z = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n$$

$$P(y=1 \mid X) = \frac{1}{1+e^{-(\beta_0 + \beta_1X_1 + \dots + \beta_nX_n)}}$$

Properties:
- σ(z) → 1 as z → ∞.
- σ(z) → 0 as z → −∞.
- σ(0) = 0.5.
- It is smooth and differentiable everywhere, which matters for gradient-based optimization.
- It is S-shaped.

The chapter plots it over z ∈ [−10, 10] with a dashed "decision boundary (p = 0.5)" line.

## Odds, log-odds, and the odds ratio

- **Odds** = p / (1 − p). Example: P(rain) = 0.75 gives odds of 0.75/0.25 = 3, so rain is 3 times as likely as no rain.
- **Log-odds (logit)** = log(p / (1 − p)). Logistic regression models the **log-odds as a linear function** of the features:
  $$\log\frac{P(y=1\mid X)}{1-P(y=1\mid X)} = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n$$
  Exponentiating and rearranging recovers the sigmoid. This is why it counts as a generalized linear model: a *transformation* of the probability is linear in the features.
- **Odds ratio (slides' worked example)**:

  | | Show A | Show B |
  |---|---|---|
  | Male | 200 | 100 |
  | Female | 50 | 150 |

  The odds that a male watches Show A are 200/100, or 2 to 1. For a female they are 50/150, or 1 to 3. The odds ratio is (2/1)/(1/3) = **6**, so "males are 6 times more likely to be watching Show A than females."

### Interpreting results (slides)

The model produces **log odds**, which "usually need to be converted to be useful." In the R exercise you exponentiate them with `exp(coef(model))` to get odds ratios. You run significance tests on the adjusted odds ratio (AOR), which plays the role of a regression coefficient. The slides list two ways:
- Basic p-values or confidence intervals.
- **ROC/AUC** ("hit rate"): how well the independent variables classify individuals into the categories of the dependent variable.

In the sklearn example, coefficients are read on the log-odds scale. Because the features are standardized, "a one standard deviation increase in HoursStudied increases the log-odds of passing by β₁."

## Training: loss and optimization (MLBook)

- **Binary cross-entropy (log loss)** for one observation is `L(y, p̂) = −[y log p̂ + (1−y) log(1−p̂)]`. Over m examples, `J(β)` is the average of these losses.
- The loss heavily penalizes confident wrong predictions. It is derived from **maximum likelihood** (Bernoulli outcomes) and is **convex**, so it has a global minimum. See [[course/ml-eval-metrics]] for log loss as an evaluation metric.
- **Gradient descent** updates each coefficient as `βj := βj − α ∂J/∂βj`, where α is the learning rate. The gradient is `(1/m) Σ (p̂ − y) xj`. This looks like linear regression's gradient, except that p̂ comes from the sigmoid.
- **Math connections**:
  - The sigmoid's derivative is `σ(z)(1 − σ(z))`, which makes gradient descent efficient.
  - The linear part is a dot product, `z = βᵀX`.
  - Matrix operations let you compute over many samples at once.

## Making predictions and the threshold

There are two steps:
1. Compute `p̂ = σ(βᵀX)`.
2. Apply a **decision threshold**, typically 0.5: predict 1 if p̂ ≥ 0.5, otherwise 0.

Raising the threshold makes positive predictions more conservative (fewer false positives). Lowering it makes them more aggressive (fewer false negatives). The ROC curve helps pick the threshold for a given application.

## Python walkthrough (MLBook)

The chapter uses a **synthetic pass/fail dataset**:
- `np.random.seed(42)` and 200 students.
- `HoursStudied` ~ Uniform(0, 10) and `PreviousScore` ~ Uniform(40, 100).
- `z = −8 + 0.7·hours + 0.1·score + N(0,1)` and `Passed = (sigmoid(z) > 0.5)`.

Steps:
1. Plot the data as a scatter colored by pass/fail.
2. Split with `train_test_split(X, y, test_size=0.25, random_state=42)`.
3. Scale with **`StandardScaler`** (fit on train, transform test). Scaling matters here because it helps gradient descent converge faster and puts features on equal footing.
4. Fit `LogisticRegression(random_state=42, max_iter=1000)` and read `intercept_[0]`, `coef_[0][0]`, and `coef_[0][1]`.
5. Predict: `predict_proba()` returns `[P(fail), P(pass)]` columns, and `predict()` returns labels.
6. Evaluate with `confusion_matrix` (laid out TN, FP / FN, TP), then `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `classification_report(target_names=['Failed','Passed'])`. A Plotly annotated heatmap visualizes the confusion matrix.
7. Plot an **ROC curve** with `roc_curve` and `roc_auc_score` on `predict_proba[:, 1]`, against a dashed random-classifier diagonal. AUC = 1.0 is perfect, 0.5 is random, and below 0.5 means the predictions are inverted.
8. Plot the **decision boundary** as a mesh-grid contour of P(pass) over the two scaled features. Logistic regression's boundary is linear.

Metric definitions given in the chapter:
- Accuracy = (TP+TN)/all
- Precision = TP/(TP+FP)
- Recall (sensitivity) = TP/(TP+FN)
- F1 = harmonic mean of precision and recall

The regression-types chapter in [[course/linear-regression]] includes a shorter preview. It builds `Is_Adult = (Age >= 18)`, fits `LogisticRegression()`, and prints `predict_proba`, `predict`, `intercept_`, `coef_`, and a confusion matrix. Its point is that classification metrics (accuracy, precision, recall, F1, ROC-AUC) replace MSE and R².

## Extensions

- **Multiclass**:
  - **One-vs-Rest (OvR)**: one classifier per class.
  - **Multinomial/softmax**: generalizes the sigmoid to several classes.
  - The chapter states that sklearn's `LogisticRegression` uses OvR by default.
- **Regularization**: add an **L2 (ridge)** penalty `λΣβj²` or an **L1 (lasso)** penalty `λΣ|βj|` to the cross-entropy loss. In sklearn, the parameter **`C`** controls strength, and **smaller C means stronger regularization**.
- **Assumptions**:
  - The log-odds are linear in the features.
  - Observations are independent.
  - Features are not highly correlated (no multicollinearity).
  - Samples are large enough.
- **Limitations**:
  - It can't capture complex non-linear relationships without feature engineering.
  - It is sensitive to outliers.
  - Its decision boundaries are linear.
- Strengths named in the chapter: efficiency, interpretability (coefficients indicate feature importance), effectiveness on linearly separable problems, and serving as a building block for more complex models.

## In-class exercise (slides, R-based)

Using the **`mtcars`** dataset, build a logistic model that predicts engine shape **`vs`** (V-shaped = 0, straight = 1) from **`wt`** (weight) and **displacement**. Steps:
1. Fit with `glm()` (binomial family).
2. Interpret the coefficients as odds ratios with `exp(coef(ModelName))`.
3. Assess fit with residual variance and the **Hosmer–Lemeshow test**, using `hoslem.test()` from the `ResourceSelection` package.
4. Build a second model and compare **AIC** (Akaike information criterion; lower is better).
5. Compute a **pseudo-R²** with `pR2()` from the `pscl` package.
6. Answer: "Is this a good model?"

The "Example in Code" slide has no extractable text, and the materials include no Python version of this exercise or R→Python conversion notes. The Python counterparts are the sklearn calls in the walkthrough above.

## Related topics

- [[course/linear-regression]] — the linear predictor, OLS, and the GLM motivation (OLS can predict outside [0, 1] for binary targets).
- [[course/ml-eval-metrics]] — the confusion matrix, precision, recall, F1, ROC/AUC, and log loss in depth.
- [[course/knn]] and [[course/decision-trees]] — other classifiers that predict class probabilities and use the same threshold and evaluation ideas.
- [[course/data-prep]] — scaling (`StandardScaler`) and train/test splitting used in the walkthrough.
- [[course/ensemble-methods]] — logistic regression appears as one of the voting-ensemble members.

## Sources

- `raw/course/DS-3021/logistic-regression/2026-03-19-logistic-regression-slides.pptx` — lecture slides (Generalized Linear Models: Logistic Regression)
- `raw/course/MLBook/logistic-regression/2026-05-21-logistic-regression-overview.ipynb` — MLBook chapter: logistic regression overview with sklearn walkthrough
