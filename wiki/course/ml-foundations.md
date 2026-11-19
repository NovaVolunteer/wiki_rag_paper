# Machine Learning Foundations

Week 4 of DS-3021 ("04: ML Concepts and Foundations". The in-class notebook's Colab link points to the repo folder `04_ML_Concepts_I_Foundations`). This is the conceptual on-ramp to the modeling half of the course. It covers what machine learning is (and isn't), supervised vs. unsupervised vs. reinforcement learning, when a problem is "Suitable for Machine Learning," and Brian Wright's four-phase **Data Science Lifecycle**, which later weeks keep referring back to. The week also includes a first hands-on model (k-NN and a decision tree on iris) and two Phase I case studies (Walker Advertising and a K-12 math-software recommender). It comes after the coding refresher in [[course/python-basics]] and before [[course/data-prep]].

## What is machine learning?

### Definitions

The slides, the overview reading, and the "What is Machine Learning" notebook all give the same three definitions:

- **Arthur Samuel** (coined the term in 1959 at IBM): "A field of Computer Science that gives computers the ability to learn without being explicitly programmed."
- **Goodfellow et al., *Deep Learning***: "The ability [for systems] to acquire their own knowledge, by extracting patterns from raw data."
- **Tom Mitchell** (Carnegie Mellon): "A computer program is said to learn from experience E with respect to some set of tasks T and performance measure P if its performance [at] tasks in T, as measured by P, improves with experience E."

The in-class slide deck opens with live polls ("What is Machine Learning?", "What is not Machine Learning?", "What is the goal of ML?") and a slide titled "'Learning' and a 'Model'". The poll results aren't in the deck.

### AI vs. ML vs. deep learning

The notebook uses a nested-categories graphic: **machine learning is a subcategory of AI, and deep learning is a type of machine learning** that focuses on neural networks. The instructor's framing is that when people say "AI" in current popular discussion, what they really mean is deep learning.

### ML as a general-purpose technology

ML is described as a **general-purpose technology (GPT)**: "a new method of producing and inventing that is important enough to have a protracted aggregate impact." Like electricity or the internet, it applies across domains, "everything from Marketing to Agriculture to Sports." The slides give examples to discuss:
- **ImageNet** error rates fell from 30% in 2010 to under 3% "today." The slide asks why getting below 5% matters.
- **Deep Blue vs. Kasparov**: Deep Blue searched about 200 million chess positions per second, while Kasparov searched maybe 5 to 10, "yet he played almost at the same level… why?"

Two caveats follow: "we remain remarkably far away" from human-like general intelligence, and machines can't do the full range of tasks humans can.

### Machine vs. human

A comparison table splits the work. **Machines** are better at performing calculations quickly, performing repetitive tasks, and following pre-defined rules. **Humans** are better at understanding context, thinking through the problem, asking the right questions, selecting the right tools, and interpreting results.

### How machines learn

The process has three parts (citing *Introduction to ML* by Chandramouli, Dutt and Das):
1. **Data Input**: past data is used as the basis for future decisions.
2. **Abstraction**: the underlying algorithm represents the input data in a broader way.
3. **Generalization**: the abstracted representation is generalized into a framework for making decisions.

## Types of machine learning

- **Supervised learning**: "pattern discovery when inputs (x) and outputs (y) are known." There is a **target variable**. The slide example is input = voter, output = political affiliation. **Classification and regression are both supervised.** The notebook's framing: "we understand the end point of the problem but have not yet discovered the road to get there." The speaker notes use a wine example: you define the categories (bad, good, great) ahead of time and the algorithm sorts wines into them.
- **Unsupervised learning**: the inputs have no target outputs. The output is "not given (to be discovered)," and the goal is to "impose structure on the inputs (x) to say something meaningful about the data." In the notebook's framing, "we do not know the end point or the road." In the wine example, you group wines by region, year, price, or type to see which ones cluster together. The standard e-commerce use is **customer segmentation**.
- **Reinforcement learning**: described as a combination of the other two. It doesn't need labelled input-output pairs. It uses the knowledge available to adapt to its environment. The driving analogy: instead of working out the best route ahead of time (supervised), "we simply start driving to learn the fastest way." Another example is animals adapting to their environment to maximize food and reduce danger.

The course "will mostly focus on supervised methods with some unsupervised."

### The social media exercise

The notebook uses a social media platform that wants to keep users on the site as long as possible (more ads, more revenue). Students identify three things:
- **Target variable**: the variable we want to understand better. Answer key: amount of time users spend on the site.
- **Features**: the variables used to understand the target. Answer key: clicks, customer demographics, time of day, web browser, browsing history, type of content, etc.
- **Independent business metric**: how the organization tracks whether the algorithm delivers value. Answer key: revenue generated after the model is deployed.

The notebook notes there is likely more than one right answer. For the unsupervised version of the same problem, the answer key is to segment the site's customers into groups that aren't known in advance.

### k-means worked example (slides)

Both slide decks include a hand-worked k-means example under the heading "Example of K-Means, What is Happening?" It shows what unsupervised learning looks like. Given D = {2, 4, 10, 12, 3, 11, 20, 25, 30} and k = 2, with means randomly set to m1 = 3 and m2 = 4:

| Iteration | K1 | K2 | m1 | m2 |
|---|---|---|---|---|
| 1 | {2, 3} | {4, 10, 12, 11, 20, 25, 30} | 2.5 | 16 |
| 2 | {2, 3, 4} | {10, 12, 11, 20, 25, 30} | 3 | 18 |
| 3 | {2, 3, 4, 10} | {12, 11, 20, 25, 30} | 4.75 | 19.6 |
| 4 | {2, 3, 4, 10, 11, 12} | {20, 25, 30} | 7 | 25 |
| 5 | same | same | 7 | 25 |

The algorithm stops when the clusters and means stop changing. See [[course/kmeans-clustering]] for the full treatment.

## Is the problem "Suitable for Machine Learning" (SML)?

Successful ML "requires very detailed specifications on what is to be learned and data to support that learning activity." It also takes trial-and-error feature engineering and, "most importantly," embedding the product into normal business operations so the efficiencies are actually realized. The slides list what makes a task suitable for ML. They note that most recent successes are supervised and that ML's competency is narrow compared with human decision-making:

1. **Learning a function that maps well-defined inputs to well-defined outputs.** Predicting Y from X still might not capture the causal effect.
2. **Large data** with input-output pairs is available or can be created. More training data gives a more accurate model.
3. **Clear feedback with well-definable goals and metrics** ("optimize flight patterns, not a single flight").
4. **Reasoning and diverse background knowledge aren't needed.** ML is good at empirical associations but "terrible at decision making that requires common sense of historical knowledge."
5. **No need for the reason behind the decision to be clear.** A neural network may use millions of numerical weights.
6. **Tolerance for error or sub-optimal solutions.** ML outputs are probabilistic, so some error is always assumed.
7. **What's being learned shouldn't change rapidly over time.** ML works best when future data is distributed roughly like the training data. If it isn't, systems need to be in place to refresh the algorithms.

The "What is ML" notebook condenses this into three **ideal conditions**:
- **Large datasets**: "small datasets with a few hundred rows just do not work well."
- **Inputs need to map well to outputs**: called "possibly the **most important** lesson." The lifecycle "is less about building algorithms and more about proving the data you have can address the problem at hand."
- **Causality is not required**: unlike classical statistics, ML only generates predictive probabilities. We can say features are **predictive** of the target, not that they **cause** it, so ML shouldn't be used when you need to know *why*. The notebook links a Statistical Horizons article on prediction vs. causation, which uses "dependent" variable to mean the same thing as target.

## The Data Science Lifecycle ("Brian's Version")

The course uses one lifecycle diagram all semester. It has **four phases** separated by **gate reviews (G1, G2, G3)** and feeds into "Reports – Dashboards – Products." An opening slide also contrasts "Engineering of Machine Learning Algos versus Software Development."

### Phase 1: Idea Development

The notebook calls this "likely the most important part of the entire lifecycle." Its steps are:
1. **Question ID**: almost certainly a piece of a larger problem, so be as specific as possible.
2. **Business Understanding**: may take extra team members and many conversations to align with the organization's goals.
3. **Evaluation Criteria**: how the algorithm will be measured, and the benchmark it must meet.
4. **Data Acquisition | ETL**: find data that matches the problem. Sources include online databases, internal company data, APIs, web scraping, and real-time sensors. The notebook points to Jon Kropko's *Surfing the Data Pipeline* book.

Phase 1 usually ends with an **analytical plan** presented to stakeholders. It has four parts: (1) a detailed problem explanation, including scope (what *not* to do) and the target variable (supervised) or goal (unsupervised); (2) data knowledge: amount, provenance, biases (too old, missing, duplicates), prior use, and rate of change; (3) the algorithmic approach and evaluation measures, with performance benchmarks if possible; (4) an **independent business metric**. The tip: "The more energy that is spent on this step the better the project."

### Phase 2: Data Preparation and Problem Exploration

1. **Data Understanding + EDA**: dig into the data's potential and pitfalls and fix issues. This "usually takes the most time out of all the steps."
2. **Initial Model Building**: a simple model, possibly on a sample of the data, to see how well the data matches the problem.
3. **Initial Model Evaluation**: separated from step 2 because this is where you apply the metric chosen in Phase 1.

From Phase 2 onward, many decisions affect the results. Best practice is to **heavily document your choices** so you can revisit or justify them, which reflects an **experimental mindset**.

### Phase 3: Solution Development

**Model training and testing** means either keep tuning the initial model or deliberately choose more algorithms based on the data, "not a 'try everything and see what sticks' approach." It includes:
- **Feature engineering**: creating or modifying features. It is experimental, can make the model *worse*, and needs domain knowledge and careful documentation. The slides add, "Likely going to dedicate a week to just this topic."
- **Algorithm (hyperparameter) tuning**: hyperparameters are set *before* training (learning rate, regularization strength, number of trees in a random forest, number of layers in a neural net). Model parameters are learned from the data. Search methods include grid search, random search, and Bayesian optimization, and the best values are chosen on a validation set.
- **Evaluation**: assess performance on data not used in training (accuracy, precision, recall, F1, AUC) to detect **overfitting** (good on training data, poor on unseen data) and **underfitting**. Decide *which* measure to use and *why* **before** training. Benchmarks translate into how much risk (error) you're willing to accept.

### Phase 4: Solution Deployment (ML Engineering)

Phase 4 "can be one of the hardest parts." Software engineers and data scientists work hand in hand. It covers **algorithm deployment** (possibly through A/B testing), **data drift** and **model drift analysis** ("no ML solution lasts forever"), and **business metric tracking**. Models should be evaluated continuously so you know when to retrain.

> ⚠️ Possible conflict: the lifecycle notebook's intro names the four phases "Idea Development, Data Preparation and Problem Exploration, Solution Development and **Model Development**." Its own section headings, and every slide version, call Phase 4 **Solution Deployment**. Also, the notebook lists "Initial Model Building" twice in Phase 2; the slides' step is "Initial Model Evaluation." The lifecycle box labels also differ between versions. The older reading has "Data Acquisition – ETL" and "Optimization – Hyperpara and Evaluation." A later slide in the same reading, and the bootcamp deck's second diagram, use "Data Acquisition/Representation" and "Final Model Development." Another bootcamp summary slide uses Problem Development, Data Gathering | ETL, Analytical Plan, … Final Model Optimization & Evaluation, … Business Metric Tracking.

## Key ML terms, organized by phase

The slides list "SOME Key ML Methods/Terms" by phase:
- **Phase 1**: prediction vs. inference, independent metric for business value, target variable and features, classification vs. regression, probabilistic interpretation, and data acquisition/gathering.
- **Phase 2**: variable classes/types, scaling/normalizing and one-hot encoding, missing data, **baseline (prevalence)**, data partitioning/sampling, and EDA (summary stats and visuals).
- **Phase 3**: parameters vs. hyperparameters, thresholding, feature engineering, the bias–variance tradeoff, model evaluation, and non-parametric modelling (random state).

The terms the slides explain in detail:
- **Prediction vs. inference**: "Almost all ML methods are focused on prediction not causation or inference." That's why performance is judged by how well a model predicts, not by how much each variable reduces error.
- **Independent metric for business value**: a measure separate from the model that shows whether the model is providing value. Examples: a Netflix recommender (number of user clicks), a spam blocker (number of viruses in the network), market clustering (did sales increase).
- **Target vs. features**: the target is "the variable we are trying to predict." Features are "the variables used for predicting the target," and feature engineering can add more of them.
- **Classification vs. regression**: classification predicts membership in defined categories (binary or multi-level target). Examples: male vs. female from thousands of pictures, a winning season or not, loan default. **Key point: classification outputs are not 0/1 but percentages**, and a multi-category target gets one probability per category for every row. Regression predicts a number from a continuous target. Examples: a game score, amount of rain, a person's weight.
- **Probabilistic interpretation**: the course focuses mostly on classification, "a much more common ML goal." Treat outputs as risk measures or as a quantification of uncertainty. The same 75% probability might be good enough for a marketing email opening but not for whether a drug is effective.
- **Thresholding**: the cutoff where a predicted probability becomes 0 or 1. The **default is 50%**, but you might raise it to 75% or 90% to be extra careful. See [[course/ml-eval-metrics]].
- **Data brainstorming**: *Data to Concept* asks whether the available data supports the target and goal (how hard it is to gather, whether there's enough, its rate of change, whether it's the right source). *Learning Difficulty* asks how complex or vague the target is (class imbalance, whether the data links to the problem, past use, whether the target is hard to measure or break into parts, and what risk level is acceptable).
- The slides for **baseline (prevalence)**, **bias versus variance**, and **evaluation** are title-only or image-based in the extracted decks. Evaluation "gets an entire week" ([[course/ml-eval-metrics]]).

### Booking.com lessons (extra material)

The slides list lessons from Booking.com:
- **Swiss Army Knife**: highly adaptable ML.
- **Offline Health Check**: use randomized controlled trials to test model outputs against business metrics such as customer conversion. Higher model performance doesn't necessarily mean more value.
- **Make a Target Before you Shoot**: for example, define what "date flexibility" means.
- Learning Difficulty, Data to Concept, and **Selection Bias** (does the model perform better for a subset of the target?).
- **Speed Kills**: reduce latency through sparsity, model redundancy, caching, etc.
- **Keep a watchful eye**: use specialized monitoring tools.
- **Experimental design as a best practice**: "Experimentation through Randomized Controlled Trials is ingrained into Booking.com culture."

## ML1 in-class notebook: a first model on iris

The notebook loads iris from the UCI repository (`from ucimlrepo import fetch_ucirepo`, `fetch_ucirepo(id=53)`) and copies the `class` column from `y` into `X`. It asks, "What mental models can we see from these data sets? What data science questions can we ask?"

- **Three-way split**: `train_test_split(X, test_size=0.3, stratify=X['class'])` gives 70% train and 30% test, stratified so each split has the same class distribution. A second `train_test_split(test, test_size=0.5, stratify=test['class'])` splits the 30% into **test and validation** halves (15%/15% of the original). Each split is checked with `.describe()`.
- **k-NN**: `KNeighborsClassifier(n_neighbors=10)` is fit on `train.drop(['class'], axis=1).values` (features) and `train['class'].values` (labels).
- **Decision tree**: `DecisionTreeClassifier(random_state=0)` is fit on the same data. The notebook counts predicted vs. true test labels for Iris-versicolor, Iris-virginica, and Iris-setosa, then builds `sklearn.metrics.confusion_matrix(...)` with those labels. It plots a confusion matrix with `ConfusionMatrixDisplay.from_estimator(neigh, X_test, y_test, cmap=plt.cm.Blues)`, replacing the commented-out older `plot_confusion_matrix` call. That plot is for the k-NN model. It draws the tree with `plot_tree(clf, filled=True, feature_names=...)`.
- **Patterns in data**: `X.groupby(['class'])` followed by `['sepal length'].describe()` (also sepal width and petal length) shows how the classes differ. A `plotly.express` 3-D scatter plots sepal length, petal length, and petal width colored by class. The `px` import is commented out at the top, so this cell needs it restored.
- **"Mild disclaimer"**: "Do not worry about understanding the machine learning in this example!" k-NN is covered later ([[course/knn]], [[course/decision-trees]]). The discussion questions are: What is the purpose of data splitting? What can we learn from testing/validation? How do we know if a model is working? How could we find the model error? Students are encouraged to change the test size or `n_neighbors` and see what changes.

## R → Python differences (week 4 conversion notes)

| R | Python | Note |
|---|---|---|
| `rbind()` | `sklearn.model_selection.train_test_split()` | Splits data into train/test. The Python function has simpler parameters and makes it easy to control the train/test proportion |
| `knn()` | `sklearn.neighbors.KNeighborsClassifier()` | Very similar. The difference is what you call on the fitted model: R checked the model summary against the target, while Python calls `model.score(independent, target)` |
| `tapply()` | `df.groupby()` | Same effect, different approach. R groups and applies `summary` to a column in one call. Python groups first and calls functions in separate lines (summary stats by species) |

General difference: R's functions here are built in, while Python uses **sklearn**.

## Labs / case studies

- **Foundations lab: "DS Lifecycle Phase I Case Study"** (`foundations-lab-overview.md`). The key points to cover are developing DS questions, metrics tied to the question, target variables, model type, data brainstorming, and data to concept. Groups read the case and answer: What problems does the organization face? What is the target variable, and is the solution regression or classification? How would they know it's working (business metric)? What other issues might the team face? Then they work through: How hard is the data to gather? Is the target hard to measure or break into parts? What level of uncertainty (risk) is acceptable? Groups get **20 to 30 minutes**, then discuss, and submit a **written summary** (assign a note taker).
  - **The case**: a non-profit building an **MVP recommender engine** to guide **K-12 school districts' investments in Mathematics software**, because billions of dollars are wasted every year on software that doesn't fit schools' needs. The data is a survey of Midwest districts (urban, suburban, and rural) with **60 Likert-scale (1–5) questions** on demographics, technology infrastructure and proficiency, teacher preparedness, administrative support, and how past software implementations went. There are **1,500 responses from 30 districts, with no district contributing more than 100**. Challenges include a small dataset, response bias, and inconsistent interpretation of questions. The planned strategy is EDA, then feature engineering (clustering districts, composite indices such as "readiness for tech adoption"), then collaborative filtering or hybrid recommenders, with human experts validating at first. Success has to go beyond recommendation accuracy to include student performance, teacher satisfaction, and long-term cost savings. Scaling nationally raises questions about regional differences and over-represented district types in the training data.
- **Walker Advertising case** (Darden case UVA-QA-0912, Oct. 11, 2019, field-based). It is a second Phase I case about a firm that isn't data-driven moving into analytics and ML.
  - **Background**: Mary Ann Walker founded **Los Defensores** in 1984 in LA, when she was a 22-year-old court translator, to connect Spanish-speaking consumers with bilingual attorneys. It started with 5 subscribing attorneys practicing auto-accident and workers' compensation law. She added **1-800-THE-LAW2 (TL2)** for English speakers in 1985. The State Bar of California challenged attorney pooling for advertising, and the California Supreme Court sided with Walker on free-speech grounds. Private-equity firm **ClearLight Partners** invested in 2015. **Quentin Kluthe** joined as CFO in 2017.
  - **The business problem**: grow beyond California and shift from **subscription revenue to pay-per-lead**. That requires forecasting how much advertising spend drives how many callers (leads 1–4 weeks out). Costs (mostly media) lined up only partly with sales.
  - **Key facts**: 85% of revenue-generating responses are phone calls, 80% from Los Defensores and 20% from TL2. Digital accounts for 15% of leads (website 10%, chat/text 5%). There are 10 unique phone numbers for traditional advertising and about 25 for digital. The contact center is open 24/7, and the target answer time is 8–10 seconds (the call-center norm is 30). Roughly 2% of leads lead to seven-figure outcomes. Leads are distributed fairly by a rotating "wheel." 60–70% of new contracts drop within 12 months, and a three-month sale is recommended. Law firms spend 1.7% of income on advertising. KPIs meant different things to different groups (static SSRS reports, power pivots). In 2018 the company built a data warehouse with **Power BI** reporting on top, and data scientist Nitesh Prakash started a lead-forecast model. Kluthe's closing question about a 90,000-variable model: "The prediction itself might be more accurate, but would you ever understand it?"
  - **Exhibit 1 data dictionary**: a **Calls** table (CallId, StartTime, City/State, Area, CompanyShortName [LD Spanish, TL2 English, LRD Spanish web], LawType [MVA, L&E, Premise, WC], MediaSource, MediaGroup [traditional/digital], Weekof) and a **Media Placements** table (StartTime, PostLogDate, MediaType [Radio/TV/Cable], ISCI tape code, Duration, CompanyShortName, LawTypeDescription/LawSubTypeDescription, MediaSource/Station Area, MediaMonth/Quarter/Year, WeekOf).

## Related topics

- [[course/python-basics]]: the pandas tools (`groupby`, `describe`, `drop`) used in the ML1 notebook
- [[course/data-prep]]: Phase 2 of the lifecycle: variable types, scaling and one-hot encoding, missing data, prevalence baselines, partitioning
- [[course/knn]] and [[course/decision-trees]]: the two models previewed on iris here
- [[course/kmeans-clustering]]: the unsupervised method behind the slides' worked k-means example
- [[course/ml-eval-metrics]]: thresholding and evaluation, which the slides defer to "an entire week"
- [[course/ml-bias]]: selection bias and biased training data, raised in the Booking.com lessons and the K-12 case
- [[course/final-project]]: applies the full lifecycle, from analytical plan to evaluation
- [[course/course-overview]]: where this week sits in the semester

## Sources

- `raw/course/DS-3021/ml-foundations/2024-10-29-ml-overview-reading.pdf`: "Machine Learning Overview" reading (PDF of the slide deck)
- `raw/course/DS-3021/ml-foundations/2024-10-29-walker-advertising-case-study.pdf`: Walker Advertising Darden case (UVA-QA-0912)
- `raw/course/DS-3021/ml-foundations/2024-10-29-week04-r-to-python-changes.docx`: week 4 R→Python conversion notes
- `raw/course/DS-3021/ml-foundations/2025-02-13-ml1-in-class-notebook.ipynb`: ML1 in-class notebook (iris, k-NN, decision tree)
- `raw/course/DS-3021/ml-foundations/2026-03-10-ml-bootcamp-slides.pptx`: ML overview / bootcamp lecture slides
- `raw/course/DS-3021/ml-foundations/2026-03-10-ml-in-class-slides.pptx`: in-class poll slides plus k-means example
- `raw/course/MLBook/ml-foundations/2026-05-21-ds-lifecycle.ipynb`: Data Science Lifecycle and Experimental Thinking chapter
- `raw/course/MLBook/ml-foundations/2026-05-21-foundations-lab-overview.md`: DS Lifecycle Phase I case study lab (K-12 recommender)
- `raw/course/MLBook/ml-foundations/2026-05-21-what-is-ml.ipynb`: "What is Machine Learning" chapter
