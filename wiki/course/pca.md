# Principal Component Analysis (PCA)

A dimensionality-reduction technique: transform high-dimensional data into a lower-dimensional form while retaining as much **variance** (which the course treats as "information") as possible. This topic comes from the course's MLBook, the online-textbook version of the material, as two paired pages: a math-first chapter (`pca.ipynb`) and a hands-on examples notebook (`pca-examples.ipynb`) that implements the same algorithm twice on the Iris dataset, once from scratch in NumPy and once with scikit-learn. Like [[course/kmeans-clustering]], PCA is **unsupervised**: it never looks at the target variable.

## Why reduce dimensions at all

The chapter frames PCA as a response to the **curse of dimensionality**. As the number of features grows, computational cost grows (the chapter says "exponentially"), visualization becomes impossible beyond 3 dimensions, model complexity rises and leads to overfitting, and correlated features add redundancy. PCA addresses all four by replacing the original features with a new set of **uncorrelated** variables (the principal components) ordered by how much of the data's variance each one captures.

The chapter's "use PCA when" list:
- **Reducing computational cost** when there are hundreds or thousands of features
- **Visualization**, by reducing to 2–3 dimensions for plotting
- **Removing multicollinearity** when features are highly correlated
- **Noise reduction**, since minor components often represent noise
- **Feature extraction**, creating new features that capture the most important patterns

Applications listed: image processing (facial recognition, compression), genomics (gene expression), finance (portfolio risk, stock-price patterns), NLP (topic modeling, document similarity), and computer vision (object detection, image reconstruction).

## Math foundations: covariance, eigenvectors, eigenvalues

The chapter builds PCA on two prerequisites.

**Variance and covariance.** Variance is a single variable's spread around its mean, Var(X) = 1/(n−1) Σ(xᵢ − x̄)². Covariance measures how two variables move together, Cov(X,Y) = 1/(n−1) Σ(xᵢ − x̄)(yᵢ − ȳ). For *p* features, the **covariance matrix Σ** is a symmetric *p × p* matrix with variances on the diagonal and pairwise covariances off it.

**Eigenvectors and eigenvalues.** For a square matrix *A*, **v** is an eigenvector with eigenvalue λ if *A***v** = λ**v**: applying *A* only scales **v** and does not change its direction. Eigenvalues are found by solving the characteristic equation det(*A* − λ*I*) = 0. In PCA:
- The **eigenvectors** of the covariance matrix are the principal components, the directions of maximum variance.
- The **eigenvalues** are the amount of variance each component explains. Larger eigenvalue means a more important component.

## The algorithm: five steps

1. **Standardize.** Center each feature (X − μ) and usually also scale it to unit variance (z-score: (X − μ)/σ). *Why:* features on larger scales would otherwise dominate the components. Standardizing makes each feature contribute equally. This is the same scale-sensitivity argument the course makes for [[course/knn]] and k-means, for the same distance/variance reason.
2. **Covariance matrix.** Σ = (1/(n−1)) XᵀX on the centered/standardized *n × p* data matrix.
3. **Eigen-decomposition.** Solve Σ**v**ᵢ = λᵢ**v**ᵢ, which gives *p* eigenvalue/eigenvector pairs.
4. **Sort and select.** Order the eigenvalues λ₁ ≥ λ₂ ≥ … ≥ λₚ and keep the eigenvectors for the top *k*.
5. **Project.** Z = X Wₖ, where Wₖ is the *p × k* matrix of chosen eigenvectors and Z is the new *n × k* data.

### Variance explained and how many components to keep

The proportion of variance explained by component *i* is λᵢ / Σⱼ λⱼ. The **cumulative** variance explained by the first *k* components is the sum of the first *k* eigenvalues over the sum of all of them. The course's **rule of thumb** is to keep enough components to explain **at least 80–95%** of total variance. The examples notebook operationalizes this with a scree-style bar chart of per-component variance next to a cumulative-variance line with a dashed 95% threshold, then finds the smallest *k* that crosses each of 80/90/95/99% with `np.argmax(cumvar >= threshold) + 1`.

## Two ways to see what PCA is doing

**Geometric view.** The principal components are new, mutually **orthogonal** axes. PCA *rotates* the data so the most variance lies along PC1, the next most along PC2, and so on. The transformed data are the projections of the original points onto those axes.

**Optimization view (connection to calculus).** Find the unit vector **w** that maximizes the variance of the projected data: maximize **w**ᵀΣ**w** subject to ‖**w**‖ = 1. The Lagrangian is L = **w**ᵀΣ**w** − λ(**w**ᵀ**w** − 1). Setting ∂L/∂**w** = 2Σ**w** − 2λ**w** = 0 gives Σ**w** = λ**w**, which is exactly the eigenvalue equation. The variance-maximizing direction is therefore the eigenvector with the largest eigenvalue. Later components maximize variance in directions orthogonal to the earlier ones. The point the chapter is making: the eigen-decomposition recipe is the *solution* to a "maximize variance" problem, not an arbitrary procedure.

## Worked example: Iris, from scratch and with scikit-learn

The examples notebook uses `sklearn.datasets.load_iris()`: 4 features (sepal length/width, petal length/width) and a 3-class species target. The target is used **only to color plots**, never to fit PCA.

**From scratch (NumPy):**
- Standardize manually with `(X - X.mean(axis=0)) / X.std(axis=0)`, then build the covariance matrix as `X_standardized.T @ X_standardized / (n_samples - 1)` and visualize it as a heatmap (`px.imshow`).
- `np.linalg.eig(covariance_matrix)` returns the eigenvalues and eigenvectors (the eigenvectors are the *columns*). The notebook checks Σv = λv numerically with `np.allclose`.
- Sort with `np.argsort(eigenvalues)[::-1]`, compute variance explained and cumulative variance, and project onto the first two eigenvectors (`X_standardized @ W`, with W of shape 4×2) for a PC1-vs-PC2 scatter colored by species.
- **Loadings interpretation.** The notebook reads the eigenvector coefficients (`W.T` as a feature × PC table/heatmap). **PC1**: all four features load positively, especially petal length and width, so it captures *overall flower size*. **PC2**: sepal length and sepal width load in opposite directions, so it captures *shape* variation.

**With scikit-learn:**
- `StandardScaler().fit_transform(X)`, then `PCA(n_components=2).fit_transform(X_scaled)`. Read `explained_variance_ratio_` (variance per component) and `components_` (the eigenvectors, one per *row*, unlike `np.linalg.eig`'s columns).
- **Verification.** The manual and sklearn variance ratios match (`np.allclose`). The components match **up to sign**: an eigenvector and its negation describe the same direction, so the check accepts either `manual == sklearn` or `manual == -sklearn`. Expect sign flips between implementations. They are not errors.
- **Automatic component selection.** `PCA(n_components=0.95)` (a float rather than an integer) keeps however many components are needed to retain 95% of variance. `pca.n_components_` reports how many that was.
- **Inverse transform.** `pca.inverse_transform()` maps the 2-D projection back into the original 4-D (scaled) space. The mean squared difference from the original is the **reconstruction error**, i.e. the information lost by dropping components. The notebook compares original vs. reconstructed values for the first sample.
- **3-D view.** `PCA(n_components=3)` with `px.scatter_3d`, with each axis labeled by its percent of variance explained.

The notebook's stated takeaway is that on Iris the **first two components capture ~95%** of the variance, which is why a 2-D plot of a 4-feature dataset loses very little.

## Caveats (as taught)

- PCA assumes **linear** relationships between features.
- Components are **linear combinations** of the original features, so they can be harder to interpret than the raw variables. Reading the loadings, as in the Iris example, is how you recover meaning.
- PCA is **sensitive to scaling**: "always standardize your data first."
- PCA is **unsupervised**: it "doesn't consider the target variable." In the Iris example, species is only used afterward to color the plots and see whether the components separate the classes.

## Python tooling summary

| Task | Code used in the course |
|---|---|
| Standardize | `StandardScaler().fit_transform(X)` (or manual z-score) |
| Covariance (manual) | `X.T @ X / (n - 1)` |
| Eigen-decomposition (manual) | `np.linalg.eig()`, then sort with `np.argsort(...)[::-1]` |
| Fit/transform | `sklearn.decomposition.PCA(n_components=k).fit_transform()` |
| Variance per component | `.explained_variance_ratio_`; cumulative via `np.cumsum()` |
| Principal axes | `.components_` (rows = components) |
| Keep X% variance | `PCA(n_components=0.95)`, then read `.n_components_` |
| Reconstruct | `.inverse_transform()` |
| Plots | `plotly.express` (`scatter`, `scatter_3d`, `imshow`), `matplotlib` for variance bars |

Both pages run in the browser through JupyterLite (the rocket icon, then "Live Code"). The first execution may take a moment to load the kernel and packages.

## Historical note and references

Both pages credit PCA to Karl Pearson (1901), later refined/extended by Harold Hotelling (1933). For deeper treatment they point to Jolliffe's textbook on PCA, the Jolliffe & Cadima review article, and the scikit-learn `PCA` documentation.

## Related topics

- [[course/kmeans-clustering]]: the other unsupervised method in the course. Both work on features alone with no target, and both are scale-sensitive.
- [[course/knn]]: KNN's slides flag the curse of dimensionality ("often works best with 25 or fewer dimensions"), which is the problem PCA is designed to reduce. The standardize-first logic is the same.
- [[course/data-prep]]: standardization/scaling as a preprocessing step, which PCA requires.
- [[course/linear-regression]]: multicollinearity among predictors is one of the problems the chapter lists PCA as addressing.

## Sources

- `raw/course/MLBook/pca/2026-05-21-pca.ipynb`: MLBook PCA chapter (motivation, variance/covariance, eigen-decomposition, 5-step algorithm, variance explained, geometric and Lagrange-multiplier views, when to use, applications)
- `raw/course/MLBook/pca/2026-05-21-pca-examples.ipynb`: MLBook PCA examples notebook (Iris; from-scratch NumPy implementation vs. scikit-learn, loadings, automatic component selection, inverse transform, 3-D plot)
