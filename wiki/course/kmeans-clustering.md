# K-Means Clustering

Week 10 of DS-3021. The course's first unsupervised topic: instead of predicting a labeled target, k-means partitions unlabeled data into `k` groups of similar points based on distance alone. There's no y to check yourself against while training — validation has to happen after the fact, against something the algorithm never saw.

## Unsupervised vs. supervised — what's actually different

Every prior topic (KNN, decision trees) starts from a labeled target: input `x`, known output `y`. Clustering flips that — the slides frame it literally as "Input x: Voter, Output y: Not given (to be discovered)." The goal is to impose structure on `x` alone and see whether that structure means anything, rather than to predict a value you already have examples of. That's why the course pairs clustering with exploratory data analysis (EDA) more broadly: it's presented as one of three ways to find structure in data that's too big or complex to just eyeball (the others being network analysis and text mining), useful specifically when you don't have — or don't want to assume — a target variable yet.

## The algorithm: four steps, repeated to convergence

k-means clustering: 1) randomly choose `k` data points as initial centroids, 2) assign every point to its closest centroid (by a distance metric — usually Euclidean: distance = √((x₂−x₁)² + (y₂−y₁)²)), 3) recalculate each centroid as the mean location of its current members, 4) repeat steps 2–3 until centroid assignments stop changing. A centroid is not generally an actual data point — it's just the average location of whatever points currently belong to it.

The slides work this by hand on a trivial 1-D example that's worth tracing because it makes "convergence" concrete: given `D = {2,4,10,12,3,11,20,25,30}` and `k=2`, starting from arbitrary means m1=3, m2=4, reassigning points and recomputing means each round settles after four iterations at `K1={2,3,4,10,11,12}` (m1=7) and `K2={20,25,30}` (m2=25) — and running the same reassignment again produces the identical split, which is the stopping condition.

The standalone worked-example slide ("Example of K-Means, What is Happening?") lists every iteration. Each row assigns points to the nearer of the current means, then recomputes each mean from its new members:

| Iteration | Means used | K1 | K2 | New m1 | New m2 |
|---|---|---|---|---|---|
| 1 | m1=3, m2=4 (random start) | {2,3} | {4,10,12,11,20,25,30} | 2.5 | 16 |
| 2 | 2.5, 16 | {2,3,4} | {10,12,11,20,25,30} | 3 | 18 |
| 3 | 3, 18 | {2,3,4,10} | {12,11,20,25,30} | 4.75 | 19.6 |
| 4 | 4.75, 19.6 | {2,3,4,10,11,12} | {20,25,30} | 7 | 25 |
| 5 | 7, 25 | {2,3,4,10,11,12} | {20,25,30} | 7 | 25 |

"Stop, since the clusters and the means found in all subsequent iterations will be the same." The trace shows how a poor random start (two means right next to each other at the low end) gets corrected. K1 grows by one or two points per round as m2 is pulled up toward the high values, until the assignment stops changing. The same slide also appears in the main deck, right after the "write out the steps in kMeans Clustering in your own words" prompt.

Because the *starting* centroids are random, k-means is non-deterministic by default: different runs can converge to different local optima. The course's code fixes this with `set.seed()` in R and `random_state=` in Python's `sklearn.cluster.KMeans` so results are reproducible for grading and comparison, not because the algorithm itself is deterministic.

## In-class example: does Congress vote in patterns?

The running example all semester for this topic is a dataset of 427 members of Congress who served a full year in 2013, each with three vote-count columns for bills introduced by Democrats: `aye`, `nay`, `other`. Clustering runs on those three numeric columns only — no party label is given to the algorithm. The workflow:

1. **Cluster.** Run k-means on `aye`/`nay`/`other` (`k=2` in the original walkthrough).
2. **Visualize the raw result.** Plot the points (aye vs. nay) labeled only by cluster number, with no reference to real-world party — this is the "before" picture, showing whatever structure k-means found on its own.
3. **Validate against ground truth withheld from training.** Re-color the *same* points by each member's actual party affiliation and compare. If the algorithm's clusters line up with real party lines, that's evidence the clustering captured something real rather than noise.

The two curated figures show exactly this before/after: the raw cluster plot has no party information in it at all, while the party-colored version shows Republican members (red) forming one tight, dense group and Democratic members (blue) forming a second group that's visibly more diffuse — occasional outliers on both sides, but a clean two-cluster split that tracks the real party divide almost perfectly. That's the interpretive payoff the slides call out explicitly: "Republicans more dense," "diffuse among Democrats," "can gauge outliers." The takeaway generalizes beyond Congress — unsupervised clustering is validated *after the fact* by checking whether the groups it finds correlate with a label you deliberately withheld, not by any accuracy score computed during training (there isn't one).

The current in-class code (`cluster_example.py`, Python/`scikit-learn`, not R) reruns this same house-votes example — fits `KMeans(n_clusters=3, ...)`, inspects `.inertia_`, and renders an interactive 3D scatter (aye/nay/other) with `plotly`. It goes one step further than the slides: it feeds the resulting cluster label back in as an extra feature and trains a `DecisionTreeClassifier` to predict party twice — once with the cluster label included, once without — comparing confusion matrices to see whether the unsupervised cluster assignment actually helps a downstream supervised model. This is a good concrete illustration that clustering doesn't have to be an end in itself; it can be feature engineering for a later supervised step.

## Choosing k: elbow method, NbClust, and silhouette score

Unlike KNN's `k` (see below), k-means' `k` isn't tuned by cross-validated accuracy — there's no accuracy to tune against. The course covers three ways to pick it:

- **Elbow method.** Run k-means across a range of `k` (the course loops 1–10), compute a fit measure at each `k` (variance explained, or equivalently within-cluster sum of squares / "WCSS"), and plot it against `k`. Look for the "kink" where adding another cluster stops buying much improvement. Explicitly flagged as *not guaranteed to work* — it depends on the data having an actual elbow to find. On the house-votes data, the kink appears around `k=2`, which matches the real two-party structure.
- **NbClust (R only).** A package that runs 30 different cluster-validity indices and takes a majority vote on the best `k`. On the house-votes data, 14 of the ~24 indices that returned an answer voted for `k=2` — by far the largest bloc, again lining up with the real party split. There's no direct `NbClust` equivalent in the course's Python tooling; sklearn's `silhouette_score` is used instead.
- **Silhouette score.** For each point, compares average within-cluster distance (cohesion) against average distance to the nearest *other* cluster (separation); ranges from −1 to 1, higher is better, and a negative score means a point is probably in the wrong cluster. `cluster_example.py` loops `k=2` through `10`, computes the average silhouette score at each `k`, and picks whichever `k` maximizes it.

## Assessing cluster quality: variance explained

Separate from *choosing* `k` is asking *how good* a given clustering is — the slides frame this as maximizing inter-cluster distance while minimizing intra-cluster distance, formalized as **variance explained = between-cluster sum of squares / total sum of squares**. R's `kmeans()` returns this directly as `betweenss` and `totss` attributes (`var_exp = kmeans_obj$betweenss / kmeans_obj$totss`); at `k=2` on the house-votes data this comes out to ≈0.719, and re-running at `k=3` gives ≈0.795 — more clusters mechanically explain more variance, which is exactly why elbow plots look for a bend rather than just picking whatever `k` maximizes this number.

Python's `sklearn.KMeans` doesn't expose `betweenss`/`totss` directly — only `.inertia_` (equivalent to within-cluster SS). `cluster_example.py` reconstructs the same ratio manually: compute total sum of squares from each point's squared distance to the *global* mean, subtract `.inertia_` to get the between-cluster portion, then divide by the total.

The **`Manual Variance Explained.xlsx`** worked example builds the same formula by hand on a toy 1-D dataset (points 1–20, split into two clusters of 10) so the arithmetic is transparent before trusting a library to do it: with cluster centroids at 5 and 15 and a global center at 10.5, the total sum of squares is 665, the summed within-cluster sum of squares is 170, giving a between-cluster sum of squares of 495 and **variance explained = 495/665 ≈ 0.744**. It's the identical calculation the R and Python code run on real data, just small enough to verify by hand.

## Good, bad, and evil

The lecture closes with an explicit limitations rundown worth preserving as-is:
- **Good:** cheap — no labels required, and labels are expensive to create and maintain.
- **Trade-off:** clustering "always works" in the sense that it always produces *an* answer, but there are many competing clustering methods, the differences between them are often small and hard to characterize, and picking the right one requires actually understanding what you're doing rather than defaulting to k-means.
- **Real risks:** the curse of dimensionality; clusters that reflect poor data quality rather than real structure; non-determinism from random initial centroids (k-means can converge to a local rather than global optimum, and doesn't improve much with more data since it's working with averages); and — specific to k-means' centroid-distance mechanic — poor results on non-spherical clusters or clusters of very unequal size.

## Lab: are NBA players over- or underpaid for performance?

`Lab-Clustering.ipynb` reframes the same workflow as a scenario: students play VP of Analytics for an underperforming NBA team, clustering player performance stats (merging season totals with salary data) to find players who are high-performing but not highly paid — recruitment targets. The lab structure mirrors the in-class example step for step: select and scale features, run k-means, use the elbow method *and* silhouette score to justify a choice of `k`, evaluate cluster quality with variance explained, then interpret the clusters against a real-world variable — salary — that was deliberately left out of the clustering itself, exactly the same validation move as re-coloring the house-votes plot by party.

## R → Python, and the two versions of this deck

Like KNN, this material was originally taught in R and is mid-conversion to Python. It shows in the current slide deck itself: the walkthrough narrative (kmeans outputs, `betweenss`/`totss`, `NbClust`) is still illustrated with R script screenshots (`kmeans_obj_Dem$betweenss`, `ggplot2`, `install.packages("NbClust")`), even though the actual in-class code students now run (`cluster_example.py`) is Python/scikit-learn. Treat the slide-embedded R output as historical illustration of the same math the Python code computes, not as the current runnable path.

Two versions of the slide deck exist in the source folder. `Clustering_InClass_9.28.21.pdf` (57 slides, dated Sept. 28, 2021) covers the house-votes k-means walkthrough only — install/load packages in R, run `kmeans()`, visualize, validate against party, elbow method, `NbClust`, limitations — and stops there. `Clustering_InClass.pptx` is the version kept canonical for this page: it's a strict superset, retaining the same R-walkthrough content but adding material absent from the 2021 deck — a "write the k-means steps in your own words" discussion prompt, a link to an interactive k-means visualization tool (naftaliharris.com) for building intuition before the math, an explicit Euclidean-distance slide, a second real-world case study (Telenor customer-churn clustering), and an outline line pointing at the NBA lab as the follow-on "multi-dimensional clustering" exercise. The 2021 PDF is noted here for the record but was not otherwise ingested, per this project's curation rule for superseded lecture versions.

## Related topics

- [[course/knn]] — both are distance-based methods, but `k` means something different in each. In KNN, `k` is the number of labeled neighbors to poll for a vote when classifying a *new* point — a supervised hyperparameter tuned against held-out accuracy. In k-means, `k` is the number of groups to partition *unlabeled* data into — there's no accuracy to tune against, only internal fit measures (variance explained, silhouette) and, if you're lucky, an external label to validate against afterward (party, salary). Both methods share the same prerequisite, though: because everything reduces to distance, features need to be scaled before clustering or the variable with the largest raw range will dominate the distance calculation, exactly as flagged on the KNN page.
- [[course/pca]] — the course's other unsupervised method. PCA reduces the number of features rather than grouping rows, and it is likewise scale-sensitive (standardize first). It also responds to the curse of dimensionality that this page lists as a clustering risk.

## Sources

- `raw/course/DS-3021/kmeans-clustering/2026-03-10-clustering-slides.pptx` — canonical lecture deck (kept over the 2021 PDF version — see note above)
- `raw/course/DS-3021/kmeans-clustering/2026-03-10-cluster-example.py` — in-class Python/sklearn code: house-votes clustering, elbow loop, variance explained, silhouette score, cluster-as-feature decision tree comparison
- `raw/course/DS-3021/kmeans-clustering/2026-03-10-clustering-lab.ipynb` — NBA player performance/salary clustering lab assignment
- `raw/course/DS-3021/kmeans-clustering/2024-10-29-kmeans-worked-example-slide.pptx` — standalone slide with the hand-worked 1-D k-means example (D = {2,4,10,12,3,11,20,25,30}, k=2), iteration by iteration; duplicates the same slide in the main deck
- `raw/course/DS-3021/kmeans-clustering/2026-01-13-manual-variance-explained.xlsx` — hand-worked variance-explained calculation on a toy dataset
- `raw/course/DS-3021/kmeans-clustering/2024-10-29-house-votes-clusters-raw.png` — house-votes cluster plot, no party information (the "before" validation step)
- `raw/course/DS-3021/kmeans-clustering/2024-10-29-house-votes-clusters-by-party.png` — same plot re-colored by actual party affiliation (the "after" validation step)
