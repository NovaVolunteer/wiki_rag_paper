#!/usr/bin/env python
"""
One-time bulk ingest that expands the corpus from 7 topics to the full course.

Copies every remaining text-bearing DS-3021 file (excluding the xx_* folders,
which are not part of the course as taught) and the published chapters of the
MLBook (https://novavolunteer.github.io/MLBook/, source repo
github.com/NovaVolunteer/MLBook) into raw/, using the same layout and naming
convention as the original 7-topic ingest:

    raw/course/<source>/<topic>/<YYYY-MM-DD>-<short-description>.<ext>

<source> is DS-3021 or MLBook. Dates are the source file's modified date
(DS-3021) or the MLBook commit date. raw/ is immutable: existing files are never
overwritten.

Curation rules (same as wiki/course/README.md "Curation notes"):
  * exact duplicates of a file already in raw/ are skipped;
  * where DS-3021 and the MLBook both hold a file with the same name, the MLBook
    copy (the current published version) is kept;
  * superseded slide decks, PDF exports of included decks, and .py exports of
    included notebooks are skipped;
  * non-text files (csv, xlsx, images) and the binary .ppt (no converter) are
    skipped.
Every skip is listed in SKIPPED with its reason.

    python ingest_course_materials.py --mlbook <path to MLBook clone> [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

MLBOOK_COMMIT = "40b4199e3d85b8dbf7921c25eac702f510b9c1aa"
MLBOOK_DATE = "2026-05-21"

# (source path relative to DS-3021/, topic, short description)
DS3021 = [
    # course-overview
    ("Introduction_Framework_Overview_8.25.22.pptx", "course-overview", "intro-framework-overview-slides"),
    ("Introduction_Framework_Overview_1.pptx", "course-overview", "intro-defining-the-field-slides"),
    ("A Data Science Degree for the Data Age.pptx", "course-overview", "ds-degree-for-the-data-age-slides"),
    ("A Degree for the Data Age_Foundations_ML.pptx", "course-overview", "foundations-ml-degree-slides"),
    # python-basics
    ("02_function_basics/02_changes.docx", "python-basics", "week02-r-to-python-changes"),
    ("02_function_basics/copilot_example.py", "python-basics", "copilot-example"),
    # ml-foundations
    ("04_ML_Concepts_I_Foundations/04_changes.docx", "ml-foundations", "week04-r-to-python-changes"),
    ("04_ML_Concepts_I_Foundations/machine_learning_bootcamp.pptx", "ml-foundations", "ml-bootcamp-slides"),
    ("04_ML_Concepts_I_Foundations/Machine_Learning_In_Class.pptx", "ml-foundations", "ml-in-class-slides"),
    ("04_ML_Concepts_I_Foundations/machine_learning_overview.pdf", "ml-foundations", "ml-overview-reading"),
    ("04_ML_Concepts_I_Foundations/ML1_inclass.ipynb", "ml-foundations", "ml1-in-class-notebook"),
    ("04_ML_Concepts_I_Foundations/Walker_Advertising_Case_Study.pdf", "ml-foundations", "walker-advertising-case-study"),
    # data-prep
    ("05_ML_Concepts_II_Data_Prep/Conversion_doc_ML_Bootcamp.docx", "data-prep", "ml-bootcamp-r-to-python-conversion"),
    ("05_ML_Concepts_II_Data_Prep/example_dropna.py", "data-prep", "example-dropna"),
    ("05_ML_Concepts_II_Data_Prep/machine_learning_bootcamp.pptx", "data-prep", "ml-bootcamp-data-prep-slides"),
    ("05_ML_Concepts_II_Data_Prep/machine_learning_bootcamp_II.pdf", "data-prep", "ml-bootcamp-II-reading"),
    ("05_ML_Concepts_II_Data_Prep/Train.TestLogReg.ipynb", "data-prep", "train-test-logreg-notebook"),
    # linear-regression
    ("05.1_reg_data_prep/04_hedonic_pricing.ipynb", "linear-regression", "hedonic-pricing-notebook"),
    ("05.1_reg_data_prep/linear_regression_intro.ipynb", "linear-regression", "linear-regression-intro-notebook"),
    ("05.1_reg_data_prep/linearRegression.ipynb", "linear-regression", "linear-regression-lecture-notebook"),
    ("05.1_reg_data_prep/Log Conversion.pdf", "linear-regression", "log-conversion-reading"),
    ("05.1_reg_data_prep/makeup_regression.py", "linear-regression", "makeup-regression-script"),
    # logistic-regression
    ("05.1_reg_data_prep/Logistic Regression_26.pptx", "logistic-regression", "logistic-regression-slides"),
    # existing topics: files not yet ingested
    ("06_ML_Concepts_II_KNN/data_preparation.py", "knn", "knn-data-preparation-script"),
    ("09_DT_Regression/DT_Reg_Lab.ipynb", "dt-regression", "dt-regression-lab-v2"),
    ("04_ML_Concepts_I_Foundations/Welcome to PowerPoint.pptx", "kmeans-clustering", "kmeans-worked-example-slide"),
    # final-project
    ("Final_Project/final_project_overview.md", "final-project", "final-project-overview"),
    ("Final_Project/final_project.docx", "final-project", "final-project-instructions"),
    ("Final_Project/final_project-1.docx", "final-project", "final-project-instructions-rev"),
]

# (path relative to MLBook repo root, topic, short description) -- the published TOC (myst.yml)
MLBOOK = [
    ("README.md", "course-overview", "syllabus"),
    ("class_materials.ipynb", "dev-environment", "class-materials-vscode-github"),
    ("daytwostuff.md", "dev-environment", "day-two-repo-venv-setup"),
    ("whatisml.ipynb", "ml-foundations", "what-is-ml"),
    ("dslifecycle.ipynb", "ml-foundations", "ds-lifecycle"),
    ("03_ML_Concepts_I_Foundations/Lab_Overview.md", "ml-foundations", "foundations-lab-overview"),
    ("Part_II/02_Lecture_Python.ipynb", "python-basics", "python-lecture-notebook"),
    ("Part_II/Class_II_Python.ipynb", "python-basics", "python-class-II-notebook"),
    ("04_Data_Prep_Pipe/ml_bootcamp.ipynb", "data-prep", "ml-bootcamp-notebook"),
    ("04_Data_Prep_Pipe/ch4.ipynb", "data-prep", "acquiring-data-from-apis"),
    ("04_Data_Prep_Pipe/ML_Bootcamp_Lab.md", "data-prep", "ml-bootcamp-lab"),
    ("05_linear_regression/types_reg_eval.ipynb", "linear-regression", "regression-types-and-evaluation"),
    ("05_linear_regression/linear_reg_overview.ipynb", "linear-regression", "linear-regression-overview"),
    ("05_linear_regression/multi-features.ipynb", "linear-regression", "multiple-features"),
    ("05_linear_regression/ols.ipynb", "linear-regression", "ols"),
    ("05_linear_regression/polynomial.ipynb", "linear-regression", "polynomial-regression"),
    ("05_linear_regression/assignment.ipynb", "linear-regression", "regression-assignment"),
    ("05_linear_regression/logistic_regression_overview.ipynb", "logistic-regression", "logistic-regression-overview"),
    ("06_knn/knn_overview.ipynb", "knn", "knn-overview"),
    ("decision_trees/08_DT_Class/DT_Class_Lab.ipynb", "decision-trees", "dt-class-lab"),
    ("PCA/pca.ipynb", "pca", "pca"),
    ("PCA/pca_examples.ipynb", "pca", "pca-examples"),
]

SKIPPED = {
    # exact duplicates of files already in raw/course/DS-3021/<topic>/ (or of the MLBook copy)
    "MLBook 06_knn/KNN.ipynb": "duplicate of knn/2024-10-29-knn-lecture-notebook.ipynb",
    "MLBook 06_knn/grad_lab.ipynb": "duplicate of knn/2024-10-29-knn-graduation-lab.ipynb",
    "MLBook 07_ML_Eval_Metrics/knn_eval_example.ipynb": "duplicate of ml-eval-metrics/2025-06-09-knn-eval-example-notebook.ipynb",
    "MLBook 07_ML_Eval_Metrics/Model_Metrics_Lab_Updated.md": "duplicate of ml-eval-metrics/2025-06-09-model-metrics-lab.md",
    "MLBook decision_trees/08_DT_Class/decision_trees.ipynb": "duplicate of decision-trees/2025-02-13-decision-trees-lecture-notebook.ipynb",
    "MLBook title.md": "title page only",
    "DS-3021 02_function_basics/Class_II_Python.ipynb": "duplicate of MLBook Part_II/Class_II_Python.ipynb",
    "DS-3021 04_ML_Concepts_I_Foundations/Lab_Overview.md": "duplicate of MLBook 03_ML_Concepts_I_Foundations/Lab_Overview.md",
    # same-named older version; MLBook copy kept
    "DS-3021 02_function_basics/02_Lecture_Python.ipynb": "older version of MLBook Part_II/02_Lecture_Python.ipynb",
    "DS-3021 05_ML_Concepts_II_Data_Prep/ml_bootcamp.ipynb": "older version of MLBook 04_Data_Prep_Pipe/ml_bootcamp.ipynb",
    "DS-3021 05_ML_Concepts_II_Data_Prep/ML_Bootcamp_Lab.md": "older version of MLBook 04_Data_Prep_Pipe/ML_Bootcamp_Lab.md",
    "DS-3021 05.1_reg_data_prep/ols.ipynb": "older version of MLBook 05_linear_regression/ols.ipynb",
    "DS-3021 05.1_reg_data_prep/assignment.ipynb": "older version of MLBook 05_linear_regression/assignment.ipynb",
    "DS-3021 05.1_reg_data_prep/revised_linearRegression.ipynb": "older (2025-03) version of linearRegression.ipynb (2026-03)",
    # .py exports of included notebooks
    "DS-3021 04_ML_Concepts_I_Foundations/ML1_inclass.py": ".py export of ML1_inclass.ipynb",
    "DS-3021 05_ML_Concepts_II_Data_Prep/ml_bootcamp.py": ".py export of ml_bootcamp.ipynb",
    "DS-3021 06_ML_Concepts_II_KNN/KNN.py": ".py export of the KNN lecture notebook (already excluded in the 7-topic ingest)",
    "DS-3021 06_ML_Concepts_II_KNN/grad_lab.py": ".py export of the graduation lab (already excluded)",
    "DS-3021 08_DT_Class/Decision Trees.py": ".py export of the decision-trees notebook (already excluded)",
    # superseded decks / PDF exports of included decks
    "DS-3021 06_ML_Concepts_II_KNN/Knn and Prob v1.pptx": "deck version of the included Knn and Prob PDF",
    "DS-3021 08_DT_Class/Decision_Trees_4.28.21.pdf": "superseded 2021 deck (already excluded)",
    "DS-3021 08_DT_Class/Decision_Trees_3.24.22.pptx": "same deck as dt-regression slides (already excluded)",
    "DS-3021 10_kMeans Clustering/Clustering_InClass_9.28.21.pdf": "superseded 2021 deck (already excluded)",
    "DS-3021 12_Ensemble_RF/Trees_Ensemble_11.9.20.pdf": "PDF export of the included ensemble deck",
    "DS-3021 14_ML_Bias/Bias_Presentation.pptx": "deck version of the included bias PDF",
    # other
    "DS-3021 04_ML_Concepts_I_Foundations/# Brian's Case Study.md": "empty template (headings only)",
    "DS-3021 05.1_reg_data_prep/15.Simple Linear Regression.ppt": "legacy binary .ppt; no text extractor available",
    "DS-3021 xx_*": "not part of the course as taught (instructor decision, 2026-11-19)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlbook", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = []
    for rel, topic, desc in DS3021:
        src = Path("DS-3021") / rel
        date = dt.date.fromtimestamp(src.stat().st_mtime).isoformat()
        plan.append((src, Path("raw/course/DS-3021") / topic / f"{date}-{desc}{src.suffix.lower()}"))
    for rel, topic, desc in MLBOOK:
        src = args.mlbook / rel
        plan.append((src, Path("raw/course/MLBook") / topic / f"{MLBOOK_DATE}-{desc}{src.suffix.lower()}"))

    for src, dst in plan:
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        status = "exists" if dst.exists() else "copy"
        print(f"[{status}] {src} -> {dst}")
        if status == "copy" and not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"{len(plan)} files; {len(SKIPPED)} skip rules")


if __name__ == "__main__":
    main()
