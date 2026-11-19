# Final Project

The capstone of DS-3021. Weeks 14–15 of the syllabus are "Final Project Prep," and projects are presented in the final exam period (May 1st, 9–12 in the Spring 2026 syllabus). The final project is worth **25% of the course grade** under the syllabus heading "Final Oral/Project." It falls under the **Practice** area of the course's data science framework (see [[course/course-overview]]): small teams answer their own data-driven question by working through the whole data science lifecycle with the models from the semester.

⚠️ Possible conflict — see "How the assignment changed across versions" below: there are three versions of the instructions (2024 R-era overview, 2026-04-15 instructions, 2026-04-16 revised rubric). They differ on the deliverable format, which models are listed, whether a fairness assessment is required, and how the project is graded.

## Purpose

All three versions use nearly the same purpose statement: **"To answer a data driven question by deploying the data science lifecycle."** By the end, students should know what it is like to:
- develop their own questions,
- match those questions with data,
- test whether the data, through a model, can answer the question,
- develop a final solution, and
- explain the weaknesses and benefits of their approach.

The goal is to "pull all the skills together that we have worked on all semester, essentially living the life of a practicing data scientist." The project is also pitched as a start on a **data project portfolio** on a student's repo or website.

## The task

- **Teams:** students work in their lab groups or project groups.
- **Question:** develop and answer one **discrete question** about a dataset of the group's choice.
- **Data:** use one of the suggested sources below, a dataset used in the past, or any other dataset, as long as it is *not one from class*.
- **Walk through the lifecycle:** the 2024 and revised 2026 versions ask for a cleanly documented deliverable that takes the reader through the project step by step, referencing the data science lifecycle and working through each stage deliberately.

### Models students can use

The revised 2026 rubric lists the following. Students may combine topics, extend them, or use more than one model to find the best one.
- Linear regression for prediction ([[course/linear-regression]])
- kNN ([[course/knn]])
- Clustering with kMeans ([[course/kmeans-clustering]])
- Logistic regression ([[course/logistic-regression]])
- Decision trees, for classification or regression ([[course/decision-trees]], [[course/dt-regression]])
- "Or try Random Forest – lots of Decision Trees" ([[course/ensemble-methods]])

## Required sections

These are the sections in the revised 2026 rubric. Notes on earlier versions are included where they differ.

1. **Question and background.** What the data is and why the group is asking the question, in a couple of paragraphs. The 2024 version also suggests citing previous research or evidence, and required groups to present their question to the instructor in office hours before starting.
2. **Exploratory Data Analysis.** Initial summary statistics and graphs, focused on the variables the group thinks matter. "Don't just show a bunch of graphs, you need a story". Each graph should relate to the question (see [[course/data-prep]]).
3. **Methods.** The techniques used and their results; this is the **training/tuning phase**. The 2026-04-15 version also asks groups to explain *why* they chose each technique.
4. **Evaluation of your model.** Choose appropriate metrics and explain the output in terms of the question; this is the **testing phase** (see [[course/ml-eval-metrics]]).
5. **Conclusions.** What the results say about the question, given the model's limitations.
6. **Team contribution** (new in the revised rubric). A high-level summary of what each member contributed. Several people working on the same part is fine.

The 2024 version also required:
- **Fairness assessment**, "if necessary, should you happen to have any protected classes" (see [[course/ml-bias]]).
- **Future work**: what further analysis is needed, or what limited this one.

The 2026-04-15 version drops both as separate sections but still lists "fairness assessment (if applicable)" in its evaluation criteria.

### Suggested team roles (revised rubric; optional)

| Role | Responsibility |
|---|---|
| Data Engineer | Sets up the environment: repo, data pipeline, requirements, Codespace, data cleaning and prep (see [[course/dev-environment]]) |
| Researcher | Background and building the question |
| Model builder | Builds the initial models for testing |
| "Red Team" | Evaluates the model and reviews the code for quality or performance gaps |
| Final reviewer | Does one last pass over everything to check quality and a clear message |

## Technical expectations (revised rubric)

- **Simple is better.** Don't make code or models more complicated than they need to be.
- **Document what you are doing as you go.**
- Follow **PEP 8** style standards.
- **Use the pipeline method.**
- Use a **robust model-development process**. Tuning and testing must be clearly documented, with every iteration shown and explained (Model 1, Model 2, Model 3, and so on).
- **Use GitHub** for code development and storage.

## Deliverables

| Version | Format | Submit / publish |
|---|---|---|
| 2024 overview | "Publishable" **RMarkdown** document, presented as "a cleanly knitted final presentation" | Publish the HTML to **RPubs**, or as a GitHub Pages site that sits "on top" of the repo |
| 2026-04-15 instructions | A **notebook** (Python) | Python files, the data used, and a link to the supporting GitHub repo |
| 2026-04-16 revised rubric | A cleanly documented final **notebook** | Code, data, and a link to the repo |

The 2026 syllabus adds that students "will be expected to build the project using github and develop a free standing website built with one of the tools presented in class (GitHub.io or MYST Book)." It also mentions "well-annotated code to include summary information in report form." The 2026 intro slides say the end-of-semester presentation "likely will require you to build a standalone website."

## Grading and the oral evaluation

- **Weight in the course:** 25% ("Final Oral/Project," per the syllabus). The syllabus also says the instructor writes **individual questions** for each student to answer with their group, "to give you practice explaining your models while also ensuring you understand what you are doing."
- **Revised rubric split:** "Oral evaluation will count for 60% of the final grade; documentation and quality of the materials will be 40%." In context this is a split of the *project* grade, not the whole course grade, which the syllabus fixes at 25% for the project.
- **Collaboration penalty:** there is a peer collaboration evaluation. A student whom peers judge as not contributing "could lose up to two full letter grades."
- **2026-04-15 evaluation criteria:**
  - documentation of each step in the process;
  - correct application and interpretation of the methods, the evaluation and, if applicable, the fairness assessment;
  - a conclusion that addresses the question using the model output;
  - quality of the oral presentation.

**Oral question areas:**
- **Method-based:** the model you chose, how you used it, and why you chose that method.
- **Problem-based:** the problem (question) you proposed, and how it connects to your method and conclusions.
- **Evaluation:** how you evaluated the model, and what conclusions about your question follow from the model's weaknesses.

**Oral format:** slots on Friday, 9–12 and 1:30–4:30, **15 minutes per group**: 12 minutes of questions and 3 minutes for the graders to take notes.

## How the assignment changed across versions

- **Language and tools.** The 2024 version is from the R era: RMarkdown, knitting, RPubs. The 2026 versions use Python notebooks, Python files, GitHub repos, and pipelines.
- **Models.** The 2024 list is kNN, kMeans clustering, decision trees, and random forest. The 2026-04-15 list replaces random forest with linear/multiple linear regression and logistic regression. The 2026-04-16 list includes all of these and brings random forest back as an option.
- **Fairness and future work.** These are required sections in 2024. In 2026, fairness appears only as "if applicable" in the evaluation criteria, and future work is dropped.
- **Grading.** Only the revised rubric gives the 60/40 oral/documentation split, the collaboration penalty, the team-contribution section, and the suggested roles.
- **Deliverable.** The syllabus asks for a free-standing website (GitHub.io or MyST). The revised rubric asks only for code, data, and a repo link. The sources don't settle which requirement applies; ask the instructor.

## Potential data sources (all versions)

- Google Dataset Search
- A COVID-19 data-source list (hosted on GitHub)
- data.world
- UCI Machine Learning Repository
- Data is Plural, a shared spreadsheet of datasets

## Related topics

- [[course/course-overview]]: the syllabus grading weight, the schedule, and the instructor's data science lifecycle that the project follows.
- [[course/dev-environment]]: GitHub, Codespaces and requirements setup, which the Data Engineer role owns.
- [[course/ml-eval-metrics]]: choosing and interpreting evaluation metrics for the Evaluation section.
- [[course/ml-bias]]: the fairness assessment when a dataset includes protected classes.
- Model pages: [[course/knn]], [[course/kmeans-clustering]], [[course/linear-regression]], [[course/logistic-regression]], [[course/decision-trees]], [[course/dt-regression]], [[course/ensemble-methods]].

## Sources

- `raw/course/DS-3021/final-project/2024-10-29-final-project-overview.md`: 2024 overview (RMarkdown/RPubs, fairness assessment and future-work sections, question check-in in office hours)
- `raw/course/DS-3021/final-project/2026-04-15-final-project-instructions.docx`: 2026 instructions (Python notebook, model list including regression and logistic regression, evaluation criteria)
- `raw/course/DS-3021/final-project/2026-04-16-final-project-instructions-rev.docx`: revised 2026 rubric (team roles, technical notes, 60/40 oral/documentation grading, collaboration penalty, oral question areas and schedule)
