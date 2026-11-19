# Course Overview: Foundations of Machine Learning (DS 3021)

Week 1 of DS-3021 and the frame for everything after it. This page pulls together the syllabus (Spring 2026) and the opening-lecture slide decks, which present data science as "a field for the data age": why the field exists, how UVA's School of Data Science (SDS) defines it, where this course sits inside that definition, and the course's logistics, schedule, grading and policies. The course is taught by the SDS faculty member who serves as Director for Undergraduate Programs and who, with Rafael Alvarado and others, built the SDS framework of the field described below. A TA supports the course and grades lab notebooks.

⚠️ Possible conflict — see "Course identity" below: the syllabus front matter titles the course "DS 3023 Foundations of Data Science Syllabus," while the body of the same syllabus gives the catalog number as **DS 3021** and the class title as **Foundations of Machine Learning**.

## Course identity and logistics (syllabus)

- **Class title:** Foundations of Machine Learning. **Subject area and catalog number:** Data Science, DS 3021. Undergraduate, graded A–F.
- **Term and time:** Spring 2026, Tuesday/Thursday 2:00–3:15 PM, School of Data Science Building, room 205. The syllabus is dated 01/13/2026.
- **Office hours:** the instructor holds them in person (SDS Building office) or virtually on Discord, Monday 10–12 or by appointment. TA office hours are TBD.
- **Course materials:** the online "Foundations of Machine Learning" book (MLBook, hosted on GitHub Pages), which also has a digital copy of the syllabus and an introduction video.
- **Conflict:** the syllabus front matter says "DS 3023 Foundations of Data Science", but the body says DS 3021 / Foundations of Machine Learning. The body agrees with the rest of the course materials (DS-3021).

## Course philosophy

The syllabus frames the course around three questions: what is data science, why is it becoming so important, and what does it take to succeed in the field. Its main principle is **"creative thinking, not blind implementation"**: "If you cannot answer why you are doing something, not only will you discover no new knowledge but you will also create new problems versus solving them."

- **Lab-based, team-based pedagogy.** Much of the work can and should be done with classmates. The course is applied, but it also includes theory and discussion sessions.
- **Mixed backgrounds are expected.** Office hours, TA support sessions and team work are meant to let students learn from each other.
- **"Live the life of a data scientist."** Students discover new knowledge, share it with peers, and learn to use the wider data science ecosystem.
- **Weekly rhythm.** Students watch short video lectures and do readings *before* class. Class time is the lab portion, mostly team-oriented ML assignments.
- The syllabus warns that the course "will move rather quickly and can be demanding at times."

### Learning objectives

1. Gain experience working in teams to solve data-driven problems with ML tools.
2. Be able to describe the field of Machine Learning.
3. Articulate the advantages and disadvantages of selected ML approaches.
4. Be able to select appropriate ML models given problems and data types.
5. Understand the importance of, and methods for, evaluating ML models.
6. Gain experience working with and supporting teammates.
7. "Have some fun along the way."

## Grading

| Component | Weight | What it is |
|---|---|---|
| Quizzes | 20% | Short, occasional (about 4) in-class quizzes of about 30 minutes each, possibly on paper. "Lots of opportunity to improve your scores." |
| Labs | 55% | In-class labs and assignments on most weeks. Submitted individually, but collaboration is encouraged. Delivered as stand-alone Jupyter notebooks graded by the TA. |
| Final Oral/Project | 25% | Group project on a dataset of the group's choice, with well-annotated code and a report, built using GitHub and shown on a free-standing website (GitHub.io or MyST Book). The instructor writes individual questions for each student to answer with their group. See [[course/final-project]]. |

**Grading scale:** 93–100 A; 90–92 A-; 87–89 B+; 83–86 B; 80–82 B-; 77–79 C+; 73–76 C; 70–72 C-; 60–70 D; below 60 F. The DS grading system also includes D+ and D-, and W for an official withdrawal, but the posted scale has only a single "D" band, written as 60–70. That band overlaps the C- band at 70.

## Tech stack and resources

- **VS Code + GitHub Codespaces:** virtual machines with a VS Code environment. See [[course/dev-environment]].
- **GitHub:** almost all course materials. Items are also posted on Canvas when needed.
- **Canvas:** assignment submission and class-wide communication.
- **Discord:** low-latency communication for groups and the whole class, also described as a way to build community.
- The syllabus also links an overview of installing Miniconda and VS Code.

**Reading list ("a starter Machine Learning Library"):** (A) the Foundations of ML supplemental book (MLBook); (B) *Evaluating Machine Learning Models* (O'Reilly, via the UVA Library); (C) *Python Machine Learning with PyTorch and Scikit-Learn*, 4th ed. This is the main book and the only one that isn't free; the School or instructor will help students who can't pay for it. (D) *Mathematics for Machine Learning* (free PDF); (E) *Machine Learning Engineering* (low cost); (F) the PEP 8 Python style guide; (G) Google's Machine Learning Crash Course; (H) the UVA Data Science Librarian, listed as "a human resource"; (I) the *Python Data Science Handbook*, free in UVA's O'Reilly database. The opening slides also point to a GeeksforGeeks Python tutorial, the *Python Data Science Handbook*, and the VS Code documentation for Python help.

## Schedule of topics (Spring 2026 syllabus)

The syllabus notes that the schedule can change to add topics, depending on student interest.

| Week | Theme | Topics / Lab | Reading |
|---|---|---|---|
| 1 | What is this "Machine Learning" you speak of? + Experimental Thinking | Synchronous short lab | — |
| 2 | What is ML + Data Science Lifecycle | Language of ML + Analytical Plan; Group Case Study lab | A. |
| 3/4 | Introduction to ML Concepts I | Data Preparation; ML Concepts lab ([[course/ml-foundations]], [[course/data-prep]]) | C. Ch. 4 + Quiz |
| 5 | Intro to ML Concepts II + Classification | ML process: kNN ([[course/knn]]) | C. pp. 98–103 |
| 6 | kNN + Evaluation | Evaluation Lab ([[course/ml-eval-metrics]]) | All of B, C Ch. 6 |
| 7 | Unsupervised: kMeans | "Let's gather together, but separately"; lab: NBA scout for the worst team in the league ([[course/kmeans-clustering]]) | — |
| 9 | Regression | Linear + Multiple Linear Regression; Regression Lab ([[course/linear-regression]]) | — |
| 10 | Special Regression | Logistic Regression, "like regression and classification had a child" ([[course/logistic-regression]]) | + Quiz |
| 11 | Sampling Methods | "How Bootstrap + Cross-Validation solves all the problems" | — |
| 12 | Nature's Perfect ML analogy: Trees Part I | Classification decision trees + calibration ([[course/decision-trees]]) | G. Ch. 14.1–14.3 |
| 13 | Trees Part II | Regression decision trees; lab "Predicting Income for Big Brother" ([[course/dt-regression]]) | F. Ch. 5, G. Ch. 8 + Quiz |
| TBD | Unsupervised: PCA | Feature engineering ([[course/pca]]) | TBD |
| TBD | Topic Exploration | Ensemble methods / Random Forest ([[course/ensemble-methods]]) | TBD |
| 14/15 | Final Project Prep | Final project overview | — |
| Exam | Final Project | Final period: May 1st, 9–12 | — |

The week numbers in this table come from the 2026 syllabus. Other topic pages in this wiki sometimes give different week numbers for the same topic, taken from older course materials.

An older version of the course map (the 2024 "framework overview" deck) lists a different, R-based sequence and tags each item with its part of the SDS framework. Its order is: What is DS (Practice), Programming R (Systems), GitHub (Systems), Programming Markdown (Design), ML Foundations, kMeans, Text Mining, kNN (Analytics), Reinforcement Lab (Practice), Evaluation (Value), Decision Trees, Random Forest (Analytics), Reinforcement Lab (Practice), Ethics/Bias in ML (Value; see [[course/ml-bias]]), and Final Projects (Practice). The 2026 syllabus replaces R with Python and adds regression, logistic regression and PCA.

## Why data science? "A field for the data age"

The opening lecture argues that data science matters because of how much data now exists, and even more because "our lives are being driven by this data and the algorithms that feed on it."

- **Scale of data:** about 4 billion social media users. The world generates about 44 zettabytes of data; "if a kilobyte was the size of a person, a zettabyte would be the size of Europe," or "6.6 stacks of iPads reaching from the earth to the moon."
- **2015 as "punctuated equilibrium":** long stability broken by sudden, radical change. By then about 90% of the world's data had been created in the previous three years. NVIDIA GPUs, open-source Spark and the iPhone, big gains in AI performance, fast growth in the data workforce and big-data research, and the housing bubble are all named as drivers.
- **Why ML has dominated:** more computing power, larger training datasets, better algorithms, and open-access software. ImageNet error rates fell from 30% in 2010 to under 3%, below the roughly 5% often cited as the human error rate.
- **AI, ML and DL as nested circles.** "Classical AI" example: medical systems that warn nurses about prescription interactions. Machine learning example: Amazon product recommendations, which use a boosted tree-based algorithm. Deep learning example: large language models. *Moving further into the circle, control and explainability decrease but predictive power increases.*
- **ML as a general-purpose technology (GPT):** like electricity or the internet, it applies across domains. Machines are trained "without the use of explicit instruction"; that is the "learning" in machine learning.
- **Human expertise still matters.** Deep Blue searched about 200 million chess positions per second, against Kasparov's 5–10, yet they played at about the same level. An amateur beat an AI Go program with a simple trick. A machine-vs-human table gives machines fast calculation, repetitive tasks and following predefined rules. It gives humans understanding context, thinking through the problem, asking the right questions, selecting tools and interpreting results.
- The 2024 deck also includes a hand-worked **k-means example**: D = {2,4,10,12,3,11,20,25,30}, k = 2, starting means 3 and 4. It converges to K1 = {2,3,4,10,11,12} (mean 7) and K2 = {20,25,30} (mean 25). See [[course/kmeans-clustering]]. The same deck has a timeline of language models: attention (2014), seq2seq (2014), Transformers ("Attention is All You Need," 2017), GPT (2018), BERT (2018) and GPT-2 (2019, initially withheld over ethical concerns).

### Brief history of the field

- **1960:** Peter Naur used "data science" as a substitute for computer science.
- **1962:** John Tukey's "The Future of Data Analysis" argued that data analysis is "intrinsically an empirical science."
- **1977:** IASC founded. **1989:** KDD founded. **2002/3:** the *Data Science Journal* and the *Journal of Data Science* founded.
- **2009:** Hal Varian, Google's chief economist, said statistician would be "the sexy job in the next ten years."
- **2010:** Kirk Borne and other astrophysicists submitted "Data Science for the Masses" to the Astro2010 Decadal Survey.
- **2012:** Harvard Business Review called data scientist "the Sexiest Job of the 21st Century." DJ Patil says he coined the term in 2008 with Jeff Hammerbacher for their roles at LinkedIn and Facebook.
- A Google Ngrams in-class exercise compares "Data Science," "Artificial Intelligence," "Machine Learning," "Dataset" and "Internet" back to 1950.

### Why the field needs to change: ML bias cases

These cases set up the course's Value strand; see [[course/ml-bias]].

- **YouTube.** Around 2011–12 YouTube switched from optimizing clicks to optimizing hours watched, and brought in Google Brain to train a deep neural network for recommendations. Recommendations drifted toward more and more extreme content, for example from jogging to ultra-marathons or from political leaning to extremism. "Machine Learning detected the pattern and optimized for it, it's that simple." By 2021 viewers watched about 1 billion hours of YouTube per day, about 137,000 years every day.
- **Social media addiction meta-analysis** (Cheng et al., 2021): 50 studies, 37,798 individuals. About 200M users are "very severe," 520M "severe" and 1B "moderate to severe," which is almost half of all social media users.
- **Google Cloud Natural Language API** (launched 2016). In 2017, tests found that "I'm Christian" scored positive while "I'm a Jew" and "I'm a gay black woman" scored negative. The API was trained on news and social media data, and Google apologized.
- **Amazon recruiting engine.** It rated résumés from one to five stars and was trained on 10 years of mostly male hiring, so it systematically scored women lower. Amazon abandoned it.
- **Facebook:** fake election news outperformed real news, and the platform played a role in the Myanmar genocide.

## Defining the field: the SDS framework

Common definitions of data science include the "classic" Venn diagram (the slides argue it overextends, since ML does need domain knowledge), CRISP-DM, and more recent lifecycle diagrams. The slides say these models share one weakness: they are "more Function than Form." UVA SDS wanted "not just pipeline/process but a conceptual model of the field," and calls the degree a **"Liberal Arts Degree for the Data Age."** The framework has four areas plus Practice:

| Area | Common theme | Key tensions | Values |
|---|---|---|---|
| **Value** | Human value: ethics combined with business planning, policy making and research motivation | enterprise vs ethics; private interest vs public good (see Plato's *Republic*) | responsibility, diversity, inclusion, flourishing, excellence |
| **Design** | Communication: human-machine interaction when consuming data and producing data products (database design, curation, visualization, HCI) | discovery vs product; user-facing vs analyst-facing | openness, authenticity, beauty, form and function |
| **Systems** | Building: infrastructure for big data (volume, velocity, variety) and pipelines in development and production; "computer technology as opposed to computer science" | development vs production; volume vs speed | speed, stability, robustness, resilience, uptime |
| **Analytics** | Mathematical models and methods: "what most consider to be the heart of data science," statistics combined with machine learning, optimization, signal processing and network analysis | inference vs prediction; analysis vs simulation | accuracy, precision, validity, truth, convergence, explainability |

**Practice** brings people together to combine expertise from the four areas. Data science teams work with each other and with outside partners on solutions that are "responsible, authentic, effective, and efficient." Practice is where activities such as data wrangling actually live, and where the core areas meet domain knowledge. At the degree level this means a dozen or more external concentrations (for example neuroscience, digital humanities, forecasting, educational analytics), research experiences across all three years, and pathways into applied graduate programs. The slides call this "a non-trivial departure from most current undergraduate models."

**This course: "a Bit of Everything but Mostly Analytics."** In the 2026 deck:
- **Systems:** the working environment and version control.
- **Design:** a presentation at the end of the semester, likely with a standalone website.
- **Value:** ethical issues in developing ML solutions.

### The instructor's data science lifecycle

The instructor's own lifecycle has four phases separated by gate reviews (G1–G3). The slide is a diagram, so the grouping of steps into phases below is reconstructed from its extracted text:
1. **Idea Development:** question ID, business understanding, data acquisition (ETL).
2. **Idea Testing:** data understanding (EDA), initial model(s), evaluation criteria and a value metric.
3. **Solution Development:** model creation and training, feature engineering, hyperparameter optimization.
4. **Solution Deployment:** deployment, data drift and model drift analysis, performance against the value metric, and reports, dashboards and products.

[[course/final-project]] asks students to work through this lifecycle.

### The DS degree argument (2026 deck)

- **Data literacy is digital-society literacy.**
- **The knowledge economy:** production and consumption based on intellectual capital. Together with Adam Smith's division of labor (1776), this leads to **hyper-specialization**. The conclusion is that success will require "a combination of technical proficiency and subject area expertise," which UVA's DS degree model is meant to provide.
- **Human capital vs market signaling:** two theories of what education is worth.
- **Type 1 and Type 2 reasoning** (Chiesi, Primi & Morsanyi, 2011). Type 1 is fast and automatic, and often wrong even when it feels right. Type 2 is slow, conscious and serial.
- The degree centers on **problem solving through probabilistic reasoning**. The deck claims this serves both the human-capital and the signaling theories, prepares students to specialize by market sector, and creates the data-literate society the data age requires.
- A jobs analysis based on 200 "Data Scientist" postings nationwide, expanding to thousands in DC, NYC and Silicon Valley, maps job requirements onto the framework.

## Activities and assignments

- **Dream Job Discussion** (posted on Canvas). Find a real or invented job with "data scientist" in the title and answer four questions:
  1. What do I love about it?
  2. Which parts of the DS framework does it involve?
  3. What do I need to know to do it?
  4. How much of that do I already know (0–100%)?
- **Icebreaker:** "Walk around, get to know each other – 5 minutes."
- **Course culture:** the 2024 deck lists Discord, the syllabus, Python resources and the **LMI Challenge** (an external competition) as early course items.

## Related topics

- [[course/dev-environment]]: the Systems strand. VS Code, GitHub, Codespaces and requirements files are set up in the first weeks.
- [[course/ml-foundations]] and [[course/data-prep]]: the first technical weeks (ML concepts I/II, data preparation).
- [[course/ml-bias]]: expands on the YouTube, Amazon and Google NLP cases and the Value strand.
- [[course/final-project]]: the Practice capstone, worth 25% of the grade.

## Sources

- `raw/course/MLBook/course-overview/2026-05-21-syllabus.md`: Spring 2026 syllabus (logistics, objectives, grading, tech stack, readings, schedule, policies)
- `raw/course/DS-3021/course-overview/2024-10-29-foundations-ml-degree-slides.pptx`: "Data Science a Field for the Data Age" opening lecture (data age, AI/ML/DL, history, bias cases, SDS framework, course resources)
- `raw/course/DS-3021/course-overview/2024-10-29-intro-framework-overview-slides.pptx`: "Defining a Field, a School and a Curriculum" (classic views, SDS framework, the instructor's lifecycle, older course map, Dream Job discussion)
- `raw/course/DS-3021/course-overview/2026-01-13-ds-degree-for-the-data-age-slides.pptx`: "Data Science Degree for the Data Age" (knowledge economy, human capital vs signaling, probabilistic reasoning, SDS framework)
- `raw/course/DS-3021/course-overview/2026-03-10-intro-defining-the-field-slides.pptx`: short 2026 intro deck ("a Bit of Everything but Mostly Analytics," Dream Job discussion)
