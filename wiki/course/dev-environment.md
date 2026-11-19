# Development Environment: VS Code, GitHub, Codespaces & Requirements

Set up during the first week of DS-3021. The second class meeting ("Day Two Stuff") is a hands-on setup session. In the course's framework of the field this is the **Systems** strand, "working environment and version control" (see [[course/course-overview]]). Every later lab and the [[course/final-project]] assume students can get course code from GitHub, open it in a working VS Code environment, and manage Python packages with `pip` and a `requirements.txt` file.

## Where class materials live

The course distributes materials in three places:

1. **GitHub.** Used throughout the course, with a new repo for each module. Each repo contains Python files with code examples and exercises, the datasets for those exercises, and the lecture slides as PDFs.
2. **The supplemental interactive textbook** (the MLBook online book). It hosts videos and background material, including the setup page this wiki page draws on.
3. **Canvas.** Assignments are submitted and grades are posted here.

The syllabus lists the same stack: **VS Code + GitHub Codespaces** ("virtual machines with a VS Code environment"), **GitHub** (almost all course materials), **Canvas** (submissions and class-wide communication), and **Discord** (low-latency communication for groups and the class). It also links an overview of installing Miniconda and VS Code.

## The core workflow: fork, then clone

At the start of the course every student:

1. **Forks** the course repo to their own GitHub page. The fork is the student's own copy of the instructor's repo.
2. **Clones** the fork through VS Code into a local directory that they will use for the rest of the class.

## The three technologies

The materials page says the environment is built from **VS Code, Dev Containers (Docker), and GitHub**, and links official getting-started material for each:

- **VS Code:** "Getting Started with VS Code."
- **GitHub in VS Code:** "Using Git Source Control in VS Code."
- **Dev Containers in VS Code:** "Developing inside a Container."
- **Docker:** students install Docker locally.

The instructor has already put a **dev container** definition in the class repo. Once Docker is installed and the Dev Containers extension is added to VS Code, the environment "*should* build automagically." The course commits to making sure everyone is set up in the first week.

**Codespaces** are the cloud option. A Codespace is a GitHub-hosted virtual machine running the same VS Code environment, so nothing has to be installed locally. The Day Two session uses this route: create a Codespace from the repo, then open it in local VS Code through the **Codespaces extension**.

## VS Code extensions

Extensions add capabilities to VS Code. The **Extensions tab** on the left shows what is already loaded in the container, and students can search for and add more. The Day Two lab has students add any new extension to the **Dev Container JSON file** (usually `devcontainer.json`) through the Dev Container menu, so the extension becomes part of the container's definition. Saving that file triggers a **rebuild** message, and students then rebuild the container.

## Package management with pip and requirements.txt

Over the semester the class keeps adding packages to the `requirements.txt` for its development container. The materials list four key commands or resources:

| Command | What it does |
|---|---|
| `pip freeze > requirements.txt` | Writes every installed package and its version into a requirements file |
| `pip install -r requirements.txt` | Installs all the packages listed in the file |
| `pip install --upgrade pip` | Upgrades pip itself to the latest version; probably needed only once |
| pip requirements-file documentation | The reference for the file format |

### Virtual environments

The Day Two tasks also have students create a Python **virtual environment** from a bash terminal, activate it, and then `pip freeze` into a requirements file. A virtual environment keeps a project's packages separate from other projects. The tasks only ask students to "type the command for creating a virtual environment" and do not spell it out. The usual form is `python -m venv <name>`, activated with `source <name>/bin/activate` on Linux/macOS or in a Codespace. This is general background, not something the course materials state.

## Keeping a fork and clone up to date

The instructor adds and changes files during the semester, so students need to **sync their fork and pull into their local clone every week**:

1. On GitHub, click the **Sync fork** button on the right side of the forked repo. This brings the fork up to date with the instructor's repo.
2. **Pull** those changes into the local clone.

GitHub's "Syncing a fork" documentation covers three ways to do this: the web UI, the terminal, and the GitHub client. A video in the course materials shows how to handle commits already made to an out-of-date fork.

### Avoiding merge errors: stash, pull, pop

Students often edit class files, for example by adding personal notes as comments, which the instructor calls "totally fine and probably even a smart thing to do." Their copy then differs from the instructor's, so the weekly pull can fail with a **merge error**. The fix is three commands, run in this order:

1. **`git stash`** saves your uncommitted local changes and reverts the working directory to the HEAD commit, which gives you a clean working directory.
2. **`git pull`** brings in the new files.
3. **`git stash pop`** puts your saved changes back on top of the new files.

Tip from the materials: it can be easier to **rename class files or create your own new files** so they never conflict with the instructor's. "There's always a way to fix git issues without losing anything." If you end up in a "merge error death spiral," your files can be saved and the branch rebased.

## Day Two setup lab (tasks as assigned)

1. **Clone the repo.** Find the `DayTwoStuff` repo on GitHub, clone it to your GitHub account, and create a new Codespace from it.
2. **Open the Codespace in local VS Code** with the Codespaces extension, and get familiar with the project layout.
3. **View data.** Open a new Python script, notebook or IPython file. Load any dataset, then find and install an extension that shows the data in a data viewer.
4. **Manage packages.** Install `plotly` from the terminal and note its version. Add plotly to the requirements file with terminal commands, then reload the requirements file.
5. **Manage extensions.** Find the data-viewer extension from step 3 in the Dev Container menu and add it to the dev container JSON file. Save the file and rebuild the container when prompted.
6. **Create a virtual environment.** Open a bash terminal, create the environment, and make sure it is activated.
7. **Create a requirements.txt** with `pip freeze`.

Small inconsistencies in the sources:
- Step 1 says "clone it to your GitHub account," but copying a repo into your own account is a *fork*. Cloning makes a local copy. The materials page uses the standard fork-then-clone sequence.
- The materials page describes building a local Docker dev container, while the syllabus and the Day Two lab use GitHub Codespaces. Both run the same dev container configuration, one on your own computer and one in the cloud.

## Why it matters later

- Labs are stand-alone Jupyter notebooks that students run in this environment.
- The final project expects work on GitHub and a free-standing website. Its suggested team roles include a **"Data Engineer"** who "sets up the environment (repo, data pipeline, requirements, codespace…)" (see [[course/final-project]]).

## Related topics

- [[course/course-overview]]: the tech stack in the syllabus, and where this fits in the SDS framework (Systems).
- [[course/python-basics]]: the Python skills used inside this environment.
- [[course/final-project]]: requires GitHub, a repo with code and data, and a reproducible environment.

## Sources

- `raw/course/MLBook/dev-environment/2026-05-21-class-materials-vscode-github.ipynb`: "Getting and Using Class Materials" (where materials live, fork/clone, VS Code + Dev Containers + GitHub, extensions, pip commands, syncing forks, git stash/pull/pop)
- `raw/course/MLBook/dev-environment/2026-05-21-day-two-repo-venv-setup.md`: "Day Two Stuff" hands-on setup tasks (Codespace, data viewer, plotly, devcontainer extensions, venv, requirements.txt)
