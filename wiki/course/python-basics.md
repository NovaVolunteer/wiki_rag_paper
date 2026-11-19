# Python Basics: Functions and pandas

Week 2 of DS-3021 (the notebooks' Colab links point to the course repo folder `02_function_basics` / `02_R_function_basics`). This is the course's coding refresher before any machine learning. It has two parts: writing your own Python **functions**, and the **pandas** functions used for nearly all data manipulation in later weeks. The material was first written in R (tidyverse verbs like `filter()`, `mutate()`, `summarize()`) and converted to Python, so the R→Python mapping runs through the whole topic. Next steps are [[course/ml-foundations]] (week 4) and data preparation in [[course/data-prep]].

## Functions

### How the lecture frames functions

The lecture notebook ("Fun with functions and pandas!") starts by saying that **Python is an object-oriented language**, so functions are used to create objects or modify an existing object. A function is a block of code that only runs when it is called. You can pass it data (parameters), and it can return data as a result. Functions are first-class objects: you can assign them to variables, store them in lists, pass them as arguments to other functions, define them inside other functions, and return them from functions.

The notebook gives three rules for good functions:
- **Functions should be "pure"**: calling one again with the same inputs gives the same result. The notebook's counter-example is `sys.time()`, which is not pure.
- **No side effects**: running a function shouldn't change global variables.
- **Small and single-purpose**: a function should do one thing, though it can take many arguments.

### Function features with examples

The notebook lists seven features and gives one example of each:

| Feature | Example in the notebook |
|---|---|
| **Definition** with `def` plus a name and parameters in parentheses | `def greet(name): return f"Hello, {name}!"` |
| **Docstring**: a string literal right after the definition line that documents the function | Triple-quoted docstring inside `greet` |
| **Default parameters**, used when no argument is given | `def power(base, exponent=2)`. `power(3)` gives 9 and `power(3, 3)` gives 27 |
| **Variable-length arguments**: `*args` (positional) and `**kwargs` (keyword) | `sum_all(1, 2, 3)` gives 6. `print_info(name="Alice", age=30)` loops over `kwargs.items()` |
| **Lambda (anonymous) functions** | `square = lambda x: x * x`, so `square(5)` gives 25 |
| **Higher-order functions**: take or return functions | `apply_function(square, 4)` gives 16 |
| **Closures**: an inner function keeps variables from the enclosing scope | `outer_function("Hello, World!")` returns `inner_function`, which prints the captured message when called |

### Building functions in class

- **`my_mean`**: the notebook's first hand-built function computes `sum(x)` and a length, then returns their ratio. As written it uses R-style names (`length(x)` instead of Python's `len(x)`, and returns `Suml/N` when the variable is `Sum_val`), so it will not run as-is. Students are told to pass it a small list and to try calling the internal `Sum` and `N` variables from outside ("does this work?"). This exercise is about local scope.
- **Function factory**: `power1(exp)` defines and returns an inner `action(x)` that computes `x**exp`. Then `square = power1(2)` and `cube = power1(3)`, so `print(square(3), cube(3))` prints 9 and 27. The notebook calls a function that returns a function a higher-order function.
- **Quick exercise**: "Create a function that computes the range of a variable and then, for no good reason, adds 100 and divides by 10." Students write the steps as **pseudocode first**, then write the function.

### The Copilot example

`2024-10-29-copilot-example.py` is a short VS Code cell script (`#%%` cell markers). It shows the same exercises written by an AI coding assistant from plain-English comments. Each comment works as a prompt and the function under it is the generated result:
- `print("Hello World")`
- `# create a function that takes a list of numbers and returns the sum of the numbers` produces `def sum_list(x): return sum(x)`
- The range exercise from the lecture produces `def range_plus_100_div_10(x): return (max(x)-min(x)+100)/10`. It is called on `x = [1,2,3,4,5,6,7,8,9,10]`, so the result is (10 − 1 + 100)/10 = 10.9.

## pandas: the five key functions

Both notebooks are organized around five pandas DataFrame methods. The class notebook says these solve "the vast majority of your data-manipulation challenges" (`pd` is pandas and `df` is a DataFrame):

1. **`df.query()`**: select observations (rows) by their values, using a Boolean expression
2. **`df.sort_values()`**: reorder the rows
3. **`df.loc[]`**: pick variables by their row or column names and/or conditions
4. **`df.assign()`**: create new variables from functions of existing variables
5. **`df.describe()`**: collapse many values into a single summary

**Bonus: `df.groupby()`** can be combined with any of the five to make the operation run group by group instead of on the whole dataset. Grouping "happens in the background" and returns a grouped object, so you sometimes have to reorganize your function calls around it.

### Worked examples on the `mpg` dataset

R has built-in datasets and Python doesn't. To use the same data, the lecture installs `pydataset` (`pip install pydataset`, which is not part of the conda distribution), then runs `from pydataset import data` and `mpg_data = data('mpg')`.

- **`.query()`**: `mpg_data.query('cyl==4')` keeps 4-cylinder cars. Practice: keep rows where `cty > 20` and the manufacturer is `'volkswagen'`.
- **`.sort_values()`**: the full signature shown is `df.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)`. The parameters that matter for the course are `by` (a column name or list of names), `ascending` (`False` reverses the order), `ignore_index` (`True` renumbers rows 0 to n−1 instead of keeping the original indices), and `na_position` (NaNs go last by default, `'first'` moves them to the top). Examples: `sort_values(by='year')`, then `sort_values(by=['year', 'hwy'], ascending=False)` to find the best highway mileage in the most recent years, then the same sort with `ignore_index=True` for a clean top-5 list. Practice: sort by model and hwy.
- **`.loc[]` and slicing**: keep columns with `df[['col1', 'col2']]` or `df.loc[:, 'col1':'col3']`. The `:` in the row position means "all rows". The lecture explains the `col3` by saying Python slicing is upper-bound exclusive (see the conflict note below). To remove a few columns, use `df.drop([cols], axis=1)`, where `axis=1` drops columns instead of rows.
- **Selecting by name pattern (`weather.csv`)**: R can use `startswith()` inside `select()`. Python can't, so the lecture shows two alternatives. The long way builds a list comprehension, `[col for col in weather.columns if col.startswith('d')]`, and indexes with it. The short way uses a regex: `weather.filter(regex='^d')`. Practice: select the first 7 days of data.
- **`.assign()`**: converts mileage to km/l with `mpg_data.assign(cty=mpg_data.cty * 1.6 / 3.8, hwy=mpg_data.hwy * 1.6/3.8)`. Inside `.assign()` you have to say which DataFrame the column comes from (`mpg_data['cty']` or `mpg_data.cty`). R's `mutate()` works on the current data frame automatically. Assigning to a name that doesn't exist yet creates a new column.
- **`.describe()`**: summary statistics, shown on `mpg_data` and on the converted `km_conv`. It only shows numeric columns. For all columns and their types, use **`.info()`, the equivalent of R's `str()`**.

### pandas data types

The lecture includes a table of pandas' basic data types (an embedded image, `pandas_datatypes.png`). Its point is that you need to know your column types so you can fix problems, for example values stored as strings when you need to calculate with them. In the class notebook, the R concept of a **factor** with "levels" becomes a pandas **categorical** variable: `admit_2['rank'] = admit_2['rank'].astype("category")`, and you can check the result with `.info()`.

### Other Python basics covered

- Working directory: `import os; os.getcwd()`
- Loading data with a relative path: `pd.read_csv("../data/bank.csv")`, where `..` goes up one level to the repo's data folder. The notebook suggests `read_excel` for Excel files and cloning the class repo for easy data access.
- Basic types: integer, float, string, list, and dictionary, with example operations: addition, `string.upper()`, `len(my_list)`, and `my_dict["name"]`
- Control flow: `if`, `for`, and `while` (a counter loop from 0 to 4)
- List comprehensions: `[x * x for x in my_list]` gives `[1, 4, 9, 16, 25]`
- Classes and objects: a `Person` class with `__init__(self, name, age)` and a `greet()` method. Classes define new types of objects, and objects are instances of classes.

## The class notebook example: graduate admissions (Pandas Practice Lab)

The "Class II" notebook works one problem end to end using the method students must follow in the lab: **question → pseudocode → single-line pandas transactions → chain the lines into a function**.

- **Data**: `LogReg.csv`, a dataset of students admitted to a graduate Data Science program, with columns `admit`, `gpa`, and `rank`. The instructor says "I actually don't remember where this data came from."
- **Question**: "Are there differences in the admitted number of students on average by GPA and school rank?" The suggested approach is to find the function calls by starting at the end of the question and working backwards.
- **Pseudocode**: group by school rank, calculate mean GPA, count total admitted, calculate % admitted, and sort in descending order by mean GPA.
- **Steps**:
  - Convert `rank` to a category.
  - `x1 = admit_2.groupby(['rank'])`
  - Note the catch: **you can't use `.assign()` on a grouped DataFrame**. The notebook shows two workarounds:
    1. Assign onto the original DataFrame (`admit_2.assign(admitted=x1.admit.sum(), tot_students=x1.admit.count(), meanGPA=x1.gpa.mean())`). This keeps every old column and fills the result with NaNs, so you then have to select columns (`x2[['admitted','tot_students','meanGPA']]`) and either slice rows `x2[1:5]` (ranks 1 to 4) or call `.dropna()`. The notebook calls `.dropna()` a simplified version of `.drop(df.isna())`.
    2. Start with an empty `pd.DataFrame()` and assign the new columns to it. The notebook says this second method "worked *much* better."
  - Add the admit rate with a lambda: `.assign(ave_admit = lambda x: x['admitted'] / x['tot_students'])`.
  - Sort with `sort_values(by='meanGPA', ascending=False)`.
- **Function**: `avg_admission(x)` combines everything into one chained expression, `.assign(...).assign(...).sort_values(...)`. The notebook says this sequential calling is called **chaining**. You can split it up for clarity, but chaining is compact.

## R → Python differences (week 2 conversion notes)

| R | Python | Note |
|---|---|---|
| `filter()` | `query()` | Select observations by values or conditions. Equivalent |
| `arrange()` | `sort_values()` | `sort_values(by='column', ascending=True)`. `True` is the default |
| `select()` | `loc[]` | `[:, :]` is `[row, column]`. There are several other ways to select (see the lecture notebook) |
| `mutate()` | `assign()` | Columns built from existing values must be referenced as `df.col` inside `.assign()`. `mutate()` assumes the current data frame |
| `summarize()` | `describe()` | Equivalent |
| `group_by()` | `groupby()` | Small difference. Some chaining has to be done before or after the grouping, so check the docs |
| `str()` | `info()` | Equivalent |

Broader differences listed in the conversion doc:
- **Paradigm**: R is function-oriented and Python is object-oriented. The doc calls them complementary and says the two are often used interchangeably.
- **Built-in datasets**: use a module to get R's datasets, then `data('mpg')`.
- **Reading files**: R uses `rio` or `readr`. Python just uses pandas.
- **`startswith()`**: works inside R's `select()`. In Python you determine the columns beforehand or use regex.
- **Data types**: R factors are pandas category types.
- **Factorization pitfall (lab)**: the R file factorized both `admit` and `rank`. Doing the same in Python "resulted in calculation issues later," which is why the notebook only converts `rank`. The doc's advice: "Make sure your data types are working for you in the end!"
- **Chaining**: R uses the pipe `%>%` (written as `>%>` in the doc). Python calls methods in sequence, as in `df.assign().filter().sort_by()`.
- **Lambda functions** make small functions easier to define in Python.
- **Slicing**: the Python notebook explains and gives examples of slicing.

> ⚠️ Possible conflict: the week-2 conversion doc says to install and import **`pydatasets`** (`pip install pydatasets`, `from pydatasets import data`). The lecture notebook uses **`pydataset`** (`pip install pydataset`, `from pydataset import data`). The notebook's spelling matches its working code. Students should follow the notebook.

*General clarification, not from the sources:* the lecture's "upper-bound exclusive" rule holds for position-based slicing such as `x2[1:5]`. Label-based `.loc` slices in pandas include the end label, so check the output when you slice columns by name.

## Labs / assignments

- **Pandas Practice Lab** (the Class II notebook): use any dataset in the class repo, or the suggested `nfl2008_fga.csv` (NFL 2008 field-goal attempts) or the UCI red wine quality data, or pick two of your own. For each of two datasets, write: (1) a question, (2) pseudocode that answers it, (3) single-line transactions using pandas functions, and (4) the lines piped together into a function. Group work is fine for coding help, but each student needs their own questions and results. Comment your code. You don't need to explore alternative methods as deeply as the instructor's example does.
- **In-lecture exercises**: the range-plus-100-divided-by-10 function (pseudocode first), and "now you try" prompts for each pandas method.

## Related topics

- [[course/dev-environment]]: setting up Python, VS Code, and Colab/Codespaces, which these notebooks assume (the class notebook loads data from a `/workspaces/...` Codespaces path)
- [[course/ml-foundations]]: week 4, where these pandas skills (`groupby().describe()`, `.drop()`) are used for the first ML example
- [[course/data-prep]]: categorical conversion, `.astype("category")`, one-hot encoding, and other cleaning that builds on the data-types material here
- [[course/course-overview]]: where this week fits in the semester

## Sources

- `raw/course/DS-3021/python-basics/2024-10-29-copilot-example.py`: AI-assistant (Copilot) function-writing example script
- `raw/course/DS-3021/python-basics/2024-10-29-week02-r-to-python-changes.docx`: week 2 R→Python conversion notes
- `raw/course/MLBook/python-basics/2026-05-21-python-class-II-notebook.ipynb`: Class II Pandas Practice Lab notebook (graduate admissions example plus assignment)
- `raw/course/MLBook/python-basics/2026-05-21-python-lecture-notebook.ipynb`: lecture notebook "Fun with functions and pandas!"
