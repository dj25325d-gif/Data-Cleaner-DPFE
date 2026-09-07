# 🧹 Data Cleanser — Patient Health Records

> **A practical Data Preprocessing & Feature Engineering project focused on missing-value imputation, outlier detection, outlier treatment, and preparation of a machine-learning-ready dataset.**

---

## 📌 Table of Contents

- [📖 Project Overview](#-project-overview)
- [🎯 Objective](#-objective)
- [🏥 Problem Statement](#-problem-statement)
- [📊 Dataset](#-dataset)
  - [Dataset Structure](#dataset-structure)
  - [Data Quality Challenges](#data-quality-challenges)
- [🔄 Project Workflow](#-project-workflow)
- [🧩 Missing Value Handling](#-missing-value-handling)
- [🚨 Outlier Detection & Treatment](#-outlier-detection--treatment)
- [🧼 Final Cleaning Strategy](#-final-cleaning-strategy)
- [🤖 Machine Learning Preparation](#-machine-learning-preparation)
- [📈 Validation & Quality Checks](#-validation--quality-checks)
- [📁 Project Structure](#-project-structure)
- [🛠️ Technologies & Libraries](#️-technologies--libraries)
- [▶️ How to Run the Project](#️-how-to-run-the-project)
- [📋 Project Deliverables](#-project-deliverables)
- [🎥 Project Video](#-project-video)
- [💡 Key Learning Outcomes](#-key-learning-outcomes)
- [🚀 Future Improvements](#-future-improvements)
- [👤 Author](#-author)

---

## 📖 Project Overview

**Data Cleanser** is a hands-on data preprocessing project built around a synthetic **Patient Health Records** dataset.

The project demonstrates how a data analyst can take a dataset containing:

- Missing values
- Categorical variables
- Numerical variables
- Extreme observations / outliers
- A binary target variable

and transform it into a **clean, validated dataset suitable for downstream machine-learning workflows**.

The project places particular emphasis on comparing different approaches rather than applying a single cleaning technique without evaluation.

---

## 🎯 Objective

The main objective of this project is to practice **Data Preprocessing and Feature Engineering**, with a strong focus on:

1. 🔎 Identifying missing values
2. 🧩 Applying multiple missing-value imputation techniques
3. 📊 Comparing imputation strategies
4. 🚨 Detecting numerical outliers
5. 🛠️ Applying multiple outlier-treatment techniques
6. 📈 Comparing the effect of different treatments
7. 🧼 Producing a final cleaned dataset
8. 🤖 Preparing the data for downstream machine-learning tasks
9. 📝 Documenting the complete data-cleaning process

---

## 🏥 Problem Statement

Imagine working as a **Data Analyst for a healthcare company**.

You receive patient health records containing inconsistent reporting, missing observations, and extreme measurements.

Your task is to clean and prepare the dataset while preserving the integrity of the target variable:

> **`disease_risk` — 0 = Low Risk, 1 = High Risk**

The cleaned dataset should be reliable enough to support future predictive modelling.

---

# 📊 Dataset

## Dataset Structure

The project uses the following patient-level variables:

| Field | Data Type | Description | Role |
|---|---|---|---|
| `patient_id` | String / Int | Unique identifier for each patient | Identifier |
| `age` | Integer | Patient age in years | Numerical feature |
| `gender` | Categorical | Male / Female | Categorical feature |
| `region` | Categorical | North / South / East / West | Categorical feature |
| `bmi` | Float | Body Mass Index | Numerical feature |
| `blood_pressure` | Float | Average systolic blood pressure (mmHg) | Numerical feature |
| `cholesterol` | Float | Cholesterol level (mg/dL) | Numerical feature |
| `glucose` | Float | Fasting glucose level (mg/dL) | Numerical feature |
| `disease_risk` | Binary Int | 0 = Low Risk, 1 = High Risk | 🎯 Target |

### 📌 Dataset Size

The project dataset contains:

- **10,000 patient records**
- **9 original columns**
- A binary target variable
- Intentionally introduced missing values
- Synthetic extreme values for outlier analysis

> ⚠️ **Note:** The dataset is synthetic and intended for educational purposes. It does not represent real patient records and should not be used for medical decision-making.

---

## 🧪 Data Quality Challenges

The raw dataset intentionally contains data-quality issues so that different preprocessing techniques can be demonstrated.

### Missing Values

Missing observations are present in:

- `age`
- `gender`
- `region`
- `bmi`
- `cholesterol`
- `glucose`

The target variable `disease_risk` is kept intact.

### Outliers

Synthetic extreme observations are present in:

- `bmi`
- `blood_pressure`
- `cholesterol`
- `glucose`

These allow multiple outlier-detection and treatment techniques to be compared.

---

# 🔄 Project Workflow

The overall project follows this workflow:

```text
                    ┌─────────────────────┐
                    │   📂 Raw Dataset    │
                    │ Patient Health Data │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 🔎 Initial Profiling│
                    │ Shape • Types •     │
                    │ Statistics • Quality│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ 🧩 Missing Data  │   │ 🚨 Outlier       │
          │ Analysis         │   │ Analysis         │
          └────────┬─────────┘   └────────┬─────────┘
                   │                      │
                   ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │ Mean / Median    │   │ Z-Score          │
          │ Most Frequent    │   │ IQR              │
          │ Random Sample    │   │ Percentile       │
          │ Missing Indicator│   │ Winsorization    │
          │ KNN              │   └────────┬─────────┘
          │ MICE             │            │
          └────────┬─────────┘            │
                   │                      │
                   └──────────┬───────────┘
                              ▼
                   ┌─────────────────────┐
                   │ 📊 Compare Results  │
                   │ Evaluate strategies │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ 🧼 Final Cleaning   │
                   │ Median + Mode +     │
                   │ Winsorization      │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ ✅ Quality Checks   │
                   │ Missing • Duplicates│
                   │ IDs • Target • Range│
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ 🤖 Model-Ready Data │
                   │ Encoding & Feature  │
                   │ Preparation         │
                   └─────────────────────┘
```

---

# 🧩 Missing Value Handling

The project does not rely on a single imputation method.

Instead, several approaches are implemented and compared.

## Techniques Used

| Technique | Variable Type | Purpose |
|---|---|---|
| 🟦 Mean Imputation | Numerical | Replace missing values with the mean |
| 🟩 Median Imputation | Numerical | Replace missing values with the median |
| 🟨 Most Frequent | Categorical | Replace missing categories with the mode |
| 🟧 Missing Indicator | Numerical / Categorical | Preserve information about missingness |
| 🟪 Random Sample Imputation | Numerical | Sample observed values to replace missing observations |
| 🟥 KNN Imputer | Numerical | Multivariate nearest-neighbour imputation |
| 🟫 MICE-style / Iterative Imputation | Numerical | Model-based iterative multivariate imputation |

### Example

For numerical variables such as `bmi`, both mean and median imputation are evaluated.

For categorical variables such as `gender` and `region`, most-frequent imputation is demonstrated.

The KNN workflow also considers feature scaling before distance-based imputation.

---

# 🚨 Outlier Detection & Treatment

Four approaches are demonstrated.

## 1. 📐 Z-Score Method

The Z-score approach identifies observations that are unusually far from the mean.

Used for identifying extreme values in:

- `bmi`
- `blood_pressure`
- `cholesterol`
- `glucose`

---

## 2. 📦 IQR Method

The Interquartile Range method identifies observations outside the lower and upper bounds calculated from Q1, Q3 and the IQR.

This approach is useful because it is less dependent on the assumption of normally distributed data.

---

## 3. 📊 Percentile Method

The percentile approach identifies extreme observations using the lower and upper percentile boundaries.

In this project, the **1st and 99th percentiles** are used as the treatment boundaries.

---

## 4. ✂️ Winsorization

Instead of deleting extreme observations, Winsorization caps values at selected percentile limits.

In the final cleaning strategy:

```text
Lower limit → 1st percentile
Upper limit → 99th percentile
```

This allows the project to retain all patient records while reducing the influence of extreme synthetic observations.

---

# 🧼 Final Cleaning Strategy

After comparing the different approaches, the final cleaned dataset follows this strategy:

```text
Numerical Missing Values
        ↓
Median Imputation
        ↓
Categorical Missing Values
        ↓
Most-Frequent Imputation
        ↓
Extreme Numerical Values
        ↓
1st–99th Percentile Winsorization
        ↓
Final Validation
```

### Why this approach?

The final workflow prioritizes:

- 🛡️ Retaining all records
- 📉 Reducing the influence of extreme values
- 🧮 Robust treatment of numerical missing values
- 🏷️ Sensible treatment of categorical missing values
- 🤖 Preparing data for downstream modelling

---

# 🤖 Machine Learning Preparation

The project also demonstrates preparation of the cleaned data for future machine-learning workflows.

Categorical variables are encoded using **One-Hot Encoding**.

For example:

```text
gender
   ↓
gender_Female
gender_Male
```

and:

```text
region
   ↓
region_East
region_North
region_South
region_West
```

The target variable remains:

```text
disease_risk
```

with:

```text
0 → Low Risk
1 → High Risk
```

### ⚠️ Identifier Handling

`patient_id` is retained in the cleaned dataset for record identification and traceability, but it should **not be used as a predictive feature** when constructing the machine-learning feature matrix.

---

# 📈 Validation & Quality Checks

A major part of the project is validating the final result rather than assuming the cleaning process worked correctly.

The final dataset is checked for:

| Quality Check | Expected Result |
|---|---|
| 🔍 Missing values | 0 |
| 🆔 Duplicate patient IDs | 0 |
| 🧾 Duplicate rows | 0 |
| 🎯 Target missing values | 0 |
| 🔢 Numerical validity | Valid ranges |
| 📊 Outlier treatment | Verified |
| 🏷️ Categorical encoding | Verified |
| 📐 Dataset dimensions | Preserved where appropriate |

The project also compares the dataset **before and after preprocessing** to understand the impact of each treatment.

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
📦 Data-Cleanser/
│
├── 📓 Data_Cleanser.ipynb
│
├── 📂 data/
│   ├── patient_health_records_raw.csv
│   ├── patient_health_records_cleaned.csv
│   └── patient_health_records_model_ready.csv
│
├── 📄 Data_Cleanser_Project.pdf
│
├── 🎥 project_video_link.txt
│
└── 📖 README.md
```

> 💡 If your submission instructions require the CSV files to remain in the repository root, keep them there and adjust the paths in the notebook accordingly.

---

# 🛠️ Technologies & Libraries

The project is implemented in **Python** using Jupyter Notebook.

### Core Technologies

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 📓 Jupyter Notebook | Interactive analysis |
| 🐼 Pandas | Data manipulation |
| 🔢 NumPy | Numerical operations |
| 📊 Matplotlib | Data visualization |
| 🎨 Seaborn | Statistical visualization |
| 🤖 Scikit-learn | Imputation and preprocessing |

### Key Scikit-learn Components

The notebook demonstrates tools including:

```python
SimpleImputer
KNNImputer
IterativeImputer
StandardScaler
```

---

# ▶️ How to Run the Project

## 1️⃣ Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2️⃣ Navigate to the project directory

```bash
cd Data-Cleanser
```

## 3️⃣ Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5️⃣ Open

```text
Data_Cleanser.ipynb
```

## 6️⃣ Run the notebook

For the cleanest reproduction:

> **Kernel → Restart Kernel and Run All Cells**

Make sure the input CSV is available at the path expected by the notebook.

---

# 📋 Project Deliverables

The completed project includes:

- [x] 📂 Raw patient dataset
- [x] 🔎 Initial data profiling
- [x] 🧩 Missing-value analysis
- [x] 🧮 Mean imputation
- [x] 📊 Median imputation
- [x] 🏷️ Most-frequent categorical imputation
- [x] 🚩 Missing indicators
- [x] 🎲 Random sample imputation
- [x] 👥 KNN imputation
- [x] 🔄 MICE-style / Iterative imputation
- [x] 📐 Z-score outlier detection
- [x] 📦 IQR outlier detection
- [x] 📊 Percentile outlier detection
- [x] ✂️ Winsorization
- [x] 📈 Before/after comparisons
- [x] 🧼 Final cleaned dataset
- [x] 🤖 Model-ready dataset
- [x] ✅ Final quality validation
- [x] 📝 Project documentation

---

# 🎥 Project Video

A recorded walkthrough is included as part of the project submission.

The video demonstrates:

- 👤 Face/webcam visibility
- 💻 Screen recording
- 📓 Jupyter Notebook walkthrough
- 🧩 Missing-value techniques
- 🚨 Outlier techniques
- 🧼 Final cleaning strategy
- 📊 Results and interpretation

### 🔗 Video Link

**[▶️ Watch the Project Walkthrough](YOUR_VIDEO_LINK_HERE)**

> Replace `YOUR_VIDEO_LINK_HERE` with the final Google Drive / YouTube / approved video URL before submitting the GitHub repository.

---

# 💡 Key Learning Outcomes

Through this project, the following practical skills are demonstrated:

### 🧩 Missing Data

- Understanding missing-data patterns
- Calculating missing-value percentages
- Applying univariate imputation
- Applying multivariate imputation
- Preserving missingness information with indicators
- Comparing imputation strategies

### 🚨 Outlier Analysis

- Understanding statistical outliers
- Applying Z-score detection
- Applying IQR detection
- Applying percentile-based detection
- Comparing detection methods
- Removing versus capping extreme observations
- Applying Winsorization

### 🧼 Data Cleaning

- Building reproducible preprocessing workflows
- Validating cleaned data
- Preserving identifiers
- Protecting the target variable
- Comparing raw and processed datasets

### 🤖 Machine Learning Preparation

- Encoding categorical variables
- Separating identifiers from predictive features
- Preparing structured numerical data
- Producing a downstream modelling dataset

---

# 🚀 Future Improvements

This project can be extended beyond the current assignment by adding:

- 🤖 Classification models such as Logistic Regression, Decision Trees and Random Forest
- 📊 Model evaluation using accuracy, precision, recall and F1-score
- 📈 ROC-AUC analysis
- ⚖️ Class-imbalance analysis
- 🔬 Feature importance analysis
- 🧪 Cross-validation
- 🔧 Hyperparameter tuning
- 📦 Reproducible preprocessing pipelines
- 🧠 Explainable AI techniques
- 📊 Interactive dashboards

These extensions would turn the current preprocessing project into a complete **end-to-end healthcare risk prediction workflow**.

---

# 🏆 Project Summary

This project demonstrates an end-to-end approach to **data quality assessment and preprocessing**.

Rather than simply removing problematic records, the project investigates multiple strategies, compares their results, and selects an appropriate final cleaning workflow.

The final objective is to transform:

```text
❌ Raw + Missing + Extreme Values
              ↓
       🔎 Data Profiling
              ↓
      🧩 Missing Imputation
              ↓
       🚨 Outlier Analysis
              ↓
       ✂️ Outlier Treatment
              ↓
        🧼 Data Cleaning
              ↓
        ✅ Validation
              ↓
       🤖 Model Preparation
```

into a dataset that is:

> **Clean • Validated • Documented • Reproducible • Ready for downstream analysis**

---

# 👤 Author

**Dushyant V**

---

## ⭐ If you found this project useful

Feel free to explore the notebook, review the preprocessing decisions, and build upon the workflow.

**Data quality is the foundation of reliable analysis and machine learning. 🚀**
