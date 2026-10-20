# DATA 201 Project 1 — Bank Marketing Classification

## Project overview

This project uses the **Bank Marketing** dataset to predict whether a customer will subscribe to a term deposit (`yes` or `no`). The aim is to compare classification models and consider the trade-off between contacting customers unlikely to subscribe and overlooking customers who might subscribe.

## Dataset

- **File:** `bank-full.csv`
- **Observations:** 45,211 customers
- **Columns:** 17 (16 predictors and the response variable)
- **Response variable:** `y` — whether a customer subscribed to a term deposit (`yes` / `no`)
- **Class distribution:** 39,922 `no` (88.3%) and 5,289 `yes` (11.7%)

The classes are imbalanced, so accuracy alone does not tell the full story.

## Methods

The project compares four classifiers:

1. Logistic Regression
2. Linear Discriminant Analysis (LDA)
3. Quadratic Discriminant Analysis (QDA)
4. K-Nearest Neighbors (KNN)

The data was divided into training and test sets. Predictors were preprocessed for modeling, including categorical encoding and standardization where appropriate. For KNN, 3-fold stratified cross-validation compared **K = 3, 7, 15, and 31** using ROC-AUC. **K = 31** scored highest among those tested (mean cross-validation ROC-AUC **0.7461**).

## Results

The following results are from the held-out test set, using the models' default classification decisions:

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.33% | 66.55% | 17.67% | 27.93% |
| LDA | 89.18% | 57.49% | 29.02% | **38.57%** |
| QDA | 88.26% | 49.70% | **31.10%** | 38.26% |
| KNN (K = 31) | 89.02% | **70.19%** | 10.68% | 18.54% |

The LDA model had a test **ROC-AUC of 0.7673**. Its confusion matrix included **307 true positives, 227 false positives, 751 false negatives, and 7,758 true negatives**.

## Main finding and recommendation

**LDA is the recommended model to pilot**, because it achieved the highest F1 score among the four classifiers, balancing precision and recall. QDA identified slightly more subscribers but also generated more false positives. KNN was more precise but missed substantially more actual subscribers.

This recommendation is preliminary. The bank should evaluate marketing-contact costs, customer value, and the decision threshold before deploying a model.

## Limitations

- **Class imbalance:** The large number of non-subscribers can make accuracy misleading.
- **Missed subscribers:** Even the recommended model misses many customers who would subscribe.
- **Real-world decision costs:** Contact costs and revenue per successful subscription were not available for optimizing the classification threshold.
- **Generalizability:** Performance should be checked on more recent customer campaigns before deployment.

## Repository contents

- **Google Colab notebook (`.ipynb`)** — code, model predictions, cross-validation graph, confusion matrices, evaluation metrics, and ROC curve.
- **`bank-full.csv`** — Bank Marketing dataset.
- **`README.md`** — project overview and main findings.

## How to run

1. Open the project notebook in **Google Colab**.
2. Upload `bank-full.csv` when prompted, or update the notebook's file path to match its location.
3. Run the notebook cells in order to reproduce the preprocessing, classification models, and evaluation figures.

**Python packages:** `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

---

**Course:** DATA 201 — Statistical Methods in Data Science  
**Assignment:** Project 1 — Classification in Practice
