# Credit-risk model validation: can we trust the probabilities and the explanations?

A bank's model gives each loan applicant a probability of default. A model can rank applicants well and still give probabilities that are wrong, or explanations that change when the data or the feature set changes. This project checks all three on real loan data.

Research question: can a more accurate machine-learning model also produce reliable probabilities and explanations?

Short answer for this dataset: mostly yes. XGBoost is slightly better than logistic regression, and both give honest probabilities. The risks come from common practices. Class weighting breaks the probabilities, dropping a sensitive column moves errors onto groups, and single coefficients depend on which correlated columns are in the model.

## Key findings

- XGBoost beats logistic regression: AUC +0.0077 (95% paired bootstrap interval 0.0048 to 0.0105) and log-loss -0.0027 (-0.0034 to -0.0019).
- Both models are well calibrated. Their expected calibration error (0.0026 and 0.0034) is about what chance alone gives with this many applicants (mean 0.0025, 95th percentile 0.0037), so that metric could not separate the models.
- Class weighting (defaulters weighted n0/n1 = 11.39 times) leaves AUC unchanged at 0.7537 but raises the average predicted default risk from 8% to 42% (log-loss 0.590, ECE 0.340).
- The cause is an exact shift of the model's score by log(n0/n1) = 2.43. Platt scaling, fitted on a separate calibration set, found an intercept of -2.4316 against -2.4325 from the algebra. Simply subtracting 2.43 repairs the probabilities as well as Platt scaling does.
- Removing gender costs about 0.0024 AUC. Without it, women's default risk is over-predicted by about 0.6 percentage points and men's under-predicted by about 1.2, both outside sampling noise. With gender in the model, no group gap was larger than noise.
- The top of the explanation is stable: `EXT_SOURCE_2` and `EXT_SOURCE_3` were the top two XGBoost variables in all 15 retrained models. The middle moves (`AMT_CREDIT` ranged from 6th to 11th).
- `AMT_CREDIT` and `AMT_GOODS_PRICE` have correlation 0.988. Dropping one changes the other's logistic-regression coefficient from 1.27 to 0.23, so single coefficients cannot be read on their own.

## Data

Home Credit Default Risk (Kaggle). Two tables:

- `application_train`: 307,511 applicants, 8.07% defaulted.
- `bureau`: 1,716,428 past credit records for 305,811 applicants.

Every cleaning decision is logged with its reason in `decisions.md`. The main ones:

- `DAYS_EMPLOYED = 365243` is a placeholder. It appears in 55,374 rows, and 55,352 of them are pensioners. I replaced it with a missing value and added a flag column. Default rate is 5.4% for these rows against 8.7% for the rest, which is mostly a pensioner effect.
- 44,020 applicants have no bureau record, and their default rate is higher (10.1% against 7.7%), so I kept a `HAS_BUREAU` flag.
- The bureau table has several rows per applicant, so I collapsed it to one row per applicant before joining, and checked that the join did not duplicate anyone.
- I dropped 41 columns that were more than 50% missing. That left 92 columns, or 197 features after encoding.
- Stratified 60/20/20 split: train 184,506, calibration 61,502, test 61,503, `random_state=42`. Preprocessing was fitted on the training set only. The test set is my own split of Kaggle's training file, because Kaggle's real test file has no labels.

## Method

1. Baselines: logistic regression (scikit-learn) and XGBoost, without class weighting so the probabilities stay honest.
2. Logistic regression built from scratch in NumPy (`src/logreg.py`), fitted with gradient descent and with Newton's method, and checked against scikit-learn.
3. Calibration checks: reliability diagrams, log-loss, Brier score, expected calibration error.
4. A class-weighting experiment, repaired three ways: Platt scaling, isotonic regression and a direct shift of the score.
5. Gender and subgroup checks (gender, pensioners, applicants with no bureau record).
6. Bootstrap confidence intervals: 1,000 resamples of the test set, with paired comparisons between models.
7. Explanation stability: 15 XGBoost models retrained on resampled training data, with TreeSHAP values summed per original variable, and logistic-regression coefficients across the same kind of resamples.

## Results

Test set, 95% bootstrap intervals in brackets.

| Model | AUC | Log-loss | Brier |
|---|---|---|---|
| Predict the average for everyone | 0.5000 | 0.2805 | 0.0742 |
| Logistic regression | 0.7535 [0.7466, 0.7601] | 0.2476 [0.2427, 0.2528] | 0.0682 [0.0666, 0.0698] |
| XGBoost | 0.7613 [0.7546, 0.7680] | 0.2450 [0.2403, 0.2500] | 0.0675 [0.0659, 0.0692] |

Class weighting and its repairs:

| Model | AUC | Log-loss | Brier | ECE | Mean predicted risk |
|---|---|---|---|---|---|
| Plain logistic regression | 0.7535 | 0.2476 | 0.0682 | 0.0026 | 8.05% |
| Weighted, raw | 0.7537 | 0.5901 | 0.2020 | 0.3403 | 42.1% |
| Weighted + Platt scaling | 0.7537 | 0.2477 | 0.0682 | 0.0030 | 8.02% |
| Weighted + subtract log(n0/n1) | 0.7537 | 0.2477 | 0.0682 | 0.0021 | 8.06% |
| Weighted + isotonic regression | 0.7533 | 0.2481 | 0.0683 | 0.0022 | 8.02% |

True default rate on the test set: 8.07%. Platt and the plain subtraction cannot be told apart. Isotonic regression is slightly worse than Platt on log-loss (by 0.0003, interval 0.0001 to 0.0005).

Gender and subgroups:

- Predicted minus observed default rate, in percentage points, with gender in the model: all gaps within sampling noise (largest 0.29).
- Without gender: logistic regression women +0.58 [0.31, 0.81], men -1.18 [-1.58, -0.78]. XGBoost women +0.60 [0.34, 0.84], men -1.24 [-1.64, -0.84].
- AUC is lower where there is less information: about 0.72 to 0.74 for pensioners and for applicants with no bureau record, against 0.76 for applicants with one. Probabilities stay close to the observed rates in every group.

Explanation stability:

- XGBoost, 15 retrained models: `EXT_SOURCE_2` rank 1 and `EXT_SOURCE_3` rank 2 in all 15. `AMT_GOODS_PRICE` ranks 3 to 4, `CODE_GENDER` 3 to 5, `EMPLOYED_YEARS` 4 to 6. The same five variables were in the top five in 93 to 100% of the models.
- Logistic regression coefficients across resamples: `AMT_CREDIT` 1.20 (standard deviation 0.05), `AMT_GOODS_PRICE` -1.06 (0.05), and their sum 0.14 (0.03). Stable under resampling, but the `AMT_CREDIT` coefficient falls from 1.27 to 0.23 when `AMT_GOODS_PRICE` is dropped.

## The maths

Handwritten derivations are in `derivations/logistic_regression_derivations.pdf`.

For one applicant, with z = w·x + b, probability p = 1 / (1 + e^(-z)), outcome y in {0, 1} and loss L = -[y ln p + (1 - y) ln(1 - p)]:

- dp/dz = p(1 - p)
- dL/dz = p - y, because the p and (1 - p) in dL/dp cancel against dp/dz
- Gradient: dL/dw = (p - y)x and dL/db = p - y
- Hessian: d²L/dw² = p(1 - p)x², d²L/dw db = p(1 - p)x, d²L/db² = p(1 - p)

Over n applicants, with an L2 penalty lambda = 1/n on all coefficients except the intercept (the same objective as scikit-learn with C = 1):

- Gradient: (1/n) sum of (p_i - y_i) x_i, plus lambda times theta
- Hessian: (1/n) sum of p_i(1 - p_i) x_i x_i^T, plus lambda times the identity (intercept excluded)

Checks:

- On a 5-student toy problem, the gradient formula matches a numerical nudge to six decimals at three different points, and the Hessian formula was checked by hand at w = 0, b = 0 (result [[0.5, 0], [0, 0.25]]). The from-scratch fit gives w = 1.0904 and b = -0.6227, and scikit-learn gives 1.0901 and -0.6229.
- The one-hot columns of each categorical variable add up to 1 for every row, so the Hessian without a penalty is singular (smallest eigenvalue -5e-17). The penalty makes it invertible (1.2e-6).
- Newton's method converged in about a dozen steps (36 seconds). Gradient descent was still 5e-4 away from the optimum after 2,000 steps (164 seconds).
- scikit-learn's default solver stops with a slope of about 3.4e-4. Run to a tight tolerance (newton-cholesky, tol 1e-12), it gives exactly the same coefficients and predictions as my Newton fit (test AUC 0.75355, log-loss 0.24764). The early-stopped version has very different intercept and category coefficients but almost the same predictions, so coefficients on redundant columns should not be interpreted.

Why class weighting shifts the score. Weighting defaulters by w = n0/n1 acts like repeating each defaulter w times. For applicants with true default probability p, the odds the weighted model learns are w times the true odds p/(1 - p). Taking logs, the score moves up by log(w) = 2.43, and subtracting it recovers the original probabilities when the model form is correct. The fitted relationship between the weighted and plain scores was 1.011 times the plain score plus 2.457, close to the predicted slope 1 and shift 2.43.

## Limitations

- The bootstrap measures luck in which applicants fall in the test set. It does not include variation from retraining on different training data, or from the calibration set.
- Platt scaling and isotonic regression were each fitted once on the calibration set.
- The stability check uses 15 retrained models, a smaller XGBoost (200 trees against 400), and 3,000 test rows for the SHAP values, so the rank ranges are probably narrower than the true spread.
- ECE is dominated by sampling noise at this sample size, so I did not use it to rank models.
- One dataset from one lender. I did not test whether the findings hold as the population changes over time.
- Gender is used in some models only to study its effect. Using it in real credit decisions is restricted in many places.

## How I used AI

I used an AI assistant (Claude) to help plan the project, write and debug code, and explain concepts. I derived the gradient and Hessian by hand and checked them numerically. I checked results against scikit-learn, an independent calculation or a numerical test wherever one was available, and kept a log of every decision in `decisions.md`.

## Reproducing

1. Download `application_train.csv` and `bureau.csv` from the Home Credit Default Risk competition on Kaggle and put them in `data/`. The data is not included in this repository.
2. `pip install pandas numpy scikit-learn xgboost scipy matplotlib pyarrow`
3. Run the notebooks in order. Random seeds: split 42, bootstrap 42, stability check 7.

## Repository

- `notebooks/01_data_audit.ipynb`: data audit, cleaning, split, baselines, calibration, bootstrap and stability checks
- `notebooks/02_gradient_descent_five_students.ipynb`: the toy example and the gradient checks
- `src/logreg.py`: logistic regression with Newton's method
- `derivations/logistic_regression_derivations.pdf`: handwritten maths
- `decisions.md`: every cleaning and modelling decision, with the reason
- Plots: `reliability_baseline.png`, `reliability_balanced.png`, `newton_vs_gd.png`, `weighting_shift.png`, `bootstrap_auc_diff.png`, `explanation_stability.png`
