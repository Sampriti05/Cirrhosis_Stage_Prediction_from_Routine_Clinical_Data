# Cirrhosis Stage Prediction from Routine Clinical Data

Machine learning workflow to predict the **histologic stage (1-4)** of primary biliary cirrhosis (PBC) from routine clinical and biochemical variables, with a focus on finding the **smallest useful feature set** and keeping the results interpretable.

## Purpose

Cirrhosis severity cannot be captured by a single biomarker, and neighbouring stages look clinically similar. This project asks:

- Can routine variables (bilirubin, albumin, symptoms, etc.) predict cirrhosis stage?
- What is the smallest feature subset that keeps cross-validated performance high?
- Which features drive the predictions, and do they make clinical sense?

## Dataset

- [Cirrhosis Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset) (fedesoriano, Kaggle), from the Mayo Clinic PBC trial (1974-1984)
- 418 patients in the raw file, 312 in the randomised trial cohort used for modelling
- Predictors: age, sex, drug, symptoms (ascites, hepatomegaly, spiders, edema) and lab values (bilirubin, cholesterol, albumin, copper, alkaline phosphatase, SGOT, triglycerides, platelets, prothrombin)
- Target: `Stage` (1-4). Stage 1 is only 5.1% of patients, so the data is heavily imbalanced

## Repo Contents

- `Optimization_with_AI_code.ipynb`: full code (cleaning, EDA, feature selection, models, SHAP, clustering)
- `Optimization_with_AI_report.pdf`: written project report

## Steps

1. **Clean data.** Removed 106 non-trial patients whose trial-only variables were all missing (418 to 312 rows). Median-imputed the few remaining gaps. Dropped `ID`, `N_Days` and `Status` to prevent data leakage.
2. **Explore data.** Class imbalance, correlations and boxplots by stage. Hepatomegaly, albumin and spiders correlate most with stage.
3. **Select features.** Compared Lasso (16 features, Macro F1 0.387) with Beam Search (9 features, Macro F1 0.407). Beam Search won with a smaller set: Albumin, SGOT, Hepatomegaly, Copper, Edema, Triglycerides, Age, Bilirubin, Ascites.
4. **Compare models.** Logistic Regression, Random Forest and XGBoost on the 9 features, using stratified 5-fold cross-validation and **Macro F1** (to handle class imbalance).
5. **Tune.** GridSearchCV on Random Forest.
6. **Interpret.** SHAP for global importance and individual patient explanations.
7. **Cluster.** K-Means to check whether biomarkers form natural stage groups.

## Results

| Model | CV Macro F1 |
|---|---|
| Random Forest (tuned: depth 4, 100 trees, min split 3) | 0.4599 |
| Logistic Regression | 0.4064 |
| XGBoost | 0.3423 |
| Random Forest (default) | 0.3264 |

- **Logistic Regression** was the only untuned model to detect Stage 1 patients well (62.5% recall, vs 6.2% for Random Forest and 12.5% for XGBoost).
- **Shallow trees worked best.** Limiting Random Forest depth acted as regularisation on this small dataset.
- **SHAP** showed Hepatomegaly, Bilirubin, Albumin and SGOT as the most influential features, consistent with liver physiology.
- **K-Means** did not recover the stages (Adjusted Rand Index 0.083). Silhouette score favoured 2 clusters (early vs late), suggesting stages overlap and progression is gradual.


## Limitations

- Small dataset (312 patients) from a single 1970s-80s trial, so it may not generalise to modern patients
- Severe class imbalance: only 16 Stage 1 patients, so early-stage recall is unreliable
- Imputation and feature selection were done on the full dataset before cross-validation, which can make scores slightly optimistic
- The tuned Random Forest score comes from the same cross-validation used to choose its parameters, so the 0.4599 is likely overestimated
- Overall performance is modest (Macro F1 around 0.4), so this is not ready for clinical use
- Beam Search and GridSearchCV only explore a limited search space

## Future Work

- Fit imputation and feature selection inside each cross-validation fold, and use nested cross-validation for tuning
- Address class imbalance with SMOTE or similar methods
- Add survival analysis using `N_Days`
- Include imaging and histology variables, not just blood biomarkers
- Validate on a more recent, larger and independent cohort

## References

- Mayo Clinic Primary Biliary Cirrhosis (PBC) trial data, via Kaggle (fedesoriano)
