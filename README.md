Baseline Multiclass Diabetes Prediction

This repository contains the **Option A baseline implementation** supporting the coursework review of **Abnoosian, Farnoosh and Behzadi (2023)**, *Prediction of diabetes disease using an ensemble of machine learning multi-classifier models*.

## Dataset and limitations

- Public Iraqi Patient Dataset for Diabetes (IPDD), original source: https://data.mendeley.com/datasets/wj9rwkp9c2/1
- Original CSV: 1,000 observations, 14 columns including identifiers and target.
- Standardize Gender and CLASS strings, remove identifier columns and 174 repeated clinical-feature + class rows, leaving **826 distinct recorded clinical profiles**. These are not verified independent patients.
- Class labels: N=0 (non-diabetic), P=1 (prediabetic), Y=2 (diabetic).
- Input variables: Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI.
- Authors used a different 993-row working dataset including separately sourced FBS values; **results cannot be directly compared** as a controlled experiment.

## Implementation structure

1. Data loading and audit → class-text normalization → clinical-profile de-duplication.
2. Stratified 80/20 holdout (random state 42).
3. RandomForestClassifier baseline, training-only GridSearchCV, dummy majority baseline.
4. RF evaluation (accuracy, macro metrics, class recalls, confusion matrix, macro OVO-AUC, prediabetic average precision); HbA1c exploratory ablation and training-only nested CV.
5. Raw-input reusable RF pipeline; LogisticRegression with StandardScaler fitted inside each training fold and its own GridSearchCV.
6. Optional RF+LR probability averaging: equal-weight versus **training-only out-of-fold AUC weights**.
7. Model saving/reloading, consistency assertions, figures, CSV outputs and final ZIP archive.

## Interpreting results

- Untuned/tuned RF held-out accuracy: **0.9759**, macro-F1 **0.9093** on n=166 (8 true prediabetic cases).
- Tuned LR held-out accuracy: **0.9398**, macro-F1 **0.8340**.
- Exploratory two-model ensembles: accuracy **0.9699**, macro-F1 **0.9176**, and *identical class predictions* for equal versus AUC weights. AUC differences do not imply the AUC-weighted ensemble improved class decisions.
- Paper Table 9 reports best ensemble accuracy **0.998** and AUC **0.999**, whereas the article abstract gives average accuracy **0.9887**. These values are left as printed and are **not directly comparable** with this baseline, given different datasets and evaluation protocols.
- Statistical evidence of strong HbA1c association should not be confused with proof of the target construction rule or clinical generalization.

## Requirements

- Python 3.10+ recommended, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`.
- Runs in Google Colab; also runs in Jupyter locally with the dataset CSV alongside the notebook.
- Saved `.joblib` models should only be loaded from trusted sources. Binary model artifacts and data are not required in the GitHub repository; the notebook regenerates outputs.
