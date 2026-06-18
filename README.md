# Brain Stroke Prediction (Classification)

**Author:** Maria | **Date:** 19/06/2026 | **Version:** v1.0   | **Github:** [Brain Stroke Classification Project](https://github.com/vcmeamaria/brain-stroke-classification-project)


## Overview

This notebook predicts whether a patient is likely to suffer a stroke. The target variable is `stroke`, where 1 means a stroke occurred and 0 means it did not. Because the target has two classes, this is a classification problem.

The training data is `brain_stroke_data.csv`, which has 4,482 patient records and 11 columns. The features are a mix of demographic and clinical details, such as age, gender, hypertension, heart disease, average glucose level, BMI and smoking status.

The main metric is recall for the stroke class. The reasoning given is that in a medical setting, missing a real stroke case is worse than raising a false alarm, so finding as many true cases as possible matters most.

## Unseen data

The final model is applied to `brain_stroke_unseen.csv`. This file has 499 records and 10 columns. The `stroke` column is not included, as it is the value the model has to predict.

## Preprocessing decisions

There were no missing values and no duplicate rows in either file. The `Unknown` value in smoking status was kept as a real category rather than treated as missing, since it represents patients whose history was not recorded. The `children` value in work type was also kept, as it represents genuine observations.

Encoding was split by feature type. The binary columns gender, ever married and residence type were encoded as 0 and 1 using an ordinal encoder. The multi class columns work type and smoking status were one hot encoded, which avoids implying any order between categories. The numeric columns age, average glucose level and BMI were standardised with StandardScaler, which matters for logistic regression. Hypertension and heart disease were already stored as 0 and 1, so they were passed through unchanged. All of this was wrapped in a ColumnTransformer so the same steps could be applied consistently and without leakage.

The data was split 80 to 20 with stratified sampling to keep the same class balance in both sets. The classes were very imbalanced, at roughly 95 percent no stroke and 5 percent stroke, so SMOTE was used to balance the training data. SMOTE was applied only after the split and only to the training set, so the test set kept its original balance.

## Models trained and the final choice

Four models were trained on the SMOTE balanced data and tested on the original test set: logistic regression, a decision tree, a random forest and gradient boosting. The decision tree depth was chosen by cross validation.

Logistic regression was chosen as the final model. On the test set it gave the highest recall for the stroke class at 0.7778, well above the other models. The reasoning given is that recall is the priority, since the goal is to catch as many stroke cases as possible.

## How the model was tuned

The chosen model was tuned with GridSearchCV using 5 fold stratified cross validation, optimised on the F1 score for the stroke class. The search covered the regularisation strength, solver and penalty. The best settings were C of 0.1, an L2 penalty and the liblinear solver, with a best cross validated F1 of 0.8157.

## Final performance on the test set

The tuned logistic regression scored as follows on the test set:

- Accuracy 0.7202
- Precision for the stroke class 0.1268
- Recall for the stroke class 0.7778
- F1 for the stroke class 0.2181

Tuning made almost no difference compared with the baseline logistic regression, which had recall of 0.7778 and F1 of 0.2208.

## Predictions file

The predictions were saved to `brain_stroke_predictions.csv`. It contains a single column called `Predictions` with one row for each of the 499 unseen records, holding a 0 or 1 value. In the unseen set the model predicted no stroke for about 67.9 percent of records and stroke for about 32.1 percent.

## Limitations and future work

The dataset has relatively few stroke cases, and some features contain unknown values. The model would need external validation on independent data before any real world use. Future work noted in the notebook includes building extra features from medical knowledge, testing more advanced models, trying different probability thresholds to improve recall, and validating the model on other datasets.

---
AI disclosure: AI was utilised to support coding fixing & markdown improvement.
