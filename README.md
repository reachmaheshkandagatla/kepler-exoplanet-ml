# Kepler Exoplanet Classification

**Machine Learning Assignment 2 | BITS Pilani WILP**  
**Name:** Mahesh Kandagatla  
**BITS ID:** 2026AJ05552

**Kaggle Data:** https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results
**Live application:** https://kepler-exoplanet-ml.streamlit.app/  
**GitHub repository:** https://github.com/reachmaheshkandagatla/kepler-exoplanet-ml

## 1. Problem statement

In this assignment, I worked with NASA's Kepler Objects of Interest (KOI) data to predict whether an observation is classified as **CANDIDATE**, **CONFIRMED**, or **FALSE POSITIVE**. I trained five machine learning models on the same dataset and compared their results using six evaluation metrics.

I also built a Streamlit application so that the models can be tested through a browser. The app opens with a sample test dataset already loaded, and users can switch between models or upload their own labeled CSV file.

The aim is to compare the methods and understand where they make mistakes. These predictions are not a confirmation of a planet.

## 2. Dataset description

I used the supplied `space_data.csv` file, which contains **9,564 rows and 50 columns**. The target column is `koi_disposition`.

| Class | Number of observations |
|---|---:|
| FALSE POSITIVE | 5,023 |
| CONFIRMED | 2,293 |
| CANDIDATE | 2,248 |

After reviewing the columns, I selected **37 numerical features** for training. I excluded identifiers and fields that could reveal information too closely related to the target, such as `koi_pdisposition`, `koi_score`, and the `koi_fpflag_*` columns. I handled missing feature values using median imputation within each model pipeline.

I split the dataset into **7,662 training rows** and **1,902 test rows**. I used `kepid`, the star identifier, to keep observations from the same star together. This avoids having the same star represented in both training and test data.

For a more reliable comparison, I also ran five-fold group-aware cross-validation on the training data.

## 3. GitHub repository and application

- **GitHub:** https://github.com/reachmaheshkandagatla/kepler-exoplanet-ml
- **Streamlit app:** https://kepler-exoplanet-ml.streamlit.app/

The repository contains the application code, the five saved models, the feature list, and `data/test_data.csv`. The app loads the trained models rather than retraining them each time a user selects one.

## 4. Models and test results

I trained Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). I evaluated all five models on the same 1,902-row test set.

For Precision, Recall, and F1, I used the **macro average**, which gives equal weight to the three classes. AUC is the macro-averaged one-vs-rest ROC AUC. I chose F1 as the main comparison metric because I did not want good performance on the largest class to hide poor results on the others.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7250 | 0.8598 | 0.6581 | 0.6651 | 0.6470 | 0.5436 |
| **Decision Tree** | **0.7382** | 0.8535 | **0.6947** | **0.7031** | **0.6985** | **0.5702** |
| KNN | 0.7103 | 0.8306 | 0.6471 | 0.6596 | 0.6478 | 0.5216 |
| Gaussian Naive Bayes | 0.3927 | 0.7142 | 0.4959 | 0.4648 | 0.3586 | 0.2149 |
| SVM | 0.7371 | **0.8755** | 0.6782 | 0.6696 | 0.6562 | 0.5611 |

### What I observed

**Logistic Regression:** This was a reasonable baseline, with 72.50% accuracy. However, it identified only about 28% of the actual CANDIDATE observations. This suggests that the classes may not be easy to separate using this model's linear decision boundaries.

**Decision Tree:** This gave me the best test F1 score (0.6985), as well as the highest test accuracy and MCC among the five models. It also performed better on CANDIDATE observations than the other models. One concern is that its training accuracy was 84.12%, compared with 73.82% on the test data, so it may be overfitting to some extent.

**KNN:** KNN achieved 71.03% test accuracy. Its training accuracy was 100%, but this is expected with the distance-weighted setup because training observations can be their own nearest neighbors. The lower test score shows that the training result does not reflect how well it generalizes.

**Gaussian Naive Bayes:** This performed the worst, with 39.27% accuracy and a F1 of 0.3586. One possible reason is that the numerical measurements are related to each other, while Naive Bayes makes simplifying assumptions about how features behave within each class. I have not tested that explanation directly.

**SVM:** SVM was close to Decision Tree on accuracy (73.71%) and had the highest AUC (0.8755). Its training and test accuracies were also fairly close. However, it struggled with the CANDIDATE class, which lowered its F1 score.

## 5. Cross-validation results

I used five-fold `StratifiedGroupKFold` cross-validation on the training set, keeping observations from the same star in the same fold.

| Model | Mean accuracy | Accuracy SD | Mean F1 | F1 SD | Mean MCC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.7163 | 0.0138 | **0.6792** | 0.0143 | 0.5417 |
| SVM | **0.7379** | 0.0072 | 0.6629 | 0.0113 | **0.5679** |
| Logistic Regression | 0.7251 | 0.0107 | 0.6510 | 0.0101 | 0.5473 |
| KNN | 0.6951 | 0.0128 | 0.6344 | 0.0150 | 0.5016 |
| Gaussian Naive Bayes | 0.4227 | 0.0110 | 0.3949 | 0.0089 | 0.2455 |

Decision Tree again had the highest average F1. SVM had the best average accuracy and MCC, so the preferred model depends on the metric being prioritized. I used F1 for my final choice because all three classes are important in this task.

I had already examined the test results during my initial model comparison. Therefore, I treat those numbers as exploratory results rather than claiming that the test set remained completely untouched throughout model selection.

## 6. Why some results are lower than expected

The most difficult class was **CANDIDATE**. For Decision Tree, there were 438 actual CANDIDATE observations in the test set. The model classified **225 correctly**, but predicted **80 as CONFIRMED** and **133 as FALSE POSITIVE**. That gives a CANDIDATE recall of **51.37%**.

A possible reason is that CANDIDATE observations share some measured characteristics with the other two groups. The data is also imbalanced: **53.7% of the test rows are FALSE POSITIVE**. This means a model can achieve reasonable overall accuracy while still missing many candidates.

These are possible explanations based on the results, not proven causes. I would need to investigate the features and the misclassified observations more closely to establish why the errors occur.

## 7. Model selected

I selected **Decision Tree** as the best model for this assignment. It achieved the highest F1 on both the held-out test set (**0.6985**) and cross-validation (**0.6792**). It also identified more CANDIDATE observations correctly than the other models.

SVM is a strong alternative, particularly if accuracy, MCC, or AUC is the priority. However, given my focus on balancing performance across the three classes, Decision Tree was the more suitable choice.

## 8. How the application works

The Streamlit app provides a dropdown to select any of the five trained models. By default, it loads the included test dataset, so the results appear without requiring the examiner to upload a file. There is also an option to upload another labeled CSV containing the required 37 features and the `koi_disposition` column.

After a model is selected, the app shows the six evaluation metrics, confusion matrix, classification report, a short performance analysis, and a table of predictions. Predictions can be downloaded as a CSV file.

### Running the project locally

The saved models were trained with **scikit-learn 1.6.1**. I recommend Python 3.11 for compatibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 9. Project files

```text
kepler-exoplanet-ml/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── test_data.csv
└── models/
    ├── decision_tree.joblib
    ├── logistic_regression.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── feature_names.json
```

## 10. Conclusion and possible improvements

This assignment helped me compare models beyond accuracy alone. Decision Tree performed best overall by my chosen F1 measure, but none of the models handled the CANDIDATE class particularly well. That was the main limitation I observed.

If I continue this work, I would examine which features contribute most to the errors, test class-weighted methods, tune model parameters using group-aware cross-validation, and evaluate the final approach on a new independent dataset. I would also investigate whether the models can be better calibrated before interpreting their probability estimates.