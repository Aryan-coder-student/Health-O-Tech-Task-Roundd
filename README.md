# Task Round Description: Immunization Prediction Model Enhancement

## Objective
The primary goal is to increase the accuracy of an immunization prediction model using the provided dataset. This involves preprocessing data, exploring various machine learning classifiers, and utilizing BayesSearchCV for hyperparameter tuning. The benchmark accuracy target is 80%.

## Dataset Overview
The dataset consists of demographic factors influencing vaccination rates, including:
- **Age:** Age of individuals
- **Gender:** Gender of individuals (values: "Male," "Female")
- **Region:** Geographical area (values: "urban," "rural")
- **State:** Indian states where immunization has occurred
- **Community:** Community classification (values: "sc," "st," "obc," "general")
- **Vaccination_Status:** Vaccination status (0 for not vaccinated, 1 for vaccinated)

---

## Preprocessing Pipeline

1. **Data Cleaning:**
   - Duplicates were removed from the dataset to ensure data quality.

2. **Data Splitting:**
   - The dataset is divided into features (`X`) and target (`y`) for model training and evaluation.

3. **Categorical and Numerical Columns Identification:**
   - Categorical and numerical columns are identified separately to apply appropriate transformations.

4. **Label Encoding and Scaling:**
   - A `ColumnTransformer` is used to apply `RobustScaler` to numerical columns and pass categorical columns through unchanged.
   - Custom label encoders are created for each categorical column to handle non-numeric data, transforming it into a numerical format for model compatibility.

5. **Pipeline Construction:**
   - The entire preprocessing setup is integrated into a `Pipeline` that:
     - Encodes categorical labels with `FunctionTransformer`.
     - Scales numerical features with `RobustScaler`.

This preprocessing pipeline ensures that categorical data is encoded and numerical data is scaled, preparing the data for optimal performance across various machine learning models.

---

## Classifier Summaries

1. **XGBClassifier**
   - **Best Parameters:**
     - `colsample_bytree`: 0.867  
     - `gamma`: 0.282  
     - `learning_rate`: 0.041  
     - `max_depth`: 4  
     - `n_estimators`: 180  
     - `reg_alpha`: 0.373  
     - `reg_lambda`: 0.459  
     - `subsample`: 0.767  
   - **Accuracy:** 81.8%
   - **Classification Report:**  
     - Precision: 0 (82%), 1 (82%)  
     - Recall: 0 (87%), 1 (74%)  
     - F1-score: 0 (85%), 1 (78%)  
   ![XGBClassifier](https://github.com/user-attachments/assets/fe92ca1a-b7db-4519-b9df-30dea7f3de6b)

2. **GradientBoostingClassifier**
   - **Best Parameters:**
     - `learning_rate`: 0.095  
     - `loss`: "exponential"  
     - `max_depth`: 3  
     - `max_features`: "log2"  
     - `min_samples_split`: 3  
     - `n_estimators`: 145  
     - `subsample`: 0.831  
   - **Accuracy:** 82.4%
   - **Classification Report:**  
     - Precision: 0 (83%), 1 (82%)  
     - Recall: 0 (87%), 1 (76%)  
     - F1-score: 0 (85%), 1 (79%)  
   ![GradientBoostingClassifier](https://github.com/user-attachments/assets/e4f89509-6df9-406a-9d9e-235c130f0b07)

3. **RandomForestClassifier**
   - **Best Parameters:**
     - `bootstrap`: False  
     - `criterion`: "gini"  
     - `max_depth`: 20  
     - `max_features`: "log2"  
     - `min_samples_leaf`: 3  
     - `min_samples_split`: 3  
     - `n_estimators`: 251  
   - **Accuracy:** 80.6%
   - **Classification Report:**  
     - Precision: 0 (82%), 1 (78%)  
     - Recall: 0 (84%), 1 (76%)  
     - F1-score: 0 (83%), 1 (77%)  
   ![RandomForestClassifier](https://github.com/user-attachments/assets/199e3eed-7006-4fee-b15b-467ef299d172)

4. **CatBoostClassifier**
   - **Best Parameters:**
     - `bagging_temperature`: 0.445  
     - `border_count`: 237  
     - `depth`: 3  
     - `iterations`: 143  
     - `l2_leaf_reg`: 4  
     - `learning_rate`: 0.0962  
   - **Accuracy:** 82.8%
   - **Classification Report:**  
     - Precision: 0 (83%), 1 (83%)  
     - Recall: 0 (88%), 1 (75%)  
     - F1-score: 0 (85%), 1 (79%)  
   ![CatBoostClassifier](https://github.com/user-attachments/assets/2c0912f0-a757-415c-bddf-cb06aed55148)

5. **SVC**
   - **Best Parameters:**
     - `C`: 8.39  
     - `degree`: 5  
     - `gamma`: "scale"  
     - `kernel`: "rbf"  
   - **Accuracy:** 76.6%
   - **Classification Report:**  
     - Precision: 0 (81%), 1 (71%)  
     - Recall: 0 (76%), 1 (77%)  
     - F1-score: 0 (79%), 1 (74%)  
   ![SVC](https://github.com/user-attachments/assets/7646a4fb-a270-4142-b164-75b68bb08287)

6. **AdaBoostClassifier**
   - **Best Parameters:**
     - `learning_rate`: 0.839  
     - `n_estimators`: 182  
   - **Accuracy:** 82%
   - **Classification Report:**  
     - Precision: 0 (83%), 1 (80%)  
     - Recall: 0 (86%), 1 (77%)  
     - F1-score: 0 (84%), 1 (79%)  
   ![AdaBoostClassifier](https://github.com/user-attachments/assets/2dd29e7e-c4d7-4e21-a879-6afe1a781ddc)

7. **Logistic Regression**
   - **Best Parameters (BayesSearchCV):**  
     - `C`: 41.01  
     - `max_iter`: 246  
     - `penalty`: 'l2'  
     - `tol`: 0.000183  
   - **Accuracy:** 77.6%
   - **Classification Report:**  
     - Precision: 0 (81%), 1 (74%)  
     - Recall: 0 (80%), 1 (75%)  
     - F1-score: 0 (80%), 1 (74%)  
   ![Logistic Regression](https://github.com/user-attachments/assets/a6f70bbd-9796-4788-a758-de6d9a38eb39)

---

## Best Model

The **CatBoostClassifier** emerged as the best-performing model based on BayesSearchCV, achieving an accuracy of **82.8%**. This model demonstrated the highest F1-scores for both classes and had well-balanced precision and recall metrics, making it the most reliable choice for the immunization prediction task.
![Best Model](https://github.com/user-attachments/assets/2b49f1b2-8903-47d8-8df6-6e786feba245)

---
