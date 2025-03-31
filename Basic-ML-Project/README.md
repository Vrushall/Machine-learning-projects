### Overview

This project explores and compares different machine learning models, specifically Logistic Regression and Random Forest, using hyperparameter tuning via GridSearchCV. The models are evaluated based on accuracy, precision, and recall to determine their effectiveness.

### Models Used

Regression models

-Ordinary least Square(OLS) 

-Lasso

-Ridge

-Random Forest Regressor

Classification models

-Logistic Regression (L1 & L2 Regularization)

-Random Forest Classifier

### Key Features

-Hyperparameter tuning using GridSearchCV

-Model evaluation using Root Mean Squared Error, Accuracy, Precision, and Recall

-Comparison of L1 vs. L2 regularization in Logistic Regression

-Feature importance analysis

-Performance visualization using scatterplot and confusion matrices

### Data 

The dataset used in this project is included in this repository.

### Libraries

Required Python libraries: *os pandas numpy seaborn matplotlib itertools sklearn*

### Findings

**Regression Task**

**Data Insights**:

- Most features are normally distributed, except for variables like *avgAnnCount, avgDeathsPerYear, popEst2015, studyPerCap, MedianAge, PctBachDeg18_24, PctBachDeg25_Over, PctWhite, PctBlack, PctAsian,* and *PctOtherRace*.
- Identified data inconsistencies:
-- 48 instances where AvgHouseholdSize is less than 1, indicating possible errors.
-- 26 instances where MedianAge is over 116, which is unrealistic as it surpasses the age of the oldest living person.

**Model Performance**:
- **Ordinary Least Squares (OLS) Regression**: RMSE: 17.8915
- **Lasso Regression**: Best alpha: 0.0695 | RMSE: 17.8910 |Top 5 Features: *PctBachDeg25_Over, incidenceRate, PctPublicCoverageAlone, povertyPercent* and *medIncome*
- **Ridge Regression**: Best alpha: 100 | RMSE: 17.9194
- **Random Forest Regressor**: Best Parameters: {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200} | RMSE: 17.9240 | Top 5 Features: *PctBachDeg25_Over, incidenceRate, medIncome, PctHS25_Over* and *avgDeathsPerYear*

**Final Comparison on Test Data**:

-Lasso Regression Test RMSE: 23.0048

-Random Forest Test RMSE: 21.1888

Both models indicate that PctBachDeg25_Over, incidenceRate, and medIncome are among the most influential features affecting the target variable.


**Classification Task**

**Data Insights**:

- Most data is normally distributed, except for *WallMotionIndex* and *epss*.
- *PericardialEffusion* is binary.
- Outliers exist but are not removed as they represent different heart conditions that may impact classification.

**Model Performance**:

- **Logistic Regression (No Regularization)**: Accuracy: 0.7407 | Precision: 0.6667 | Recall: 0.4444
- **Logistic Regression (L1 Regularization)**: Accuracy: 0.7407 | Precision: 0.7500 | Recall: 0.3333 | Best C Value: 0.2976 | L1 Model Coefficients: Feature selection removed one feature.
- **Logistic Regression (L2 Regularization)**: Accuracy: 0.7407 | Precision: 0.7500 | Recall: 0.3333 | Best C Value: 0.0695 | L2 Model Coefficients: No feature removal, coefficients shrunk to prevent overfitting.
- **Findings on L1 vs L2 Regularization**: Both models perform identically, indicating that the removed feature in L1 had little impact on classification. L1 performs feature selection by eliminating the least useful feature. L2 smooths coefficients to prevent overfitting.
- **Random Forest Classifier**: ssifier: Accuracy: 0.8148 | Precision: 0.8333 | Recall: 0.5556 | Feature Importance Ranking: 1: *WallMotionIndex* 2: *AgeAtHeartAttack* 3: *FractionalShortening* 4: *lvdd* 5: *epss* 6: *PericardialEffusion*

**Final Observations**:

- The Random Forest Classifier outperforms Logistic Regression models in accuracy, precision, and recall.
- Feature importance ranking provides insights into which attributes contribute most to heart disease classification.

**Conclusion**

This project explores both regression and classification tasks using multiple machine-learning models. The findings highlight the significance of feature selection, regularization, and model performance evaluation. The Random Forest models consistently outperform their counterparts in both regression and classification tasks, demonstrating their robustness in handling complex datasets.

Future improvements could involve fine-tuning hyperparameters further, exploring ensemble methods, and addressing data inconsistencies for better predictions.


This project is open-source and available for learning purposes.


