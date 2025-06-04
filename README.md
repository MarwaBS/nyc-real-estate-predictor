# Price_Prediction
# README: Real Estate Price Zone Classification

## **Project Overview**
This project aims to classify real estate properties into predefined price zones (Low, Medium, High, and Very High) based on features such as the number of bedrooms, bathrooms, square footage, and more. By using machine learning, the goal is to predict price zones effectively, providing insights for property valuation and market analysis.

---

## **Data Features and Preprocessing**
### **Feature Engineering**
1. **Derived Features:**
   - `PRICE_PER_BED`: Price divided by the number of bedrooms.
   - `PRICE_PER_BATH`: Price divided by the number of bathsrooms.
   - `TOTAL_ROOMS`: Sum of bedrooms and bathrooms.

2. **Binning Square Footage:**
   - Created a `SQFT_CATEGORY` feature:
     - **Small:** Less than 1,000 sqft.
     - **Medium:** Between 1,000 and 2,000 sqft.
     - **Large:** Greater than 2,000 sqft.

3. **Price Zones:**
   - Binned property prices into four categories using the following thresholds:
     - **Low:** $0 – $500,000
     - **Medium:** $500,001 – $1,000,000
     - **High:** $1,000,001 – $2,000,000
     - **Very High:** $2,000,001+

4. **ZIP Code Handling:**
   - Converted `POSTCODE` to `ZIPCODE` as a string for classification.

### **One-Hot Encoding**
Categorical features were encoded using **OneHotEncoder** to transform them into numerical representations.

### **Data Splitting**
- Split the dataset into **training** (80%) and **test** (20%) sets to evaluate the model effectively.

---

## **Model Training**
### **Classification Algorithm**
The **Random Forest Classifier** was selected for its robustness and capability to handle mixed data types.

#### **Hyperparameter Tuning:**
- Used **RandomizedSearchCV** to optimize hyperparameters, including:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `class_weight`

---

## **Evaluation Metrics**
### **Training Performance:**
- **Accuracy:** 100%
- **Classification Report:**
  - Precision, recall, and F1-score for all classes were 1.0.

### **Test Performance (Before SMOTE):**
- **Accuracy:** 96.69%
- Confusion Matrix:
  - High: 108 correct, 2 misclassified
  - Low: 161 correct, 4 misclassified
  - Medium: 250 correct, 1 misclassified
  - Very High: 7 correct, 11 misclassified

### **Test Performance (After SMOTE):**
- **Accuracy:** 97.79%
- Significant improvement in recall for the Very High class:
  - **Before SMOTE:** 67%
  - **After SMOTE:** 72%

### **Test Performance (After Tuning):**
- **Accuracy:** 97.79%
- Confusion Matrix:
  - Improved recall for the Very High class to **72%**.
- Classification Report:
  - Macro average F1-score: 93%
  - Weighted average F1-score: 98%

---

## **Model Comparison**

### **XGBoost**
- **Accuracy:** 98.71%
- **Confusion Matrix:**
[[110 0 0 0] [ 0 165 0 0] [ 0 1 250 0] [ 6 0 0 12]]
- **Classification Report:**
          precision    recall  f1-score   support
      0       0.95      1.00      0.97       110
      1       0.99      1.00      1.00       165
      2       1.00      1.00      1.00       251
      3       1.00      0.67      0.80        18

  accuracy                           0.99       544
  macro avg       0.99      0.92      0.94       544
- **Why XGBoost?**
- **XGBoost** is a gradient boosting algorithm known for its high performance and efficiency. It performs exceptionally well with imbalanced classes and complex patterns. This model was able to achieve high accuracy and precision for most classes, especially for the "Low" and "Medium" price zones, with a slight drop in recall for the "Very High" class, which could be further optimized.

### **Random Forest**
- **Accuracy:** ~98%
- **Confusion Matrix:**
[[107 0 0 3] [ 0 162 3 0] [ 0 1 249 1] [ 4 0 1 13]]
- **Classification Report:**
          precision    recall  f1-score   support
      0       0.96      0.97      0.97       110
      1       0.99      0.98      0.99       165
      2       0.99      0.99      0.99       251
      3       0.76      0.72      0.74        18

  accuracy                           0.98       544
  macro avg       0.93      0.92      0.92       544
  weighted avg   0.98      0.98      0.98       544
- **Why Random Forest?**
- **Random Forest** is an ensemble learning method that is simple to implement and interprets results well. It performed strongly overall, with high precision and recall for most classes, but the performance for the "Very High" class was slightly lower compared to XGBoost. Random Forest is an excellent choice for initial models but might require more refinement for complex patterns.

---

## **Conclusion**
Both **XGBoost** and **Random Forest** demonstrated strong performance, with **XGBoost** showing a slight edge in terms of accuracy and handling imbalanced classes. Specifically, XGBoost achieved an accuracy of **98.71%**, while Random Forest reached **98%** accuracy.

Given its performance and ability to handle complex data, **XGBoost** is selected as the final model for the project. It provides better precision and recall for the majority of classes and has the potential for further optimization, especially in handling the "Very High" price zone more effectively.
