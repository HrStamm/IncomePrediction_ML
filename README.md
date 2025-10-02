# 💡 Predicting Household Income with Machine Learning  

This repository contains the code and reports for **Project 1 & 2** in *02450 Machine Learning* at DTU (Spring 2025).  

**Author:**  
- Valdemar Stamm Kristensen (s244742)  

**Study line:** Artificial Intelligence and Data  

---

## 📌 What’s this project about?  
Across two connected projects, we explored how machine learning can be applied to predict **annual household income** using demographic data.  

- **Project 1:** Focused on **data preprocessing and exploration** – handling missing values, standardizing ordinal features, one-hot encoding categorical variables, and visualizing feature relationships (incl. PCA for structure).  
- **Project 2:** Built on this foundation with **supervised learning models** for both regression and classification, combined with robust evaluation and statistical testing.  

---

## ⚠️ Dataset limitations  
A key limitation of the dataset is that the target variable, **annual household income**, was only provided in ordinal categories rather than as continuous values.  

To make regression possible, each bracket was mapped to a representative numeric value. While this made model training feasible, it also introduced constraints:  
- The target became **artificially continuous**, based on our mapping rather than observed income values.  
- Predictions and coefficients cannot be interpreted as exact economic relationships.  
- At the higher end, the mapping caused a **ceiling effect**, collapsing very different income levels into one value.  

This transformation was necessary, but it reduces the interpretability and generalizability of the results.  

---

## 🧠 How we did it  
- **Preprocessing:** Mode imputation, z-score normalization of ordinal features, one-hot encoding of categorical features, and mapping of income brackets.  
- **Exploration (Project 1):** Feature analysis, PCA, covariance and correlation inspection.  
- **Modeling (Project 2):**  
  - Regression: Linear Regression, Ridge Regression, Artificial Neural Network (ANN).  
  - Classification: Logistic Regression, K-Nearest Neighbors (KNN), and baseline models.  
- **Evaluation:** Nested 10-fold cross-validation for hyperparameter tuning and performance estimation.  
- **Statistical testing:** Paired t-tests to verify if performance differences were significant.  

---

## 📊 What we found  
- **Regression:**  
  - ANN achieved the lowest average MSE (≈0.53).  
  - Ridge Regression followed closely (≈0.55).  
  - Both clearly outperformed the baseline (≈1.0).  

- **Classification:**  
  - Logistic Regression had the lowest error rate (≈0.66).  
  - KNN was close (≈0.68).  
  - Both clearly outperformed the baseline (≈0.81).  

- **Statistical tests:** Confirmed that ANN > Ridge > Baseline for regression, and Logistic Regression > KNN > Baseline for classification.  

---

## 📂 What’s inside the repo  
- `Projekt2.ipynb` → Full notebook (covers Project 1 preprocessing + Project 2 modeling, results, and plots).  
- `Project 2 - code.py` → Standalone Python script of Project 2 code.  
- `corrected_data1.csv` → Dataset after preprocessing.
- `marketing.info.txt` → Information about the dataset used

---

## 🔮 Reflections & Future Ideas  
- **Regression:** ANN performed best, but Ridge offered stable results.  
- **Classification:** Logistic Regression was most effective.  
- Dataset limitations (ordinal target mapped to numbers) are the biggest bottleneck.  
- With larger and more realistic data, results would likely improve.  
- Future work could include ensemble methods, ordinal regression techniques, and validation on external datasets.  

---

## 📖 References  
- *Introduction to Statistics at DTU* (Brockhoff et al., 2024)  
- *Introduction to Machine Learning and Data Mining* (Herlau, Schmidt & Mørup, DTU, 2023)  
- scikit-learn documentation 
