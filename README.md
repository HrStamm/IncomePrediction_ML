# 💡 Predicting Household Income with Machine Learning  

This repository contains the code and report for **Project 2** in *02450 Machine Learning* at DTU (April 2025).  

**Authors:**  
- Valdemar Stamm Kristensen (s244742)  

**Study line:** Artificial Intelligence and Data  

---

## 📌 What’s this project about?  
We wanted to see how well machine learning models could predict **annual household income** using demographic data.  

To do that, we tested both regression and classification approaches:  
- **Regression:** Linear Regression, Ridge Regression, and an Artificial Neural Network (ANN).  
- **Classification:** Logistic Regression, K-Nearest Neighbors (KNN), and a baseline for comparison.  

This project builds on our first one, where we cleaned and preprocessed the data. Here we focused on model building, evaluation, and statistical testing.  

---

## ⚠️ Dataset limitations  
Our dataset had a key limitation: **the target variable (annual household income) was not continuous, but ordinal.**  

To use regression, we had to **approximate each income group by its average value.**  
- This made it possible to train regression models, but introduced uncertainty and approximation error.  
- As a result, model coefficients should be interpreted with caution – they may not fully reflect real-world economic relationships.  

---

## 🧠 How we did it  
- Preprocessing with one-hot encoding + z-score normalization.  
- Tried out different models: from simple linear regression to more flexible ANN.  
- Used **nested 10-fold cross-validation** to tune hyperparameters fairly.  
- Ran **statistical tests** (paired t-tests) to check if differences in performance were actually significant.  

---

## 📊 What we found  
- **Regression results:**  
  - ANN had the best performance (MSE ≈ 0.53).  
  - Ridge Regression was close (MSE ≈ 0.55).  
  - Both were much better than the baseline (MSE ≈ 1.0).  

- **Classification results:**  
  - Logistic Regression came out on top (error ≈ 0.66).  
  - KNN was a close second (error ≈ 0.68).  
  - Both crushed the baseline (error ≈ 0.81).  

- **Stats tests confirmed:** ANN > Ridge > Baseline for regression, and Logistic Regression > KNN > Baseline for classification.  

---

## 📂 What’s inside the repo  
- `Projekt2.ipynb` → Jupyter Notebook with full analysis, results, and plots.  
- `Project 2 - code.py` → Standalone Python script.  
---

## 🔮 Reflections & Future Ideas  
- ANN seems best for regression, Logistic Regression best for classification – at least on this dataset.  
- Dataset quality is a big limitation here since income values were approximated.  
- With more realistic and larger data, results would likely improve.  
- Next steps could be trying out ensemble methods like random forests or boosting, or exploring better ways to handle ordinal targets.  

---

## 📖 References  
- *Introduction to Statistics at DTU* (Brockhoff et al., 2024)  
- *Introduction to Machine Learning and Data Mining* (Herlau, Schmidt & Mørup, DTU, 2023)   
- scikit-learn documentation  
