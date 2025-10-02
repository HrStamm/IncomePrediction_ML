# imports:
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from dtuimldmtools import bmplot, feature_selector_lr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from scipy.stats import t
import pandas as pd

# Her har vi annual_income dataen før standarisering:
original_annual_income

# tilføjer til df:
df["Annual_Income_Num"] = original_annual_income
df["Annual_Income_Num"]

# Lav annual income om fra ordinal til continous:

income_mapping = {
    1: 5000,
    2: 15000,
    3: 25000,
    4: 35000,
    5: 45000,
    6: 55000,
    7: 65000,
    8: 72500,
    9: 100000 # 75000 or more
}

# Så mapper vi:
df["Annual_Income_Num"] = original_annual_income.map(income_mapping)
df["Annual_Income_Num"]

# ligesom de andre ordinale variable, så standardisere vi 
# igen efter det nu er en kontinuerlig variabel

print(f"Standardiserer Annual_Income_Num")

# Konverter kolonnen
numeric_values = df["Annual_Income_Num"].astype(float)

# Beregn middelværdi og standardafvigelse
mean_val = numeric_values.mean()
std_val = numeric_values.std(ddof=1)
print(f"Mean: {mean_val}, Std: {std_val}")

# Create standardized version
standardized_income = (df["Annual_Income_Num"] - mean_val) / std_val

# fjerner i df_final hvis den allerede er der
if "Annual_Income_Num_Std" in df_final.columns:
    df_final = df_final.drop(columns=["Annual_Income_Num_Std"])
if "Annual_Income" in df_final.columns:
    df_final = df_final.drop(columns=["Annual_Income"])


new_df_final = pd.DataFrame({"Annual_Income": standardized_income})
df_final = pd.concat([new_df_final, df_final], axis=1)

print(df_final.shape)
print(df_final.head())

X = df_final.drop(columns=["Annual_Income"]).values
y = df_final['Annual_Income'].values

attributeNames = list(df_final.drop(columns=["Annual_Income"]).columns)

# prnter det hele
print('X shape:', X.shape)
print('y shape:', y.shape)
print('Number of attributes:', len(attributeNames))

K = 10  # Ydre CV
CV = KFold(n_splits=K, shuffle=True)

M = X.shape[1]  # antal features
Features = np.zeros((M, K))
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))

k = 0
for train_index, test_index in CV.split(X):
    print(f"\nFold {k + 1}/{K}")

    # Splitter data:
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    internal_cv = 10  

    # Baseline 
    Error_train_nofeatures[k] = np.mean((y_train - np.mean(y_train)) ** 2)
    Error_test_nofeatures[k] = np.mean((y_test - np.mean(y_train)) ** 2)

    # Alle features (ingen feature selection)
    model_all = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.mean((y_train - model_all.predict(X_train)) ** 2)
    Error_test[k] = np.mean((y_test - model_all.predict(X_test)) ** 2)

    # Forward feature selection:
    selected_features, features_record, loss_record = feature_selector_lr(
        X_train, y_train, internal_cv, display=False
    )

    if len(selected_features) == 0:
        print("No features were selected.")
    else:
        Features[selected_features, k] = 1
        model_fs = lm.LinearRegression().fit(X_train[:, selected_features], y_train)
        Error_train_fs[k] = np.mean((y_train - model_fs.predict(X_train[:, selected_features])) ** 2)
        Error_test_fs[k] = np.mean((y_test - model_fs.predict(X_test[:, selected_features])) ** 2)

    k += 1

# Resultater
print("\nLINEAR REGRESSION WITHOUT FEATURE SELECTION")
print(f"Train error: {Error_train.mean():.2f}")
print(f"Test error:  {Error_test.mean():.2f}")

print("\nLINEAR REGRESSION WITH FEATURE SELECTION")
print(f"Train error: {Error_train_fs.mean():.2f}")
print(f"Test error:  {Error_test_fs.mean():.2f}")

# R^2
print(f"\nR^2 train: {(Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum():.2f}")
print(f"R^2 test:  {(Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum():.2f}")

# Plot: hvilke features blev valgt i hvilke fold
plt.figure()
bmplot(attributeNames, range(1, K + 1), -Features)
plt.xlabel("CV fold")
plt.ylabel("Attribute")
plt.title("Selected Features Across Folds")
plt.clim(-1.5, 0)
plt.tight_layout()
plt.show()

K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)


lambdas = np.logspace(-4, 2, 30) 
cv_errors = []

# Krydsvalidering for hver λ-værdi:
for lam in lambdas:
    fold_errors = []  
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = Ridge(alpha=lam)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        fold_errors.append(mse)
    
    avg_mse = np.mean(fold_errors)
    cv_errors.append(avg_mse)


cv_errors = np.array(cv_errors)

# Finder den lambda-værdi, der giver den laveste gennemsnitlige MSE:
optimal_lambda = lambdas[np.argmin(cv_errors)]
print("Optimal lambda (alpha):", optimal_lambda)
print("Minimum CV Error (MSE):", np.min(cv_errors))

# Plotter det hele
plt.figure(figsize=(8, 6))
plt.semilogx(lambdas, cv_errors, marker='o', linestyle='-')
plt.xlabel("Lambda (Regulariseringsparameter)")
plt.ylabel("Gennemsnitlig Mean Squared Error (MSE)")
plt.title("Generaliseringseffekt vs. Lambda i Ridge Regression")
plt.grid(True)
plt.show()

K_outer = 10  
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)

K_inner = 10  


lambdas = np.logspace(-4, 2, 10) 
hidden_units_range = [1, 3, 5, 10, 20]
results = []  

# påbegyndelsen af outer loop:
fold = 1
for outer_train_idx, outer_test_idx in outer_cv.split(X):
    X_train_outer = X[outer_train_idx]
    y_train_outer = y[outer_train_idx]
    X_test_outer = X[outer_test_idx]
    y_test_outer = y[outer_test_idx]
    
    baseline_prediction = np.mean(y_train_outer)
    baseline_error = np.mean((y_test_outer - baseline_prediction) ** 2)
    
    best_ridge_lambda = None
    best_ridge_inner_error = np.inf  
    
    inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=42)

    # påbegyndelsen af inner loop:
    for lam in lambdas:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]

            model = Ridge(alpha=lam, random_state=42)
            model.fit(X_train_inner, y_train_inner)
            y_val_pred = model.predict(X_val_inner)
            mse = np.mean((y_val_inner - y_val_pred) ** 2)
            inner_errors.append(mse)
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_ridge_inner_error:
            best_ridge_inner_error = avg_inner_error
            best_ridge_lambda = lam
    
 
    final_ridge = Ridge(alpha=best_ridge_lambda, random_state=42)
    final_ridge.fit(X_train_outer, y_train_outer)
    ridge_error = np.mean((y_test_outer - final_ridge.predict(X_test_outer)) ** 2)
    
    # ANN model selektion:
    best_ann_hidden = None
    best_ann_inner_error = np.inf
    
    for h in hidden_units_range:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]
            
            ann_model = MLPRegressor(hidden_layer_sizes=(h,), max_iter=1000, random_state=42)
            ann_model.fit(X_train_inner, y_train_inner)
            y_val_pred = ann_model.predict(X_val_inner)
            mse = np.mean((y_val_inner - y_val_pred) ** 2)
            inner_errors.append(mse)
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_ann_inner_error:
            best_ann_inner_error = avg_inner_error
            best_ann_hidden = h
    
    # Træner det endelige ANN-model med de bedste hyperparametre:
    final_ann = MLPRegressor(hidden_layer_sizes=(best_ann_hidden,), max_iter=1000, random_state=42)
    final_ann.fit(X_train_outer, y_train_outer)
    ann_error = np.mean((y_test_outer - final_ann.predict(X_test_outer)) ** 2)
    
    results.append({
        'baseline_error': baseline_error,
        'best_ridge_lambda': best_ridge_lambda,
        'ridge_error': ridge_error,
        'best_ann_hidden': best_ann_hidden,
        'ann_error': ann_error
    })
    
    print(f"Completed outer fold {fold}")
    fold += 1

results_df = pd.DataFrame(results)
print("\nNested CV Results:")
print(results_df)

print("\nAverage Test Errors:")
print("Baseline:", results_df['baseline_error'].mean())
print("Ridge:", results_df['ridge_error'].mean())
print("ANN:", results_df['ann_error'].mean())

# plotter det hele:
plt.figure(figsize=(10, 6))
plt.plot(results_df.index + 1, results_df['baseline_error'], 'o-', label='Baseline')
plt.plot(results_df.index + 1, results_df['ridge_error'], 'o-', label='Ridge')
plt.plot(results_df.index + 1, results_df['ann_error'], 'o-', label='ANN')
plt.xlabel("Outer Fold")
plt.ylabel("Test MSE")
plt.title("Test MSE across Outer Folds")
plt.legend()
plt.show()

# Laver en resultattabel:

results_df = pd.DataFrame({
    'Outer Fold': range(1, len(results) + 1),
    'h*': [res['best_ann_hidden'] for res in results],
    'E_test(ANN)': [res['ann_error'] for res in results],
    'lambda*': [res['best_ridge_lambda'] for res in results],
    'E_test(Ridge)': [res['ridge_error'] for res in results],
    'E_test(Baseline)': [res['baseline_error'] for res in results]
})

print(results_df)

# laver en liste med fejlene:
ann_errors = results_df['E_test(ANN)'].tolist()
ridge_errors = results_df['E_test(Ridge)'].tolist()
baseline_errors = results_df['E_test(Baseline)'].tolist()

print("ANN errors:", ann_errors)
print("Ridge errors:", ridge_errors)
print("Baseline errors:", baseline_errors)

# kode til parret t-test
def paired_ttest(model1, model2, alpha=0.05):
    model1, model2 = np.array(model1), np.array(model2)
    d = model1 - model2
    K = len(d)
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    t_stat = d_mean / (d_std / np.sqrt(K))
    df = K - 1
    p_val = 2 * (1 - t.cdf(abs(t_stat), df))
    t_crit = t.ppf(1 - alpha/2, df)
    margin = t_crit * (d_std / np.sqrt(K))
    ci = (d_mean - margin, d_mean + margin)

    return t_stat, p_val, ci

t_stat, p_val, ci = paired_ttest(ann_errors, ridge_errors, alpha=0.05)
print("\nPaired t-test for Ann vs Linear regression:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")

t_stat, p_val, ci = paired_ttest(ann_errors, baseline_errors, alpha=0.05)
print("\nPaired t-test for Ann vs Baseline:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")

t_stat, p_val, ci = paired_ttest(ridge_errors,baseline_errors, alpha=0.05)
print("\nPaired t-test for Linear regression vs Baseline:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")

unique_vals = np.unique(y)
mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
y_class = np.array([mapping[val] for val in y])


K_outer = 10  
outer_cv = KFold(n_splits=K_outer, shuffle=True, random_state=42)
K_inner = 10 


lambdas = np.logspace(-4, 2, 10)  
neighbors_range = [1, 3, 5, 7, 9]


results_classification = []

fold = 1
# påbegyndelsen af outer loop:
for outer_train_idx, outer_test_idx in outer_cv.split(X):
    X_train_outer = X[outer_train_idx]
    X_test_outer = X[outer_test_idx]
    y_train_outer = y_class[outer_train_idx]
    y_test_outer = y_class[outer_test_idx]
    
    unique_labels, counts = np.unique(y_train_outer, return_counts=True)
    baseline_class = unique_labels[np.argmax(counts)]
    baseline_preds = np.full(y_test_outer.shape, baseline_class)
    baseline_error = np.mean(baseline_preds != y_test_outer)
    
    best_logreg_lambda = None
    best_logreg_inner_error = np.inf  
    inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=42)
    
    # påbegyndelsen af inner loop:
    for lam in lambdas:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]
            
            C = 1.0 / lam
            logreg = LogisticRegression(C=C, penalty='l2', solver='liblinear', random_state=42)
            logreg.fit(X_train_inner, y_train_inner)
            preds = logreg.predict(X_val_inner)
            error = np.mean(preds != y_val_inner)
            inner_errors.append(error)
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_logreg_inner_error:
            best_logreg_inner_error = avg_inner_error
            best_logreg_lambda = lam
            
    C_final = 1.0 / best_logreg_lambda
    final_logreg = LogisticRegression(C=C_final, penalty='l2', solver='liblinear', random_state=42)
    final_logreg.fit(X_train_outer, y_train_outer)
    logreg_preds = final_logreg.predict(X_test_outer)
    logreg_error = np.mean(logreg_preds != y_test_outer)
    
    best_knn_k = None
    best_knn_inner_error = np.inf
    
    # KNN model selektion:
    for k_val in neighbors_range:
        inner_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]
            
            knn = KNeighborsClassifier(n_neighbors=k_val)
            knn.fit(X_train_inner, y_train_inner)
            preds = knn.predict(X_val_inner)
            error = np.mean(preds != y_val_inner)
            inner_errors.append(error)
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_knn_inner_error:
            best_knn_inner_error = avg_inner_error
            best_knn_k = k_val
            
    final_knn = KNeighborsClassifier(n_neighbors=best_knn_k)
    final_knn.fit(X_train_outer, y_train_outer)
    knn_preds = final_knn.predict(X_test_outer)
    knn_error = np.mean(knn_preds != y_test_outer)
    
    results_classification.append({
        'fold': fold,
        'baseline_error': baseline_error,
        'best_logreg_lambda': best_logreg_lambda,
        'logreg_error': logreg_error,
        'best_knn_k': best_knn_k,
        'knn_error': knn_error
    })
    
    print(f"Completed outer fold {fold}")
    fold += 1

results_df_class = pd.DataFrame(results_classification)
print("Classification Nested CV Results:")
print(results_df_class)

print("\nAverage Test Errors:")
print("Baseline:", results_df_class['baseline_error'].mean())
print("Logistic Regression:", results_df_class['logreg_error'].mean())
print("KNN:", results_df_class['knn_error'].mean())

# Plotter resultaterne:
plt.figure(figsize=(10, 6))
plt.plot(results_df_class['fold'], results_df_class['baseline_error'], 'o-', label='Baseline')
plt.plot(results_df_class['fold'], results_df_class['logreg_error'], 'o-', label='Logistic Regression')
plt.plot(results_df_class['fold'], results_df_class['knn_error'], 'o-', label='KNN')
plt.xlabel("Outer Fold")
plt.ylabel("Misclassification Rate")
plt.title("Test Error Across Outer Folds")
plt.legend()
plt.show()

# Laver samme statistik som før:
logreg_errors = results_df_class['logreg_error'].tolist()
knn_errors    = results_df_class['knn_error'].tolist()
baseline_errors = results_df_class['baseline_error'].tolist()

print("Logistic Regression errors:", logreg_errors)
print("KNN errors:", knn_errors)
print("Baseline errors:", baseline_errors)

t_stat, p_val, ci = paired_ttest(logreg_errors, knn_errors, alpha=0.05)
print("\nPaired t-test for Logistic Regression vs. KNN:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")

t_stat, p_val, ci = paired_ttest(logreg_errors, baseline_errors, alpha=0.05)
print("\nPaired t-test for Logistic Regression vs. Baseline:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")

t_stat, p_val, ci = paired_ttest(knn_errors, baseline_errors, alpha=0.05)
print("\nPaired t-test for KNN vs. Baseline:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.8f}")
print(f"95% CI:      [{ci[0]:.3f}, {ci[1]:.3f}]")