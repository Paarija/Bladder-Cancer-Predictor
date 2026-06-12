#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, recall_score, fbeta_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def evaluate_and_visualize():
    print("Loading preprocessed dataset...")
    df = pd.read_excel(r"C:\Users\hp\OneDrive\Desktop\projects\preprocessed_data.xlsx")
    
    # 1. Stratified Holdout Test Split (20% to prevent leakage)
    train_pool, holdout_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["LABEL"],
        random_state=42
    )
    
    X_train_full = train_pool.drop(columns=["genes", "LABEL"])
    y_train_full = train_pool["LABEL"]
    
    X_holdout_full = holdout_df.drop(columns=["genes", "LABEL"])
    y_holdout = holdout_df["LABEL"]
    
    print(f"Dataset split completed:")
    print(f" - Train Pool Size: {len(train_pool)} (Normal: {sum(y_train_full==0)}, Tumor: {sum(y_train_full==1)})")
    print(f" - Holdout Test Size: {len(holdout_df)} (Normal: {sum(y_holdout==0)}, Tumor: {sum(y_holdout==1)})\n")
    
    # 2. Feature Selection & Scaling (Fitted ONLY on training data to prevent leakage)
    NUM_FEATURES_TO_SELECT = 15
    selector = SelectKBest(score_func=f_classif, k=NUM_FEATURES_TO_SELECT)
    X_train_red = selector.fit_transform(X_train_full, y_train_full)
    X_holdout_red = selector.transform(X_holdout_full)
    
    selected_features = [X_train_full.columns[i] for i in selector.get_support(indices=True)]
    print(f"Top 15 selected genes: {selected_features}\n")
    
    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train_red)
    X_holdout_scl = scaler.transform(X_holdout_red)
    
    # 3. SMOTE (Applied ONLY to training data to prevent leakage)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scl, y_train_full)
    
    # 4. Define best tuned models
    models = {
        "SVM": SVC(C=0.05, kernel="linear", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(C=0.1, penalty="l1", solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.02, max_depth=2, subsample=0.7, colsample_bytree=0.7, eval_metric="logloss", random_state=42),
        "Regularized MLP (DL)": MLPClassifier(hidden_layer_sizes=(8,), activation="relu", alpha=10.0, solver="lbfgs", max_iter=1000, random_state=42)
    }
    
    # Model-specific thresholds optimized to improve NPV
    opt_thresholds = {
        "SVM": 0.15,
        "Logistic Regression": 0.05,
        "Random Forest": 0.10,
        "XGBoost": 0.10,
        "Regularized MLP (DL)": 0.10
    }
    
    comparison_results = {}
    plot_data = []
    
    # Setup plots
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predictions using optimized thresholds
        y_probs = model.predict_proba(X_holdout_scl)[:, 1]
        th = opt_thresholds[name]
        y_pred = (y_probs >= th).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = recall_score(y_holdout, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f2 = fbeta_score(y_holdout, y_pred, beta=2, zero_division=0)
        auc = roc_auc_score(y_holdout, y_probs)
        
        comparison_results[name] = {
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity,
            "NPV": npv,
            "F2-Score": f2,
            "ROC AUC": auc,
            "Confusion Matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        }
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_holdout, y_probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)
        
    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curves Comparison on Holdout Set (No Leakage)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_comparison.png", dpi=300)
    plt.close()
    print("\nSaved ROC curves plot to 'roc_curves_comparison.png'")
    
    # Plot Grouped Bar Chart of metrics
    metrics_list = ["Recall", "Specificity", "NPV", "F2-Score"]
    model_names = list(models.keys())
    
    x = np.arange(len(metrics_list))
    width = 0.15  # width of each bar
    
    # Define colors for each model
    colors = ["#440154", "#3b528b", "#21918c", "#5dc963", "#fde725"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(model_names):
        vals = [comparison_results[name]["Sensitivity (Recall)"] if m == "Recall" else
                comparison_results[name]["Specificity"] if m == "Specificity" else
                comparison_results[name]["NPV"] if m == "NPV" else
                comparison_results[name]["F2-Score"] for m in metrics_list]
        rects = ax.bar(x + (i - 2) * width, vals, width, label=name, color=colors[i])
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="semibold"
            )
            
    ax.set_ylabel("Score Value", fontsize=12)
    ax.set_title("Holdout Test Set Performance Metrics Comparison (NPV-Optimized Thresholds)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=12)
    ax.set_ylim([0.0, 1.15])
    ax.legend(title="Classifier Model", loc="lower right", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("metrics_comparison_bar.png", dpi=300)
    plt.close()
    print("Saved metrics comparison bar chart to 'metrics_comparison_bar.png'\n")
    
    # Display printed report
    print("="*120)
    print("                SUMMARY OF HOLDOUT SET RESULTS (TEST SIZE: 86, WITH NPV-OPTIMIZED THRESHOLDS)")
    print("="*120)
    print(f"{'Classifier Model':<25} | {'Threshold':<10} | {'Recall (Sens.)':<14} | {'Specificity':<11} | {'NPV':<8} | {'F2-Score':<8} | {'ROC AUC':<8} | {'Confusion Matrix'}")
    print("-"*120)
    for model_name, res in comparison_results.items():
        th_val = opt_thresholds[model_name]
        print(f"{model_name:<25} | {th_val:<10.2f} | {res['Sensitivity (Recall)']:.4f}         | {res['Specificity']:.4f}      | {res['NPV']:.4f} | {res['F2-Score']:.4f} | {res['ROC AUC']:.4f} | {res['Confusion Matrix']}")
    print("="*120)
    
    # Save the trained XGBoost model assets for Streamlit app use
    print("\nSaving XGBoost model assets...")
    xgb_model = models["XGBoost"]
    joblib.dump(xgb_model, "best_model.pkl")
    joblib.dump(xgb_model, "XGBoost_best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selected_features, "feature_names.pkl")
    
    # Compute feature importances
    importances = xgb_model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    
    top_features = []
    for idx in sorted_idx:
        feat_name = selected_features[idx]
        importance_score = float(importances[idx])
        top_features.append({
            "feature": feat_name,
            "importance": importance_score,
            "median_value": float(train_pool[feat_name].median()),
            "min_value": float(train_pool[feat_name].min()),
            "max_value": float(train_pool[feat_name].max())
        })
        
    metadata = {
        "best_model_name": "XGBoost",
        "best_hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.02,
            "max_depth": 2,
            "subsample": 0.7,
            "colsample_bytree": 0.7
        },
        "cv_results": {},
        "holdout_test_results": {
            "Sensitivity (Recall)": float(comparison_results["XGBoost"]["Sensitivity (Recall)"]),
            "Specificity": float(comparison_results["XGBoost"]["Specificity"]),
            "Negative Predictive Value": float(comparison_results["XGBoost"]["NPV"]),
            "F2-Score": float(comparison_results["XGBoost"]["F2-Score"]),
            "ROC AUC": float(comparison_results["XGBoost"]["ROC AUC"])
        },
        "top_features": top_features,
        "num_features": len(selected_features)
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("XGBoost assets successfully exported for application use:")
    print(" - best_model.pkl / XGBoost_best_model.pkl")
    print(" - scaler.pkl")
    print(" - feature_names.pkl")
    print(" - model_metadata.json")

if __name__ == "__main__":
    evaluate_and_visualize()


# In[ ]:




