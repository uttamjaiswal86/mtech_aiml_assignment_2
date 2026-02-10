import pandas as pd
import numpy as np
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"\nClass distribution in test set:")
print(f"  Legitimate (0): {sum(y_test == 0)}")
print(f"  Fraud (1): {sum(y_test == 1)}")


def evaluate_model(model, X_test, y_test, y_pred, model_name):
    """Calculate all required metrics"""
    
    # Get probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = 'N/A'
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': auc,
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics

results = []
trained_models = {}
confusion_matrices = {}

print("\n" + "="*80)
print("TRAINING 6 MACHINE LEARNING MODELS")
print("="*80)


print("\n[1/6] Training Logistic Regression...")
print("-" * 60)
start_time = time.time()

lr_model = LogisticRegression(
    random_state=42, 
    max_iter=1000, 
    solver='lbfgs',
    class_weight='balanced'  # Handle imbalanced data
)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

metrics_lr = evaluate_model(lr_model, X_test, y_test, y_pred_lr, 'Logistic Regression')
results.append(metrics_lr)
trained_models['Logistic Regression'] = lr_model
confusion_matrices['Logistic Regression'] = confusion_matrix(y_test, y_pred_lr)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  Accuracy: {metrics_lr['Accuracy']:.4f}")
print(f"  AUC: {metrics_lr['AUC']:.4f}")
print(f"  Recall: {metrics_lr['Recall']:.4f}")

print("\n[2/6] Training Decision Tree Classifier...")
print("-" * 60)
start_time = time.time()

dt_model = DecisionTreeClassifier(
    random_state=42, 
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced'
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

metrics_dt = evaluate_model(dt_model, X_test, y_test, y_pred_dt, 'Decision Tree')
results.append(metrics_dt)
trained_models['Decision Tree'] = dt_model
confusion_matrices['Decision Tree'] = confusion_matrix(y_test, y_pred_dt)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  Accuracy: {metrics_dt['Accuracy']:.4f}")
print(f"  AUC: {metrics_dt['AUC']:.4f}")
print(f"  Recall: {metrics_dt['Recall']:.4f}")

print("\n[3/6] Training K-Nearest Neighbors...")
print("-" * 60)
start_time = time.time()

sample_size = min(50000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_indices]
y_train_sample = y_train[sample_indices]

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',
    n_jobs=-1
)
knn_model.fit(X_train_sample, y_train_sample)
y_pred_knn = knn_model.predict(X_test)

metrics_knn = evaluate_model(knn_model, X_test, y_test, y_pred_knn, 'kNN')
results.append(metrics_knn)
trained_models['kNN'] = knn_model
confusion_matrices['kNN'] = confusion_matrix(y_test, y_pred_knn)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  (Note: Trained on {sample_size} samples for efficiency)")
print(f"  Accuracy: {metrics_knn['Accuracy']:.4f}")
print(f"  AUC: {metrics_knn['AUC']:.4f}")
print(f"  Recall: {metrics_knn['Recall']:.4f}")

print("\n[4/6] Training Naive Bayes (Gaussian)...")
print("-" * 60)
start_time = time.time()

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

metrics_nb = evaluate_model(nb_model, X_test, y_test, y_pred_nb, 'Naive Bayes')
results.append(metrics_nb)
trained_models['Naive Bayes'] = nb_model
confusion_matrices['Naive Bayes'] = confusion_matrix(y_test, y_pred_nb)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  Accuracy: {metrics_nb['Accuracy']:.4f}")
print(f"  AUC: {metrics_nb['AUC']:.4f}")
print(f"  Recall: {metrics_nb['Recall']:.4f}")

print("\n[5/6] Training Random Forest (Ensemble)...")
print("-" * 60)
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

metrics_rf = evaluate_model(rf_model, X_test, y_test, y_pred_rf, 'Random Forest (Ensemble)')
results.append(metrics_rf)
trained_models['Random Forest'] = rf_model
confusion_matrices['Random Forest'] = confusion_matrix(y_test, y_pred_rf)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  Accuracy: {metrics_rf['Accuracy']:.4f}")
print(f"  AUC: {metrics_rf['AUC']:.4f}")
print(f"  Recall: {metrics_rf['Recall']:.4f}")

print("\n[6/6] Training XGBoost (Ensemble)...")
print("-" * 60)
start_time = time.time()

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

metrics_xgb = evaluate_model(xgb_model, X_test, y_test, y_pred_xgb, 'XGBoost (Ensemble)')
results.append(metrics_xgb)
trained_models['XGBoost'] = xgb_model
confusion_matrices['XGBoost'] = confusion_matrix(y_test, y_pred_xgb)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
print(f"  Accuracy: {metrics_xgb['Accuracy']:.4f}")
print(f"  AUC: {metrics_xgb['AUC']:.4f}")
print(f"  Recall: {metrics_xgb['Recall']:.4f}")

print("\n" + "="*80)
print("MODEL COMPARISON - ALL METRICS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.round(4)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to 'model_comparison_results.csv'")

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

for model_name, model in trained_models.items():
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print('='*60)
    
    y_pred = model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=['Legitimate (0)', 'Fraud (1)'],
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrices[model_name]
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Legit  Fraud")
    print(f"Actual Legit    {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"       Fraud    {cm[1][0]:6d} {cm[1][1]:6d}")
    
    # Calculate False Positives and False Negatives
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tn = cm[0][0]
    
    print(f"\nKey Metrics:")
    print(f"  True Positives (Fraud detected): {tp}")
    print(f"  False Negatives (Fraud missed): {fn}")
    print(f"  False Positives (False alarms): {fp}")
    print(f"  True Negatives (Correctly identified legitimate): {tn}")

print("\n" + "="*80)
print("BEST MODELS BY METRIC")
print("="*80)

for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
    if results_df[metric].dtype != 'object':
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"{metric:12s}: {best_model:30s} ({best_score:.4f})")


print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

for model_name, model in trained_models.items():
    filename = f"model_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"{filename}")

# Save confusion matrices
with open('confusion_matrices.pkl', 'wb') as f:
    pickle.dump(confusion_matrices, f)
print("confusion_matrices.pkl")

print("\n" + "="*80)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*80)