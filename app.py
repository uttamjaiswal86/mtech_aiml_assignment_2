import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import requests
current_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_path, '..'))

github_url = "https://raw.githubusercontent.com/uttamjaiswal86/mtech_aiml_assignment_2/refs/heads/main/sample_test_data.csv"
model_compare_csv = f'{current_path}/model/model_comparison_results.csv'


import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="2025AA05078(Uttam):ML Assignment",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .fraud-alert {
        padding: 10px;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        margin: 10px 0;
    }
    .safe-alert {
        padding: 10px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Project: Credit Card Fraud Detection ( ML Assignment - 2)")
st.markdown("""This project's application detects fraudulent credit card transactions using 6 different ML classification models. To test upload your sample transaction data (CSV) to get predictions and evaluate model performance.""")
st.markdown("---")


@st.cache_data
def fetch_github_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error("Failed to fetch data from GitHub.")
        return None

file_contents = fetch_github_data(github_url)


@st.cache_resource
def load_models():
    """Load all trained models"""
    import os
    models = {}
    model_names = {
        'Logistic Regression': 'model_logistic_regression.pkl',
        'Decision Tree': 'model_decision_tree.pkl',
        'kNN': 'model_knn.pkl',
        'Naive Bayes': 'model_naive_bayes.pkl',
        'Random Forest': 'model_random_forest.pkl',
        'XGBoost': 'model_xgboost.pkl'
    }

    possible_paths = ['', 'model/', './model/']    
    for name, filename in model_names.items():
        loaded = False
        for path in possible_paths:
            filepath = os.path.join(path, filename)
            try:
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
                    loaded = True
                    break
            except FileNotFoundError:
                continue
        
        if not loaded:
            st.warning(f"Error: Model file {filename} not found in root or model/ directory!")
    
    return models

@st.cache_data
def load_feature_names():
    """Load feature names"""
    import os
    possible_paths = ['', 'model/', './model/']
    
    for path in possible_paths:
        filepath = os.path.join(path, 'feature_names.pkl')
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue
    
    return None

# Load models and features
models = load_models()
feature_names = load_feature_names()

# Check if models were loaded successfully
if not models:
    st.error("error: No trained models found!")
    st.info("""
    Expected model files:
    - model_logistic_regression.pkl
    - model_decision_tree.pkl
    - model_knn.pkl
    - model_naive_bayes.pkl
    - model_random_forest_ensemble.pkl
    - model_xgboost_ensemble.pkl
    """)
    st.stop()

st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox(
    "Select Any ML Model",
    options=list(models.keys()),
    index=min(4, len(models) - 1)  # Default to Random Forest or last available
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Available Models")
st.sidebar.markdown("""
1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (kNN)**
4. **Naive Bayes**
5. **Random Forest** (Ensemble)
6. **XGBoost** (Ensemble)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### More about Dataset")
st.sidebar.info("""
**Credit Card Fraud Detection**
- 30 Features (V1-V28 PCA, Time, Amount)
- 100K Transactions""")

# FILE UPLOAD SECTION
st.header(" Upload Transaction Data")

# Add sample CSV download option
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your CSV file (test data recommended due to free tier limitations)",
        type=['csv'],
        help="Upload a CSV file with the same 30 features as the training data"
    )

with col2:
    st.markdown("### Need Sample Data?")
    st.download_button(
        label="Download Sample CSV",
        data=file_contents,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Download a sample CSV file with correct format (5 transactions)",
        use_container_width=True
    )
    st.caption("5 sample transactions with correct format")
st.markdown("---")

if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file).dropna()
        
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        
        # Display data preview
        with st.expander("View Data Preview"):
            st.dataframe(df.head(10))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))

        # DATA VALIDATION
        has_labels = 'Class' in df.columns
        
        if has_labels:
            st.info("Target column 'Class' detected. Will show evaluation metrics.")
            X_test = df.drop(columns=['Class'])
            y_test = df['Class']
            
            # Show class distribution
            fraud_count = sum(y_test == 1)
            legit_count = sum(y_test == 0)
            fraud_pct = (fraud_count / len(y_test)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(y_test))
            with col2:
                st.metric("Legitimate", legit_count, delta=f"{100-fraud_pct:.2f}%")
            with col3:
                st.metric("Fraudulent", fraud_count, delta=f"{fraud_pct:.2f}%", delta_color="inverse")
        else:
            st.warning("No 'Class' column found. Will only show predictions.")
            X_test = df
            y_test = None
        
        # Validate features
        if feature_names:
            missing_features = set(feature_names) - set(X_test.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                st.stop()
            
            # Ensure correct column order
            X_test = X_test[feature_names]
        
        # ============================================================================
        # MAKE PREDICTIONS
        # ============================================================================
        
        st.header("Fraud Detection Results")
        
        if st.button("Detect Fraud", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing transactions with {selected_model}..."):
                
                # Get the selected model
                model = models[selected_model]
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = None
                
                # ============================================================================
                # PREDICTION SUMMARY
                # ============================================================================
                
                st.subheader("Detection Summary")
                
                fraud_detected = sum(y_pred == 1)
                legit_detected = sum(y_pred == 0)
                fraud_rate = (fraud_detected / len(y_pred)) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analyzed", len(y_pred))
                
                with col2:
                    st.metric("Legitimate", legit_detected, 
                             delta=f"{100-fraud_rate:.2f}%")
                
                with col3:
                    st.metric("Fraudulent", fraud_detected, 
                             delta=f"{fraud_rate:.2f}%", 
                             delta_color="inverse")
                
                with col4:
                    if y_proba is not None:
                        avg_fraud_prob = y_proba[y_pred == 1].mean() if fraud_detected > 0 else 0
                        st.metric("Avg Fraud Confidence", f"{avg_fraud_prob:.2%}")
                
                st.subheader("Transaction Details")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Transaction_ID': range(1, len(y_pred) + 1),
                    'Prediction': ['FRAUD' if p == 1 else '✅ Legitimate' for p in y_pred],
                    'Prediction_Class': y_pred
                })
                
                if y_proba is not None:
                    results_df['Fraud_Probability'] = y_proba
                    results_df['Confidence'] = results_df['Fraud_Probability'].apply(
                        lambda x: f"{x:.2%}" if x > 0.5 else f"{(1-x):.2%}"
                    )
                
                if has_labels:
                    results_df['Actual_Class'] = ['Fraud' if y == 1 else 'Legitimate' for y in y_test]
                    results_df['Correct'] = ['✅' if p == a else '❌' 
                                            for p, a in zip(y_pred, y_test)]
                
                # Show flagged fraudulent transactions first
                fraud_transactions = results_df[results_df['Prediction_Class'] == 1]
                if len(fraud_transactions) > 0:
                    st.markdown("### Flagged Fraudulent Transactions")
                    st.dataframe(
                        fraud_transactions.head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.success("No fraudulent transactions detected!")
                
                # Show all transactions in expander
                with st.expander("View All Transactions"):
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name=f"fraud_predictions_{selected_model.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # ============================================================================
                # EVALUATION METRICS (if ground truth available)
                # ============================================================================
                
                if y_test is not None:
                    st.markdown("---")
                    st.header("Model Performance Evaluation")
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    mcc = matthews_corrcoef(y_test, y_pred)
                    
                    if y_proba is not None:
                        auc = roc_auc_score(y_test, y_proba)
                    else:
                        auc = None
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    
                    with col2:
                        st.metric("Recall (Sensitivity)", f"{recall:.4f}")
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    with col3:
                        if auc:
                            st.metric("AUC Score", f"{auc:.4f}")
                        st.metric("MCC Score", f"{mcc:.4f}")
                    
                    # Metric explanations
                    with st.expander("ℹUnderstanding the Metrics"):
                        st.markdown("""
                        - **Accuracy**: Overall correctness (misleading for imbalanced data)
                        - **Precision**: Of predicted frauds, how many were actually fraudulent
                        - **Recall**: Of actual frauds, how many did we catch (most important!)
                        - **F1 Score**: Harmonic mean of Precision and Recall
                        - **AUC**: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
                        - **MCC**: Matthews Correlation Coefficient (good for imbalanced data)
                        """)
                    
                    # ============================================================================
                    # CONFUSION MATRIX
                    # ============================================================================
                    
                    st.subheader("Confusion Matrix")
                    
                    cm = confusion_matrix(y_test, y_pred)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap='RdYlGn',
                            xticklabels=['Legitimate', 'Fraud'],
                            yticklabels=['Legitimate', 'Fraud'],
                            ax=ax,
                            cbar_kws={'label': 'Count'}
                        )
                        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
                        ax.set_title(f'Confusion Matrix - {selected_model}', 
                                    fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### Breakdown")
                        tn, fp, fn, tp = cm.ravel()
                        
                        st.markdown(f"""
                        **True Negatives (TN):** {tn}  
                        Correctly identified legitimate
                        
                        **False Positives (FP):** {fp}  
                        False alarms (legitimate flagged as fraud)
                        
                        **False Negatives (FN):** {fn}  
                        Missed frauds (most costly!)
                        
                        **True Positives (TP):** {tp}  
                        Correctly caught frauds
                        """)
                    
                    # ============================================================================
                    # ROC CURVE
                    # ============================================================================
                    
                    if y_proba is not None:
                        st.subheader("ROC Curve")
                        
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                               label=f'ROC curve (AUC = {auc:.4f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                               label='Random Classifier')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
                        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                                    fontsize=14, fontweight='bold')
                        ax.legend(loc="lower right")
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
                    
                    # ============================================================================
                    # CLASSIFICATION REPORT
                    # ============================================================================
                    
                    st.subheader("Detailed Classification Report")
                    
                    report = classification_report(
                        y_test, y_pred,
                        target_names=['Legitimate', 'Fraud'],
                        output_dict=True
                    )
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(
                        report_df.style.highlight_max(axis=0, 
                                                     subset=['precision', 'recall', 'f1-score']),
                        use_container_width=True
                    )
                
                st.success("Fraud detection completed successfully!")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("Please upload a CSV file to begin fraud detection")
    
    # Comprehensive sample data section
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Expected Data Format")
        st.markdown("""
        Your CSV file should contain **30 features** in the following order:
        
        **Required Columns:**
        1. **Time** - Seconds elapsed between transaction and first transaction
        2. **V1 to V28** - PCA-transformed features (anonymized for privacy)
        3. **Amount** - Transaction amount in Euros (or your currency)
        4. **Class** *(optional)* - 0 = Legitimate, 1 = Fraud (for evaluation only)
        
        **Important Notes:**
        - All features except 'Class' are required for prediction
        - Values should be numerical (no text or missing values)
        - V1-V28 features should ideally be scaled (similar to PCA output)
        - Include 'Class' column only if you want to evaluate model performance
        """)
        
        # Show expected column names
        if feature_names:
            with st.expander("View All 30 Required Column Names"):
                st.code(", ".join(feature_names))
        
        # Visual example
        st.markdown("**Example Format:**")
        example_df = pd.DataFrame({
            'Time': [0, 406],
            'V1': [-1.36, 1.19],
            'V2': [-0.07, 0.27],
            '...': ['...', '...'],
            'V28': [-0.02, 0.01],
            'Amount': [149.62, 2.69],
            'Class': ['0 (optional)', '0 (optional)']
        })
        st.dataframe(example_df, use_container_width=True)
    
    num_samples = 5
    with col2:
        st.subheader("Get Sample Data")
        st.download_button(
            label="Download Sample CSV",
            data=file_contents,
            file_name="sample_credit_card_test.csv",
            mime="text/csv",
            help=f"Download sample file with {num_samples} transactions in correct format",
            use_container_width=True,
            type="primary"
        )
        
        st.success(f"{num_samples} sample transactions")
        st.caption("Includes both legitimate and fraudulent examples")
        
        st.markdown("---")
        
        # Additional help
        st.markdown("**Need Help?** !!! Important Notes !!!")
        st.markdown("""
        - **Missing data?** All 30 features are required
        - **Wrong format?** Use the sample CSV as template
        - **Large file?** Use subset for testing (Streamlit free tier)
        - **No labels?** Omit 'Class' column for prediction only
        """)
    
    st.markdown("---")

st.markdown("---")
st.header("Model Performance Comparison")

try:
    comparison_df = pd.read_csv(model_compare_csv)
    
    st.dataframe(
        comparison_df.style.highlight_max(
            axis=0, 
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Plot comparison
    st.subheader("Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            offset = width * (i - 2)
            ax.bar(x + offset, comparison_df[metric], width, label=metric)
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Best models summary
        st.markdown("### Best Performers")
        
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_recall = comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']
        best_f1 = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
        
        st.success(f"**Best Accuracy:** {best_accuracy}")
        st.info(f"**Best Recall (Fraud Detection):** {best_recall}")
        st.warning(f"**Best F1 Score (Balanced):** {best_f1}")
        
        st.markdown("""
        ---
        **For Fraud Detection:**
        - **Recall** is most important (catch frauds)
        - **Precision** reduces false alarms
        - **F1** balances both
        """)
    
except FileNotFoundError:
    st.warning("Model comparison results not found. Train models first.")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Assignment Develoed by :- Uttam Kumar (2025AA05078)</p>
</div>
""", unsafe_allow_html=True)