"""
Credit Card Fraud Detection - Data Exploration & Preprocessing
Dataset: Credit Card Fraud Detection from Kaggle
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import zipfile

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================

print("Loading Credit Card Fraud Detection Dataset...")
current_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_path, '..'))
zf = zipfile.ZipFile(f'{parent_directory}/data/creditcard.csv.zip') 
df = pd.read_csv(zf.open('creditcard.csv'))

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of Features: {df.shape[1] - 1}")  # Excluding target
print(f"Number of Instances: {df.shape[0]}")

print("\n" + "="*80)
print("FIRST FEW ROWS")
print("="*80)
print(df.head())

print("\n" + "="*80)
print("DATASET INFO")
print("="*80)
print(df.info())

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing_values = df.isnull().sum()
print(f"Total missing values: {missing_values.sum()}")
if missing_values.sum() == 0:
    print("✓ No missing values found!")

print("\n" + "="*80)
print("TARGET VARIABLE DISTRIBUTION (Class)")
print("="*80)
class_counts = df['Class'].value_counts()
print(class_counts)
print(f"\nClass Distribution (%):")
print(df['Class'].value_counts(normalize=True) * 100)

fraud_percentage = (class_counts[1] / len(df)) * 100
print(f"\n⚠️  Dataset is highly imbalanced: {fraud_percentage:.2f}% fraudulent transactions")

# ============================================================================
# STEP 2: FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)

# Analyze Time feature
print("\nTime Feature:")
print(f"  Range: {df['Time'].min():.0f} - {df['Time'].max():.0f} seconds")
print(f"  Duration: {df['Time'].max() / 3600:.1f} hours")

# Analyze Amount feature
print("\nAmount Feature:")
print(f"  Range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
print(f"  Mean: ${df['Amount'].mean():.2f}")
print(f"  Median: ${df['Amount'].median():.2f}")

# Amount distribution by class
print("\nAmount Statistics by Class:")
print(df.groupby('Class')['Amount'].describe())

# ============================================================================
# STEP 3: DATA VISUALIZATION
# ============================================================================

# Class Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Class distribution bar chart
ax1 = axes[0, 0]
class_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
for i, v in enumerate(class_counts):
    ax1.text(i, v + 5000, str(v), ha='center', fontweight='bold')

# 2. Amount distribution
ax2 = axes[0, 1]
df[df['Class'] == 0]['Amount'].hist(bins=50, ax=ax2, alpha=0.7, label='Legitimate', color='green')
df[df['Class'] == 1]['Amount'].hist(bins=50, ax=ax2, alpha=0.7, label='Fraud', color='red')
ax2.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Amount ($)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_xlim([0, 500])  # Focus on smaller amounts

# 3. Time distribution
ax3 = axes[1, 0]
df[df['Class'] == 0]['Time'].hist(bins=50, ax=ax3, alpha=0.7, label='Legitimate', color='green')
df[df['Class'] == 1]['Time'].hist(bins=50, ax=ax3, alpha=0.7, label='Fraud', color='red')
ax3.set_title('Transaction Time Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Box plot for Amount by Class
ax4 = axes[1, 1]
df.boxplot(column='Amount', by='Class', ax=ax4)
ax4.set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
ax4.set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax4.set_ylabel('Amount ($)')
ax4.set_ylim([0, 300])
plt.suptitle('')  # Remove auto title

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'data_exploration.png'")

# ============================================================================
# STEP 4: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Select features for correlation (V1-V28, Time, Amount, Class)
correlation_matrix = df.corr()

# Find features most correlated with Class
class_correlation = correlation_matrix['Class'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with Fraud:")
print(class_correlation[1:11])  # Exclude Class itself

print("\nBottom 10 Features (Negative Correlation) with Fraud:")
print(class_correlation[-10:])

# Plot correlation heatmap for top correlated features
top_features = class_correlation[1:16].index.tolist()  # Top 15 features
top_features.append('Class')

plt.figure(figsize=(12, 10))
sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    square=True
)
plt.title('Correlation Matrix - Top 15 Features with Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Correlation matrix saved as 'correlation_matrix.png'")

# ============================================================================
# STEP 5: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Create a copy for preprocessing
df_processed = df.copy()

# Scale Time and Amount features (V1-V28 are already scaled)
print("\nScaling 'Time' and 'Amount' features...")
scaler_time_amount = StandardScaler()
df_processed[['Time', 'Amount']] = scaler_time_amount.fit_transform(df_processed[['Time', 'Amount']])

print("✓ Features scaled successfully")

# Separate features and target
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns ({len(X.columns)} features):")
print(list(X.columns))

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

# Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"Training fraud percentage: {(y_train.sum() / len(y_train)) * 100:.2f}%")

print(f"\nTest set class distribution:")
print(y_test.value_counts())
print(f"Test fraud percentage: {(y_test.sum() / len(y_test)) * 100:.2f}%")

# ============================================================================
# STEP 7: SAVE PREPROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING PREPROCESSED DATA")
print("="*80)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✓ X_train.csv")
print("✓ X_test.csv")
print("✓ y_train.csv")
print("✓ y_test.csv")

# Save feature names
import pickle
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("✓ feature_names.pkl")

# ============================================================================
# STEP 8: ADDITIONAL VISUALIZATIONS
# ============================================================================

# Class distribution in train/test
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Train distribution
y_train.value_counts().plot(kind='bar', ax=ax[0], color=['green', 'red'])
ax[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Class')
ax[0].set_ylabel('Count')
ax[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

# Test distribution
y_test.value_counts().plot(kind='bar', ax=ax[1], color=['green', 'red'])
ax[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Class')
ax[1].set_ylabel('Count')
ax[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

plt.tight_layout()
plt.savefig('train_test_distribution.png', dpi=300, bbox_inches='tight')
print("✓ train_test_distribution.png")

print("\n" + "="*80)
print("DATA EXPLORATION AND PREPROCESSING COMPLETE!")
print("="*80)
print("\n📊 Generated Files:")
print("  1. data_exploration.png")
print("  2. correlation_matrix.png")
print("  3. train_test_distribution.png")
print("  4. X_train.csv, X_test.csv")
print("  5. y_train.csv, y_test.csv")
print("  6. feature_names.pkl")
print("\n✅ Ready for model training!")
print("="*80)