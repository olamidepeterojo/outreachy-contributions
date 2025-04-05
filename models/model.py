import os
import ast
import requests
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay,  ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, auc
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from chembl_webresource_client.new_client import new_client

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
RDKIT_FEATURES = [
    Descriptors.MolWt,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.TPSA,
    Descriptors.MolLogP
]


def parse_embedding(x):
    """
    Parse the string in x using ast.literal_eval.
    If the parsed value is a dict, try to extract a numeric embedding.
    First check for key "embedding"; if missing, check for "outcome".
    Otherwise, raise an error.
    """
    val = ast.literal_eval(x)
    if isinstance(val, dict):
        if "embedding" in val:
            return val["embedding"]
        elif "outcome" in val:
            return val["outcome"]
        else:
            raise ValueError(f"Missing embedding/outcome key: {val}")
    return val

# Get script directory and build paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'SARSCoV2_3CLPro_Diamond.csv')
features_file = os.path.join(script_dir, '..', 'notebooks', 'features.csv')

# Load data
df_labels = pd.read_csv(data_file, delimiter="\t")
df_features = pd.read_csv(features_file)

if len(df_labels) != len(df_features):
    raise ValueError("Mismatch between labels and features")

# Map target variable from df_labels into df_features
df_features['Y'] = df_labels['Y']


def compute_morgan(smiles):
    """Compute Morgan fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS))
    return [0]*MORGAN_BITS

def compute_rdkit(smiles):
    """Compute RDKit descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [desc(mol) for desc in RDKIT_FEATURES]
    return [np.nan]*len(RDKIT_FEATURES)

# Process features
X_morgan = np.stack(df_features['output'].apply(parse_embedding))
X_rdkit = np.array([compute_rdkit(s) for s in df_labels['Drug']])
X = np.hstack([X_morgan, X_rdkit])
y = df_features['Y']

# Handle missing values
X = pd.DataFrame(X).fillna(X.mean()).values

# Handle class imbalance with SMOTE
smote = SMOTETomek(sampling_strategy=0.5,random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

imbalance_ratio = len(y_res[y_res == 0]) / len(y_res[y_res == 1])

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=len(y_res[y_res==0])/len(y_res[y_res==1]),
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8
    )
}

# Train and evaluate
results = {}
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    
    # Evaluation metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    threshold = 0.5
    y_pred_adjusted = (y_proba >= threshold).astype(int)
    # Print metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    print(f"PR-AUC: {auc(recall, precision):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_adjusted))
    
    # Create a figure for visualizations
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=axs[0], cmap='Blues')
    axs[0].set_title(f'{name} Confusion Matrix')
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axs[1])
    axs[1].set_title(f'{name} ROC Curve')
    axs[1].plot([0, 1], [0, 1], 'k--')
    
    # PR Curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axs[2])
    axs[2].set_title(f'{name} PR Curve')
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[-20:]
        plt.barh(range(len(sorted_idx)), importances[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f'Feature {i}' for i in sorted_idx])
        plt.title(f"{name} Feature Importance")
        plt.show()

# ChEMBL Validation
def process_chembl_data():
    """Retrieve and preprocess ChEMBL bioactivity data"""
    target = new_client.target.search('SARS-CoV-2 3CL Protease')[0]
    activities = new_client.activity.filter(
        target_chembl_id=target['target_chembl_id'],
        standard_type__in=['IC50','Ki'],
        relation='=',
        standard_units='nM'
    ).filter(standard_value__isnull=False)
    
    df_chembl = pd.DataFrame(activities)
    df_chembl = df_chembl[
        (df_chembl['canonical_smiles'].notna()) &
        (df_chembl['standard_value'].notna())
    ]
    
    # Binarize activity (using 1000nM cutoff)
    df_chembl['Y'] = df_chembl['standard_value'].astype(float) < 1000
    return df_chembl

def validate_on_chembl(model):
    """Full validation pipeline for ChEMBL data"""
    print("\n=== ChEMBL Validation ===")
    
    # Get and process data
    df_chembl = process_chembl_data()
    
    # Generate features
    X_chembl_morgan = np.array([compute_morgan(s) for s in df_chembl['canonical_smiles']])
    X_chembl_rdkit = np.array([compute_rdkit(s) for s in df_chembl['canonical_smiles']])
    X_chembl = np.hstack([X_chembl_morgan, X_chembl_rdkit])
    y_chembl = df_chembl['Y'].astype(int)
    
    # Handle missing values
    X_chembl = pd.DataFrame(X_chembl).fillna(X.mean()).values
    
    # Predict and evaluate
    y_pred = model.predict(X_chembl)
    y_proba = model.predict_proba(X_chembl)[:, 1]
    
    # Print metrics
    print(f"ROC-AUC: {roc_auc_score(y_chembl, y_proba):.3f}")
    precision, recall, _ = precision_recall_curve(y_chembl, y_proba)
    print(f"PR-AUC: {auc(recall, precision):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_chembl, y_pred))
    
    # Create a figure for visualizations
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_chembl, y_pred, ax=axs[0], cmap='Reds')
    axs[0].set_title('ChEMBL Confusion Matrix')
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_chembl, y_proba, ax=axs[1])
    axs[1].set_title('ChEMBL ROC Curve')
    axs[1].plot([0, 1], [0, 1], 'k--')
    
    # PR Curve
    PrecisionRecallDisplay.from_predictions(y_chembl, y_proba, ax=axs[2])
    axs[2].set_title('ChEMBL PR Curve')
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(y_proba[y_chembl==0], bins=20, alpha=0.5, label='Inactive')
    plt.hist(y_proba[y_chembl==1], bins=20, alpha=0.5, label='Active')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('ChEMBL Probability Distribution')
    plt.legend()
    plt.show()

# Validate best model on ChEMBL data
best_model = models["XGBoost"]
validate_on_chembl(best_model)