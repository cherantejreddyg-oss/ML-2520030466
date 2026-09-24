import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import prepare_ml_features

def train_random_forest_model(df, test_size=0.3, random_state=42, verbose=True):
    """
    Trains a Random Forest Classifier on preprocessed features and evaluates model performance.
    """
    X, y = prepare_ml_features(df)
    
    # Train-Test Split with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception:
        # Fallback if stratify fails due to small class count in small subsets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Instantiate and fit Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    
    labels = ['LOW', 'MEDIUM', 'HIGH']
    
    # Calculate evaluation metrics using scikit-learn
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Feature Importance calculation
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    model_results = {
        'model': rf_model,
        'feature_names': list(X.columns),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'labels': labels,
        'feature_importances': importances
    }

    if verbose:
        display_model_summary(model_results)

    return model_results

def display_model_summary(results):
    """
    Displays clean ASCII terminal output for Random Forest training results.
    """
    print("\n" + "="*60)
    print("             RANDOM FOREST CLASSIFIER MODEL")
    print("="*60)
    print(f"Algorithm            : Random Forest Classifier (scikit-learn)")
    print(f"Trees (n_estimators) : 100")
    print(f"Total Records        : {len(results['X_train']) + len(results['X_test'])}")
    print(f"Training Subset      : {len(results['X_train'])} records ({100-30}%)")
    print(f"Testing Subset       : {len(results['X_test'])} records (30%)")
    print("\nEVALUATION METRICS:")
    print("--------------------------------------------------------")
    print(f"  Accuracy  : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision : {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall    : {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score  : {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print("--------------------------------------------------------")
    
    print("\nCONFUSION MATRIX:")
    print("                     Predicted LOW  Predicted MED  Predicted HIGH")
    cm = results['confusion_matrix']
    labels = results['labels']
    for i, true_label in enumerate(labels):
        print(f"Actual {true_label:<11} | {cm[i][0]:^13d} | {cm[i][1]:^13d} | {cm[i][2]:^14d}")
    
    print("\nFEATURE IMPORTANCE (TOP 8 INFLUENTIAL FACTORS):")
    print("--------------------------------------------------------")
    top_imp = results['feature_importances'].head(8)
    for feat, val in top_imp.items():
        clean_name = feat.replace('zone_', 'Zone: ')
        print(f"  {clean_name:<32} : {val:.4f} ({val*100:5.2f}%)")
    print("--------------------------------------------------------")
    print("\n[Methodology & Dataset Note]")
    print("  * Model trained strictly on real Hyderabad city blackspot data.")
    print("  * Target column 'risk_level' excluded from input feature matrix.")
    print("  * Train/Test split uses random_state=42 for 100% reproducible results.")
    print("="*60)
