import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import *
from sklearn.pipeline import Pipeline

def create_preprocessing_pipeline(
    missing_threshold=0.95,
    lower_q=0.02,
    upper_q=0.98,
    scale_numeric=True,
    var_threshold=0.01,
    corr_threshold=0.9
):
    """
    Tworzy pipeline preprocesingu z zapisywaniem parametrów.
    
    Użycie:
    -------
    # Fitowanie na train
    pipeline = create_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    
    # Transformacja test/val z tymi samymi parametrami
    X_test_transformed = pipeline.transform(X_test)
    X_val_transformed = pipeline.transform(X_val)
    """
    
    steps = [
        ('one_hot', OneHotEncoder()),
        ('inf_replacer', InfinityReplacer()),
        ('drop_high_missing', HighMissingDropper(missing_threshold=missing_threshold)),
        ('missing_indicator', MissingIndicator()),
        ('imputer', CustomImputer()),
        ('winsorizer', Winsorizer(lower_q=lower_q, upper_q=upper_q)),
        ('drop_low_variance', LowVarianceDropper(var_threshold=var_threshold)),
        ('drop_high_corr', HighCorrelationDropper(corr_threshold=corr_threshold)),
    ]
    
    if scale_numeric:
        steps.append(('scaler', NumericScaler()))
    
    return Pipeline(steps)


# ===== PRZYKŁAD UŻYCIA =====

if __name__ == "__main__":
    # Wczytanie danych
    df = pd.read_csv("zbiór_7.csv")
    X = df.drop("default", axis=1)
    y = df["default"]
    
    # Podział train/test/val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
    
    # Tworzenie pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(
        missing_threshold=0.95,
        lower_q=0.02,
        upper_q=0.98,
        scale_numeric=True,
        var_threshold=0.01,
        corr_threshold=0.9
    )
    
    # Fit na train (zapisuje wszystkie parametry)
    print("\n🔧 Fitowanie pipeline na zbiorze treningowym...")
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    
    import joblib
    joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')
    print("\n💾 Pipeline zapisany do pliku 'preprocessing_pipeline.pkl'")
