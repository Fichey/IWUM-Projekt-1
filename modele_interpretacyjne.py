import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    log_loss,
    brier_score_loss,
    make_scorer,
    roc_curve
)
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

from transformers import *


# ===== CUSTOM METRICS =====

def calculate_ks_statistic(y_true, y_pred_proba):
    """
    Oblicza KS statistic (Kolmogorov-Smirnov).
    Maksymalna różnica między kumulatywnymi rozkładami klas pozytywnej i negatywnej.
    """
    # Rozdziel predykcje dla klasy pozytywnej i negatywnej
    pos_probs = y_pred_proba[y_true == 1]
    neg_probs = y_pred_proba[y_true == 0]
    
    # KS statistic używając scipy
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return ks_stat


def ks_scorer(estimator, X, y):
    """Scorer dla GridSearchCV (KS statistic)."""
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return calculate_ks_statistic(y, y_pred_proba)


# ===== SCORING METRICS =====

scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'neg_log_loss': 'neg_log_loss',
    'neg_brier': make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False),
    'ks_statistic': make_scorer(ks_scorer, needs_proba=True)
}


# ===== MODEL DEFINITIONS =====

def create_logistic_regression_grid():
    """
    Regresja logistyczna z opcjonalnymi spline'ami.
    """
    # Pipeline z opcjonalnymi spline'ami
    pipe = Pipeline([
        ('spline', SplineTransformer(n_knots=5, degree=3)),
        ('logistic', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    param_grid = {
        'spline__n_knots': [3, 15, 35, 75, 110],
        'spline__degree': [2, 3],
        'logistic__C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'logistic__penalty': ['l1', 'l2', 'elasticnet'],
        'logistic__solver': ['saga', 'lbfgs'],
        'logistic__class_weight': [None, 'balanced']
    }
    
    return pipe, param_grid


def create_logistic_regression_simple_grid():
    """
    Prosta regresja logistyczna bez spline'ów.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga', 'lbfgs'],
        'class_weight': [None, 'balanced']
    }
    
    return model, param_grid


def create_decision_tree_grid():
    """
    Drzewo decyzyjne z ograniczeniami monotoniczności.
    """
    model = DecisionTreeClassifier(random_state=42)
    
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [10, 15, 20],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
        'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
    }
    
    return model, param_grid


# ===== MODEL EVALUATION =====

def evaluate_model(model, X, y, model_name="Model"):
    """
    Oblicza wszystkie metryki dla danego modelu.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    results = {
        'model_name': model_name,
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'ks_statistic': calculate_ks_statistic(y, y_pred_proba),
        'log_loss': log_loss(y, y_pred_proba),
        'brier_score': brier_score_loss(y, y_pred_proba)
    }
    
    return results


def print_evaluation_results(results_dict):
    """
    Wyświetla wyniki ewaluacji w czytelnej formie.
    """
    df_results = pd.DataFrame(results_dict).T
    
    print("\n" + "="*80)
    print("WYNIKI EWALUACJI MODELI")
    print("="*80)
    print(df_results.to_string())
    print("="*80)
    
    return df_results


def select_best_model(df_results):
    """
    Wybiera najlepszy model na podstawie ważonej sumy metryk.
    
    Wagi:
    - ROC-AUC: 30%
    - PR-AUC: 25%
    - KS Statistic: 20%
    - Log Loss: 15% (odwrotnie - niższy jest lepszy)
    - Brier Score: 10% (odwrotnie - niższy jest lepszy)
    """
    df_normalized = df_results.copy()
    
    # Normalizacja metryk (wyższe = lepsze)
    for col in ['roc_auc', 'pr_auc', 'ks_statistic']:
        df_normalized[f'{col}_norm'] = (df_results[col] - df_results[col].min()) / \
                                        (df_results[col].max() - df_results[col].min())
    
    # Normalizacja metryk (niższe = lepsze) -> odwracamy
    for col in ['log_loss', 'brier_score']:
        df_normalized[f'{col}_norm'] = 1 - ((df_results[col] - df_results[col].min()) / \
                                            (df_results[col].max() - df_results[col].min()))
    
    # Obliczanie ważonej sumy
    df_normalized['weighted_score'] = (
        0.30 * df_normalized['roc_auc_norm'] +
        0.25 * df_normalized['pr_auc_norm'] +
        0.20 * df_normalized['ks_statistic_norm'] +
        0.15 * df_normalized['log_loss_norm'] +
        0.10 * df_normalized['brier_score_norm']
    )
    
    # Wybór najlepszego
    best_idx = df_normalized['weighted_score'].idxmax()
    
    print("\n" + "="*80)
    print("RANKING MODELI (weighted score)")
    print("="*80)
    print(df_normalized[['model_name', 'weighted_score']].sort_values('weighted_score', ascending=False).to_string())
    print("="*80)
    print(f"\n🏆 NAJLEPSZY MODEL: {best_idx}")
    print(f"   Weighted Score: {df_normalized.loc[best_idx, 'weighted_score']:.4f}")
    print("="*80)
    
    return best_idx, df_normalized


# ===== MAIN TRAINING PIPELINE =====

def train_all_models(X_train, y_train, X_val, y_val, cv=5):
    """
    Trenuje wszystkie modele używając GridSearchCV.
    """
    models = {
        'Logistic_Regression_Simple': create_logistic_regression_simple_grid(),
        'Logistic_Regression_Splines': create_logistic_regression_grid(),
        'Decision_Tree': create_decision_tree_grid()
    }
    
    trained_models = {}
    grid_results = {}
    
    for model_name, (model, param_grid) in models.items():
        print(f"\n{'='*80}")
        print(f"🔧 Trenowanie: {model_name}")
        print(f"{'='*80}")
        print(f"Parametry do przeszukania: {len(list(param_grid.values())[0]) if isinstance(list(param_grid.values())[0], list) else 'N/A'}")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            refit='roc_auc',  # Refitujemy względem ROC-AUC
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✅ Najlepsze parametry dla {model_name}:")
        print(grid_search.best_params_)
        print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        trained_models[model_name] = grid_search.best_estimator_
        grid_results[model_name] = grid_search
        
    return trained_models, grid_results


def main():
    """
    Główna funkcja programu.
    """
    # ===== WCZYTANIE DANYCH =====
    print("📂 Wczytywanie danych...")
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
    
    print(f"✅ Podział danych:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # ===== PREPROCESSING =====
    print(f"\n🔄 Wczytywanie pipeline...")
    pipeline = joblib.load('preprocessing_pipeline.pkl')
    
    print("⚙️ Transformacja danych...")
    X_train_transformed = pipeline.transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    X_test_transformed = pipeline.transform(X_test)
    
    print(f"✅ Transformacja zakończona!")
    print(f"   Train transformed: {X_train_transformed.shape}")
    print(f"   Val transformed: {X_val_transformed.shape}")
    print(f"   Test transformed: {X_test_transformed.shape}")
    
    # ===== TRAINING =====
    trained_models, grid_results = train_all_models(
        X_train_transformed, y_train, 
        X_val_transformed, y_val,
        cv=5
    )
    
    # ===== EVALUATION ON VALIDATION SET =====
    print("\n\n" + "="*80)
    print("📊 EWALUACJA NA ZBIORZE WALIDACYJNYM")
    print("="*80)
    
    val_results = {}
    for model_name, model in trained_models.items():
        val_results[model_name] = evaluate_model(model, X_val_transformed, y_val, model_name)
    
    df_val_results = print_evaluation_results(val_results)
    
    # ===== SELECT BEST MODEL =====
    best_model_name, df_normalized = select_best_model(df_val_results)
    best_model = trained_models[best_model_name]
    
    # ===== FINAL EVALUATION ON TEST SET =====
    print("\n\n" + "="*80)
    print(f"📊 EWALUACJA NAJLEPSZEGO MODELU ({best_model_name}) NA ZBIORZE TESTOWYM")
    print("="*80)
    
    test_results = evaluate_model(best_model, X_test_transformed, y_test, best_model_name)
    
    print("\nWyniki na zbiorze testowym:")
    for metric, value in test_results.items():
        if metric != 'model_name':
            print(f"   {metric}: {value:.4f}")
    
    # ===== SAVE BEST MODEL =====
    print("\n💾 Zapisywanie najlepszego modelu...")
    joblib.dump(best_model, f'best_model_{best_model_name}.pkl')
    print(f"✅ Model zapisany jako: best_model_{best_model_name}.pkl")
    
    # ===== SAVE ALL RESULTS =====
    print("\n💾 Zapisywanie wyników...")
    df_val_results.to_csv('model_evaluation_results.csv')
    
    # Zapisz szczegółowe wyniki grid search
    for model_name, grid in grid_results.items():
        pd.DataFrame(grid.cv_results_).to_csv(f'grid_results_{model_name}.csv')
    
    print("✅ Wyniki zapisane!")
    
    return trained_models, best_model, df_val_results, test_results


if __name__ == "__main__":
    trained_models, best_model, df_val_results, test_results = main()
