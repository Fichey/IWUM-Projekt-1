import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from transformers import (
    InfinityReplacer,
    HighMissingDropper,
    MissingIndicator,
    CustomImputer,
    Winsorizer,
    LowVarianceDropper,
    HighCorrelationDropper,
    OneHotEncoder,
    NumericScaler,   # może się jeszcze przydać, na razie nie używamy
    WoETransformer   # NOWY transformer, musi być dodany w transformers.py
)

import joblib


# ========= PIPELINE DLA DRZEWA DECYZYJNEGO =========

def create_tree_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    var_threshold: float = 0.01,
    corr_threshold: float = 0.9,
) -> Pipeline:
    """
    Preprocessing pod drzewo:
    - OneHotEncoder dla zmiennych kategorycznych
    - zamiana inf na NaN
    - wyrzucenie kolumn z ogromną liczbą braków
    - dodanie wskaźników braków
    - imputacja (median / most_frequent)
    - winsoryzacja (obcięcie outlierów)
    - wyrzucenie kolumn o bardzo małej wariancji
    - wyrzucenie kolumn mocno skorelowanych
    - BEZ skalowania (drzewo go nie potrzebuje)
    """
    steps = [
        ("one_hot", OneHotEncoder()),
        ("inf_replacer", InfinityReplacer()),
        ("drop_high_missing", HighMissingDropper(missing_threshold=missing_threshold)),
        ("missing_indicator", MissingIndicator()),
        ("imputer", CustomImputer()),
        ("winsorizer", Winsorizer(lower_q=lower_q, upper_q=upper_q)),
        ("drop_low_variance", LowVarianceDropper(var_threshold=var_threshold)),
        ("drop_high_corr", HighCorrelationDropper(corr_threshold=corr_threshold)),
    ]

    return Pipeline(steps)


# ========= PIPELINE DLA REGRESJI LOGISTYCZNEJ (WoE) =========

def create_logit_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    var_threshold: float = 0.01,
    corr_threshold: float = 0.9,
    n_bins: int = 5,
) -> Pipeline:
    """
    Preprocessing pod regresję logistyczną z WoE:
    - OneHotEncoder (na razie zostawiamy, bo drzewo też go ma; można później uprościć)
    - zamiana inf na NaN
    - wyrzucenie kolumn z ogromną liczbą braków
    - dodanie wskaźników braków
    - imputacja (median / most_frequent)
    - winsoryzacja
    - wyrzucenie kolumn o bardzo małej wariancji
    - WoETransformer (binning + WoE na zmiennych numerycznych)
    - wyrzucenie kolumn mocno skorelowanych JUŻ po WoE
    - BEZ skalowania (WoE jest już na sensownej skali)
    """
    steps = [
        ("one_hot", OneHotEncoder()),
        ("inf_replacer", InfinityReplacer()),
        ("drop_high_missing", HighMissingDropper(missing_threshold=missing_threshold)),
        ("missing_indicator", MissingIndicator()),
        ("imputer", CustomImputer()),
        ("winsorizer", Winsorizer(lower_q=lower_q, upper_q=upper_q)),
        ("drop_low_variance", LowVarianceDropper(var_threshold=var_threshold)),
        ("woe", WoETransformer(n_bins=n_bins)),
        ("drop_high_corr", HighCorrelationDropper(corr_threshold=corr_threshold)),
    ]

    return Pipeline(steps)


# ========= GŁÓWNY BLOK: PODZIAŁ DANYCH + FITOWANIE PIPELINE’ÓW =========

if __name__ == "__main__":
    # 1. Wczytanie danych
    df = pd.read_csv("zbiór_7.csv")

    # Zakładamy, że kolumna celu to 'default'
    X = df.drop(columns=["default"])
    y = df["default"]

    print("🔍 Rozmiar pełnego zbioru:", X.shape)

    # 2. Podział train / temp / test (60 / 20 / 20) ze stałym random_state
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    print("📚 Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # 3. Tworzymy oba pipeline’y
    tree_pipeline = create_tree_preprocessing_pipeline(
        missing_threshold=0.95,
        lower_q=0.02,
        upper_q=0.98,
        var_threshold=0.01,
        corr_threshold=0.9,
    )

    logit_pipeline = create_logit_preprocessing_pipeline(
        missing_threshold=0.95,
        lower_q=0.02,
        upper_q=0.98,
        var_threshold=0.01,
        corr_threshold=0.9,
        n_bins=5,
    )

    # 4. Fitujemy pipeline’y na zbiorze treningowym
    print("\n🌳 Fitowanie pipeline’u dla drzewa na zbiorze treningowym...")
    X_train_tree = tree_pipeline.fit_transform(X_train, y_train)
    print("   ➜ Kształt po przetworzeniu (drzewo):", X_train_tree.shape)

    print("\n📈 Fitowanie pipeline’u dla logitu (WoE) na zbiorze treningowym...")
    X_train_logit = logit_pipeline.fit_transform(X_train, y_train)
    print("   ➜ Kształt po przetworzeniu (logit+WoE):", X_train_logit.shape)

    # 5. Zapisujemy pipeline’y do plików
    joblib.dump(tree_pipeline, "preprocessing_tree.pkl")
    joblib.dump(logit_pipeline, "preprocessing_logit_woe.pkl")

    print("\n💾 Zapisano pipeline’y:")
    print("   - preprocessing_tree.pkl")
    print("   - preprocessing_logit_woe.pkl")
