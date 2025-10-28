import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("zbiór_7.csv")

nonNumeric = df.select_dtypes(exclude=["number"]).columns
nonNumericIndex = [df.columns.get_loc(col) for col in nonNumeric]

print(nonNumericIndex)

for i in nonNumericIndex:
    col = df.columns[i]
    print(f"\nKolumna: {col}")
    print(df[col].value_counts(dropna=False))


df = pd.get_dummies(df, columns=nonNumeric, prefix=nonNumeric)


X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print(X_train.shape)  # (1800, 217)
print(X_test.shape)   # (600, 217)
print(X_val.shape)    # (600, 217)


print(X_train.head()) # 3000 wierszy, 218 kolumn
print(X_train.info()) # 215 float, 5 int, 2 object
print(X_train.describe())
print(X_train.duplicated().sum()) # 0 duplikatów
print(y_train.value_counts() / X_train.shape[0] * 100 ) # około 9% danych ma default = 1



import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def prepare_missing_features(
    df: pd.DataFrame,
    missing_threshold: float = 0.95,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    var_threshold: float = 0.01,
    scale_numeric: bool = True
) -> pd.DataFrame:
    """
    Przygotowanie danych do modelowania:
      1. Zamienia inf / -inf na NaN
      2. Usuwa kolumny z brakami > missing_threshold
      3. Dodaje flagi *_missing (1 jeśli oryginalnie brak)
      4. Imputuje:
         - numeryczne: mediana
         - kategoryczne: moda (najczęstsza wartość)
      5. Winsoryzuje kolumny numeryczne (obcina 2% dolnych i 2% górnych wartości)
      6. Usuwa kolumny o bardzo niskiej wariancji (< var_threshold)
      7. (Opcjonalnie) Standaryzuje kolumny numeryczne (średnia = 0, std = 1)
    """

    df = df.copy()

    # 1. Zamień inf i -inf na NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Usuń kolumny z nadmiarem braków
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > missing_threshold].index
    if len(to_drop) > 0:
        print(f"Usunięto {len(to_drop)} kolumn z brakami > {missing_threshold*100:.0f}%: {list(to_drop)}")
        df = df.drop(columns=to_drop)

    # 3. Flagi braków
    for col in df.columns:
        if df[col].isnull().any():
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # 4. Imputacja
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    cat_cols = df.select_dtypes(exclude=[np.number, np.bool_]).columns
    if len(cat_cols) > 0:
        print(f"Imputacja kategorycznych kolumn: {list(cat_cols)}")
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # 5. Winsoryzacja
    num_cols_after = df.select_dtypes(include=[np.number]).columns
    for col in num_cols_after:
        if col.endswith("_missing"):
            continue
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = np.clip(df[col], lower, upper)

    # 6. Usuwanie kolumn o niskiej wariancji (przed skalowaniem!)
    variances = df.var(numeric_only=True)
    low_var_cols = variances[variances < var_threshold].index.tolist()
    if len(low_var_cols) > 0:
        print(f"Usunięto {len(low_var_cols)} kolumn o niskiej wariancji < {var_threshold}: {low_var_cols}")
        df = df.drop(columns=low_var_cols)

    # 7. Skalowanie (dopiero po usunięciu kolumn o niskiej wariancji)
    if scale_numeric:
        numeric_for_scaling = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if not col.endswith("_missing")
        ]
        if len(numeric_for_scaling) > 0:
            scaler = StandardScaler()
            df[numeric_for_scaling] = scaler.fit_transform(df[numeric_for_scaling])
            print(f"Przeskalowano {len(numeric_for_scaling)} kolumn numerycznych (średnia=0, std=1).")

    print("✅ Dane gotowe: brak inf/NaN, flagi braków, winsoryzacja, usunięcie low-var i standaryzacja.")
    return df



X_train = prepare_missing_features(X_train, missing_threshold=0.95)

def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Usuwa jedną kolumnę z każdej pary cech, które są mocno skorelowane.
    
    Parametry:
    -----------
    df : pd.DataFrame
        Dane wejściowe (tylko kolumny numeryczne będą analizowane)
    threshold : float, default=0.9
        Próg korelacji (względem |r|), powyżej którego kolumny uznaje się za silnie skorelowane
    
    Zwraca:
    --------
    pd.DataFrame
        DataFrame z usuniętymi kolumnami o wysokiej korelacji
    """
    
    df = df.copy()
    
    # 1️⃣ Wybierz tylko kolumny numeryczne
    num_cols = df.select_dtypes(include=[np.number, np.bool_]).columns
    corr_matrix = df[num_cols].corr().abs()  # absolutne wartości korelacji
    
    print(corr_matrix)
    
    # 2️⃣ Utwórz maskę trójkąta górnego, żeby nie powtarzać par
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 3️⃣ Znajdź kolumny do usunięcia
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"✅ Usunięto {len(to_drop)} kolumn z wysoką korelacją (> {threshold})")
        df = df.drop(columns=to_drop)
    else:
        print("✅ Brak kolumn do usunięcia wszystkie korelacje poniżej progu.")
    
    return df

X_train = remove_highly_correlated(X_train, threshold=0.8)

print(X_train.shape)