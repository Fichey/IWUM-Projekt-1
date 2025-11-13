# transformers.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




class InfinityReplacer(BaseEstimator, TransformerMixin):
    """Zamienia inf/-inf na NaN."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.replace([np.inf, -np.inf], np.nan)


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Usuwa kolumny z liczbą braków przekraczającą threshold."""
    
    def __init__(self, missing_threshold=0.95):
        self.missing_threshold = missing_threshold
    
    def fit(self, X, y=None):
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        if len(self.cols_to_drop_) > 0:
            print(f"🗑️ Zapamiętano {len(self.cols_to_drop_)} kolumn do usunięcia (braki > {self.missing_threshold*100:.0f}%)")
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.cols_to_drop_, errors='ignore')


class MissingIndicator(BaseEstimator, TransformerMixin):
    """Dodaje flagi *_missing dla kolumn z brakami."""
    
    def fit(self, X, y=None):
        self.cols_with_missing_ = X.columns[X.isnull().any()].tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.cols_with_missing_:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)
        return X


class CustomImputer(BaseEstimator, TransformerMixin):
    """Imputacja: numeryczne -> mediana, kategoryczne -> moda."""
    
    def __init__(self):
        self.imputer_num_ = None
        self.imputer_cat_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
    
    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = X.select_dtypes(include=[bool]).columns.tolist()
        self.num_cols_ = [col for col in self.num_cols_ if col not in bool_cols]
        
        self.cat_cols_ = X.select_dtypes(exclude=[np.number, np.bool_]).columns.tolist()
        
        if len(self.num_cols_) > 0:
            self.imputer_num_ = SimpleImputer(strategy="median")
            self.imputer_num_.fit(X[self.num_cols_])
        
        if len(self.cat_cols_) > 0:
            self.imputer_cat_ = SimpleImputer(strategy="most_frequent")
            self.imputer_cat_.fit(X[self.cat_cols_])
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if self.imputer_num_ is not None and len(self.num_cols_) > 0:
            X[self.num_cols_] = self.imputer_num_.transform(X[self.num_cols_])
        
        if self.imputer_cat_ is not None and len(self.cat_cols_) > 0:
            X[self.cat_cols_] = self.imputer_cat_.transform(X[self.cat_cols_])
        
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsoryzacja (obcina wartości skrajne na podstawie kwantyli)."""
    
    def __init__(self, lower_q=0.02, upper_q=0.98):
        self.lower_q = lower_q
        self.upper_q = upper_q
    
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns
        bool_cols = X.select_dtypes(include=[bool]).columns
        num_cols = [col for col in num_cols 
                    if col not in bool_cols and not col.endswith("_missing")]
        
        self.limits_ = {}
        for col in num_cols:
            lower = X[col].quantile(self.lower_q)
            upper = X[col].quantile(self.upper_q)
            self.limits_[col] = (lower, upper)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.limits_.items():
            if col in X.columns:
                X[col] = np.clip(X[col], lower, upper)
        return X


class NumericScaler(BaseEstimator, TransformerMixin):
    """Standaryzacja kolumn numerycznych (pomija bool i *_missing)."""
    
    def __init__(self):
        self.scaler_ = None
        self.num_cols_ = None
    
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = X.select_dtypes(include=[bool]).columns.tolist()
        self.num_cols_ = [col for col in num_cols 
                         if col not in bool_cols and not col.endswith("_missing")]
        
        if len(self.num_cols_) > 0:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.num_cols_])
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.scaler_ is not None and len(self.num_cols_) > 0:
            X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoding dla kolumn kategorycznych."""
    
    def __init__(self):
        self.cat_cols_ = None
        self.encoded_cols_ = None
    
    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(self.cat_cols_) > 0:
            X_encoded = pd.get_dummies(X, columns=self.cat_cols_, prefix=self.cat_cols_)
            self.encoded_cols_ = X_encoded.columns.tolist()
        else:
            self.encoded_cols_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        if len(self.cat_cols_) > 0:
            X = pd.get_dummies(X, columns=self.cat_cols_, prefix=self.cat_cols_)
            for col in self.encoded_cols_:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.encoded_cols_]
        return X


class LowVarianceDropper(BaseEstimator, TransformerMixin):
    """Usuwa kolumny o niskiej wariancji."""
    
    def __init__(self, var_threshold=0.01):
        self.var_threshold = var_threshold
    
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number, np.bool_]).columns
        variances = X[num_cols].var(numeric_only=True)
        self.low_var_cols_ = variances[variances < self.var_threshold].index.tolist()
        if len(self.low_var_cols_) > 0:
            print(f"⚠️ Zapamiętano {len(self.low_var_cols_)} kolumn o niskiej wariancji (< {self.var_threshold})")
        return self
    
    def transform(self, X):
        return X.drop(columns=self.low_var_cols_, errors='ignore')


class HighCorrelationDropper(BaseEstimator, TransformerMixin):
    """Usuwa kolumny silnie skorelowane."""
    
    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = corr_threshold

    
    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=[np.number, np.bool_]).columns
        corr_matrix = X[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.high_corr_cols_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        if len(self.high_corr_cols_) > 0:
            print(f"🔁 Zapamiętano {len(self.high_corr_cols_)} kolumn z wysoką korelacją (> {self.corr_threshold})")
        return self
    
    def transform(self, X):
        return X.drop(columns=self.high_corr_cols_, errors='ignore')




class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Transformer zamieniający zmienne numeryczne na WoE względem y (default flag).
    
    Założenia:
    - y = 1 -> 'bad' (default)
    - y = 0 -> 'good' (brak defaultu)
    
    Działa w dwóch krokach:
    1) dzieli każdą kolumnę na n_bins kwantylowych przedziałów (+ osobny bin na missing),
    2) liczy WoE dla każdego binu i zapisuje słowniki mapowań.
    """

    def __init__(self, n_bins=5, eps=0.5):
        """
        n_bins: liczba binów kwantylowych (bez binu na brak)
        eps: smoothing dodawany do liczników good/bad, żeby uniknąć WoE = +/- inf
        """
        self.n_bins = n_bins
        self.eps = eps

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y)

        # bierzemy tylko kolumny numeryczne (WoE ma sens głównie tam)
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # globalne liczebności
        self.total_good_ = (y == 0).sum()
        self.total_bad_ = (y == 1).sum()

        self.bin_edges_ = {}
        self.woe_maps_ = {}
        self.iv_ = {}

        for col in self.num_cols_:
            col_data = X[col]
            df_tmp = pd.DataFrame({"x": col_data, "y": y})

            # osobny bin na braki
            missing_mask = df_tmp["x"].isna()

            # kwantylowy binning na nie-missing
            if (~missing_mask).sum() == 0:
                # kolumna w całości pusta -> WoE = 0
                self.bin_edges_[col] = None
                self.woe_maps_[col] = {"MISSING": 0.0}
                self.iv_[col] = 0.0
                continue

            try:
                # retbins=True -> dostajemy krawędzie przedziałów
                _, bins = pd.qcut(
                    df_tmp.loc[~missing_mask, "x"],
                    q=self.n_bins,
                    duplicates="drop",
                    retbins=True
                )
            except ValueError:
                # za mało unikalnych wartości -> jeden bin
                bins = np.unique(df_tmp.loc[~missing_mask, "x"])
                if bins.size == 1:
                    bins = np.array([bins[0] - 1e-6, bins[0] + 1e-6])

            self.bin_edges_[col] = bins

            # przypisanie binów
            df_tmp["bin"] = pd.cut(
                df_tmp["x"],
                bins=bins,
                include_lowest=True
            )
            df_tmp["bin"] = df_tmp["bin"].astype(object)

            df_tmp.loc[missing_mask, "bin"] = "MISSING"

            # agregacja good/bad per bin
            grouped = df_tmp.groupby("bin")["y"]
            good = (grouped.apply(lambda s: (s == 0).sum()) + self.eps)
            bad = (grouped.apply(lambda s: (s == 1).sum()) + self.eps)

            dist_good = good / (self.total_good_ + self.eps * len(good))
            dist_bad = bad / (self.total_bad_ + self.eps * len(bad))

            woe = np.log(dist_good / dist_bad)

            # zapisujemy mapowanie: bin -> WoE
            woe_map = woe.to_dict()
            self.woe_maps_[col] = woe_map

            # policz IV tej zmiennej (przyda się później do raportu)
            iv_col = ((dist_good - dist_bad) * woe).sum()
            self.iv_[col] = iv_col

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.num_cols_:
            if col not in X.columns:
                continue

            col_data = X[col]
            bins = self.bin_edges_[col]
            woe_map = self.woe_maps_[col]

            if bins is not None:
                binned = pd.cut(
                    col_data,
                    bins=bins,
                    include_lowest=True
                ).astype(object)
            else:
                # kolumna była w całości missing przy fit
                binned = pd.Series(["MISSING"] * len(X), index=X.index, dtype=object)

            # missing -> "MISSING"
            binned[col_data.isna()] = "MISSING"

            # zamiana binów na WoE; nieznane biny -> 0.0
            X[col] = binned.map(woe_map).fillna(0.0).astype(float)

        return X



















