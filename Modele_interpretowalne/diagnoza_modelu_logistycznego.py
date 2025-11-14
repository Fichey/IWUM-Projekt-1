import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ============================================================
#                KONFIGURACJA ŚCIEŻEK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Modele_interpretowalne
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INTERP_DIR = os.path.join(BASE_DIR, "interpretowalnosc_logit")
WOE_PLOTS_DIR = os.path.join(INTERP_DIR, "woe_profils")

os.makedirs(INTERP_DIR, exist_ok=True)
os.makedirs(WOE_PLOTS_DIR, exist_ok=True)


# ============================================================
#                   POMOCNICZE FUNKCJE
# ============================================================

def load_data():
    """Wczytuje dane i robi podział 60/20/20 jak w innych skryptach."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_models_and_preproc():
    """Ładuje logit i pipeline WoE."""
    logit_path = os.path.join(MODELS_DIR, "best_logistic_regression_woe.pkl")
    preproc_logit_path = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")

    if not os.path.exists(logit_path):
        raise FileNotFoundError(f"Brak modelu logitu: {logit_path}")
    if not os.path.exists(preproc_logit_path):
        raise FileNotFoundError(f"Brak pipeline'u logitowego: {preproc_logit_path}")

    logit = joblib.load(logit_path)
    preproc_logit = joblib.load(preproc_logit_path)

    return logit, preproc_logit


def to_dataframe(X, col_names):
    """Upewniamy się, że mamy DataFrame z nazwami kolumn."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=col_names)


def sign_from_corr(c):
    if pd.isna(c):
        return "nan"
    if c > 0:
        return "positive"
    elif c < 0:
        return "negative"
    else:
        return "zero"


def safe_filename(name: str) -> str:
    """Czyści nazwę zmiennej do użycia w nazwie pliku."""
    bad_chars = ['/', '\\', ' ', '%', ':', ';', ',']
    safe = name
    for ch in bad_chars:
        safe = safe.replace(ch, "_")
    return safe


# ============================================================
#          INFORMATION VALUE (IV) NA PODSTAWIE WoE
# ============================================================

def compute_iv_for_feature(woe_series: pd.Series, y: pd.Series) -> float:
    """
    Liczy IV dla jednej cechy, zakładając że woe_series zawiera wartości WoE
    (jedna wartość WoE = jeden bin).
    """
    df = pd.DataFrame({"woe": woe_series, "y": y})
    # liczba good/bad w każdym binie WoE
    grp = df.groupby("woe")["y"].agg(
        bad="sum",
        total="size"
    ).reset_index()
    grp["good"] = grp["total"] - grp["bad"]

    total_bad = grp["bad"].sum()
    total_good = grp["good"].sum()

    # zabezpieczenie przed dzieleniem przez zero
    if total_bad == 0 or total_good == 0:
        return np.nan

    grp["p_bad"] = grp["bad"] / total_bad
    grp["p_good"] = grp["good"] / total_good

    # klasyczna formuła IV
    # IV = Σ (p_good - p_bad) * WoE
    grp["iv_term"] = (grp["p_good"] - grp["p_bad"]) * grp["woe"]
    iv = grp["iv_term"].sum()
    return float(iv)


def compute_iv_for_all_features(X_woe: pd.DataFrame, y: pd.Series) -> dict:
    """
    Liczy IV dla wszystkich kolumn w X_woe.
    Zwraca dict: {feature_name: IV}.
    """
    iv_dict = {}
    for col in X_woe.columns:
        iv_dict[col] = compute_iv_for_feature(X_woe[col], y)
    return iv_dict


# ============================================================
#               VIF – MULTICOLINEARITY DIAGNOSTIC
# ============================================================

def compute_vif_for_features(X_woe: pd.DataFrame) -> pd.DataFrame:
    """
    Liczy VIF dla każdej cechy.
    VIF_j = 1 / (1 - R^2_j), gdzie R^2_j pochodzi z regresji
    cechy j na pozostałe cechy.
    """
    features = list(X_woe.columns)
    vif_values = []

    for i, feat in enumerate(features):
        y = X_woe[feat].values
        X_other = X_woe.drop(columns=[feat]).values

        # żeby uniknąć problemów, jeśli któraś kolumna jest stała
        try:
            reg = LinearRegression()
            reg.fit(X_other, y)
            r2 = reg.score(X_other, y)
            if r2 >= 1.0:
                vif = np.inf
            else:
                vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = np.nan

        vif_values.append(vif)

    df_vif = pd.DataFrame({
        "feature": features,
        "vif": vif_values
    })

    return df_vif


# ============================================================
#                   WYKRESY PROFILU WoE
# ============================================================

def plot_woe_profile(feature, X_woe, y, save_dir=WOE_PLOTS_DIR):
    """
    Rysuje profil WoE -> częstość defaultów dla jednej cechy
    (operujemy na danych PO WoE: X_woe[feature]).
    """
    y = pd.Series(y).reset_index(drop=True)
    col = pd.Series(X_woe[feature]).reset_index(drop=True)

    df = pd.DataFrame({"woe": col, "y": y})
    # grupujemy po wartościach WoE (to już są "biny" po transformacji)
    grp = (
        df.groupby("woe")["y"]
        .agg(count="size", bad_rate="mean")
        .reset_index()
        .sort_values("woe")
    )

    if grp.shape[0] < 2:
        # za mało punktów, żeby miało sensowny wykres
        return

    safe_name = safe_filename(feature)

    plt.figure(figsize=(7, 5))
    # oś X: wartości WoE (ciągłe), oś Y: częstość defaultów
    plt.plot(grp["woe"], grp["bad_rate"], "o-", label="default rate")
    # pozioma linia średniej częstości defaultów na trainie
    overall_bad = y.mean()
    plt.axhline(overall_bad, linestyle="--", alpha=0.6,
                label=f"Średni default (train) = {overall_bad:.3f}")
    plt.axvline(0.0, linestyle=":", alpha=0.6, label="WoE = 0")

    plt.xlabel("Wartość WoE (po binningu)")
    plt.ylabel("Częstość defaultów w binie")
    plt.title(f"Profil WoE – {feature}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"woe_profile_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_woe_profiles_for_positive_betas(df_positive, X_train_woe, y_train,
                                         top_n=10):
    """
    Dla cech z dodatnim beta rysuje profile WoE -> default rate.
    Domyślnie dla TOP_N wg |beta|.
    """
    if df_positive.empty:
        print("Brak dodatnich bet – nie ma czego rysować.")
        return

    top = df_positive.sort_values("abs_beta", ascending=False).head(top_n)

    print(f"\n🖼️ Rysuję profile WoE dla {top.shape[0]} cech z dodatnim beta...")
    for _, row in top.iterrows():
        feat = row["feature"]
        print(f"   • {feat} (beta={row['beta']:.4f})")
        if feat in X_train_woe.columns:
            plot_woe_profile(feat, X_train_woe, y_train, save_dir=WOE_PLOTS_DIR)
        else:
            print(f"     ⚠ Kolumna {feat} nie występuje w X_train_woe – pomijam.")


# ============================================================
#               GŁÓWNA DIAGNOSTYKA LOGITU
# ============================================================

def diagnose_logit():
    print("📂 Wczytywanie danych...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("📦 Wczytywanie modelu logitowego i pipeline'u...")
    logit, preproc_logit = load_models_and_preproc()

    # --- nazwy cech po pipeline ---
    if hasattr(logit, "feature_names_in_"):
        feature_names = np.array(logit.feature_names_in_)
    else:
        X_tmp = preproc_logit.transform(X_train.iloc[:5])
        if isinstance(X_tmp, pd.DataFrame):
            feature_names = np.array(X_tmp.columns)
        else:
            feature_names = np.array([f"x_{i}" for i in range(X_tmp.shape[1])])

    # --- transformacja traina przez pipeline WoE ---
    print("🔁 Transformacja X_train przez pipeline logitowy (WoE)...")
    X_train_woe = preproc_logit.transform(X_train)
    X_train_woe = to_dataframe(X_train_woe, feature_names)

    # --- bety z modelu ---
    coef = logit.coef_.ravel()
    intercept = float(logit.intercept_[0])

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "beta": coef,
    })
    df_coef["abs_beta"] = df_coef["beta"].abs()
    df_coef["beta_sign"] = np.where(
        df_coef["beta"] > 0, "positive",
        np.where(df_coef["beta"] < 0, "negative", "zero")
    )
    df_coef["odds_ratio"] = np.exp(df_coef["beta"])

    # --- korelacje cech (po WoE) z targetem ---
    print("📈 Liczenie korelacji cech (po WoE) z targetem...")
    y_series = pd.Series(y_train)

    pearson_corrs = {}
    spearman_corrs = {}

    for col in X_train_woe.columns:
        col_data = X_train_woe[col]
        try:
            pearson_corrs[col] = col_data.corr(y_series, method="pearson")
        except Exception:
            pearson_corrs[col] = np.nan

        try:
            spearman_corrs[col] = col_data.corr(y_series, method="spearman")
        except Exception:
            spearman_corrs[col] = np.nan

    df_coef["corr_pearson"] = df_coef["feature"].map(pearson_corrs)
    df_coef["corr_spearman"] = df_coef["feature"].map(spearman_corrs)

    df_coef["corr_sign_pearson"] = df_coef["corr_pearson"].apply(sign_from_corr)
    df_coef["corr_sign_spearman"] = df_coef["corr_spearman"].apply(sign_from_corr)

    # --- flaga "niespójności" beta vs korelacja ---
    df_coef["inconsistent_pearson"] = np.where(
        (df_coef["beta"] > 0) & (df_coef["corr_pearson"] < 0), True,
        np.where((df_coef["beta"] < 0) & (df_coef["corr_pearson"] > 0), True, False)
    )

    df_coef["inconsistent_spearman"] = np.where(
        (df_coef["beta"] > 0) & (df_coef["corr_spearman"] < 0), True,
        np.where((df_coef["beta"] < 0) & (df_coef["corr_spearman"] > 0), True, False)
    )

    # --- IV dla każdej cechy ---
    print("📊 Liczenie Information Value (IV) dla cech...")
    iv_dict = compute_iv_for_all_features(X_train_woe, y_series)
    df_coef["iv"] = df_coef["feature"].map(iv_dict)

    # --- VIF dla cech ---
    print("📊 Liczenie VIF (kolinearność) dla cech...")
    df_vif = compute_vif_for_features(X_train_woe)
    df_coef = df_coef.merge(df_vif, on="feature", how="left")

    # --- tylko dodatnie bety ---
    df_positive = df_coef[df_coef["beta"] > 0].copy()
    df_positive = df_positive.sort_values("abs_beta", ascending=False)

    # --- statystyka podsumowująca ---
    n_total = len(df_coef)
    n_pos = (df_coef["beta"] > 0).sum()
    n_neg = (df_coef["beta"] < 0).sum()
    n_zero = (df_coef["beta"] == 0).sum()

    print("\n============================")
    print("   DIAGNOZA LOGITU (BETA vs KORELACJE, IV, VIF)")
    print("============================")
    print(f"Intercept (β0): {intercept:.4f}")
    print(f"Liczba cech po WoE: {n_total}")
    print(f"  • beta > 0 : {n_pos}")
    print(f"  • beta < 0 : {n_neg}")
    print(f"  • beta = 0 : {n_zero}")

    print("\n👉 Dodatnie bety (TOP 15 wg |beta|) z IV i VIF:")
    if len(df_positive) == 0:
        print("   Brak dodatnich współczynników — super.")
    else:
        cols_show = ["feature", "beta", "abs_beta",
                     "corr_pearson", "iv", "vif"]
        print(df_positive[cols_show].head(15).to_string(index=False))

    # Ile dodatnich betów ma "zły" kierunek względem korelacji?
    pos_inconsistent_pearson = df_positive["inconsistent_pearson"].sum()
    pos_inconsistent_spearman = df_positive["inconsistent_spearman"].sum()

    print("\n⚠️ Wśród cech z beta > 0:")
    print(f"   • niespójne z korelacją (Pearson):  {pos_inconsistent_pearson}/{len(df_positive)}")
    print(f"   • niespójne z korelacją (Spearman): {pos_inconsistent_spearman}/{len(df_positive)}")

    # kilka statystyk IV i VIF dla dodatnich bet
    if len(df_positive) > 0:
        print("\n📌 Podsumowanie IV dla cech z beta > 0:")
        print(df_positive["iv"].describe())

        print("\n📌 Podsumowanie VIF dla cech z beta > 0:")
        print(df_positive["vif"].describe())

    # --- zapisy do plików ---
    all_path = os.path.join(INTERP_DIR, "diag_logit_all_coeffs_corr_iv_vif.csv")
    pos_path = os.path.join(INTERP_DIR, "diag_logit_positive_betas_iv_vif.csv")

    df_coef.sort_values("abs_beta", ascending=False).to_csv(all_path, index=False)
    df_positive.to_csv(pos_path, index=False)

    print(f"\n💾 Zapisano pełną tabelę do: {all_path}")
    print(f"💾 Zapisano dodatnie bety do: {pos_path}")

    # --------------------------------------------------------
    #           PROFILE WoE DLA β > 0 (TOP 10)
    # --------------------------------------------------------
    plot_woe_profiles_for_positive_betas(
        df_positive=df_positive,
        X_train_woe=X_train_woe,
        y_train=y_train,
        top_n=10,
    )

    print(f"\n📁 Wykresy profili WoE zapisano w: {WOE_PLOTS_DIR}")

        # --------------------------------------------------------
    #   WYBÓR CECH DO USUNIĘCIA (NA POTRZEBY NOWEGO PIPELINE'U)
    # --------------------------------------------------------
    # Kryteria:
    #  - beta > 0
    #  - oraz (IV < 0.05 lub VIF > 10 lub niespójność z korelacją)
    drop_mask = (
        (df_coef["beta"] > 0)
        & (
            (df_coef["iv"] < 0.05)
            | (df_coef["vif"] > 10)
            | (df_coef["inconsistent_pearson"])
            | (df_coef["inconsistent_spearman"])
        )
    )

    df_drop = df_coef[drop_mask].copy().sort_values("abs_beta", ascending=False)

    drop_path = os.path.join(INTERP_DIR, "logit_features_to_drop.csv")
    df_drop.to_csv(drop_path, index=False)

    print("\n🧹 PROPOZYCJA CECH DO USUNIĘCIA Z MODELU LOGITOWEGO")
    print(f"   Liczba cech do usunięcia: {df_drop.shape[0]} / {df_coef.shape[0]}")
    if df_drop.shape[0] > 0:
        print("\n   Top 15 cech do usunięcia (wg |beta|):")
        print(
            df_drop[
                [
                    "feature",
                    "beta",
                    "abs_beta",
                    "iv",
                    "vif",
                    "corr_pearson",
                    "inconsistent_pearson",
                    "inconsistent_spearman",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    print(f"\n💾 Zapisano listę cech do usunięcia w: {drop_path}")



# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    diagnose_logit()
