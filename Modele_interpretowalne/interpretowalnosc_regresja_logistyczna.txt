import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
#                KONFIGURACJA ŚCIEŻEK
# ============================================================

# Ten plik zakładamy, że leży w:
#   .../IWUM-Projekt-1/Modele_interpretowalne/interpretowalnosc_regresja_logistyczna.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Modele_interpretowalne
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

MODELS_DIR = os.path.join(BASE_DIR, "models")
INTERP_DIR = os.path.join(BASE_DIR, "interpretowalnosc_logit")

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
PREPROC_LOGIT_PATH = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")

# podfoldery na wykresy
WYKRESY_DIR = os.path.join(INTERP_DIR, "waznosc_cech")
PDP_DIR = os.path.join(INTERP_DIR, "PDP")
ICE_DIR = os.path.join(INTERP_DIR, "ICE")

os.makedirs(INTERP_DIR, exist_ok=True)
os.makedirs(WYKRESY_DIR, exist_ok=True)
os.makedirs(PDP_DIR, exist_ok=True)
os.makedirs(ICE_DIR, exist_ok=True)

INTERP_LOCAL_DIR = os.path.join(INTERP_DIR, "interpretowalnosc_lokalna")
os.makedirs(INTERP_LOCAL_DIR, exist_ok=True)

# ============================================================
#                ANALIZA WSPÓŁCZYNNIKÓW LOGITU
# ============================================================

def load_logit_model():
    model_path = os.path.join(MODELS_DIR, "best_logistic_regression_woe.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu logitu pod ścieżką: {model_path}")
    logit = joblib.load(model_path)
    return logit


def load_logit_preprocessor():
    if not os.path.exists(PREPROC_LOGIT_PATH):
        raise FileNotFoundError(f"Nie znaleziono pipeline'u logitu: {PREPROC_LOGIT_PATH}")
    return joblib.load(PREPROC_LOGIT_PATH)


def extract_coefficients(logit):
    """
    Zwraca DataFrame z:
        - nazwą cechy
        - beta
        - |beta|
        - znakiem
        - odds_ratio = exp(beta)
    """
    coef = logit.coef_.ravel()
    intercept = float(logit.intercept_[0])

    if hasattr(logit, "feature_names_in_"):
        feature_names = np.array(logit.feature_names_in_)
    else:
        feature_names = np.array([f"x_{i}" for i in range(len(coef))])

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "beta": coef,
    })
    df_coef["abs_beta"] = df_coef["beta"].abs()
    df_coef["sign"] = np.where(
        df_coef["beta"] > 0, "positive",
        np.where(df_coef["beta"] < 0, "negative", "zero")
    )
    df_coef["odds_ratio"] = np.exp(df_coef["beta"])

    df_coef = df_coef.sort_values("abs_beta", ascending=False).reset_index(drop=True)
    return df_coef, intercept


def summarize_signs(df_coef, intercept):
    n_total = len(df_coef)
    n_pos = (df_coef["sign"] == "positive").sum()
    n_neg = (df_coef["sign"] == "negative").sum()
    n_zero = (df_coef["sign"] == "zero").sum()

    print("\n============================")
    print("   PODSUMOWANIE WSPÓŁCZYNNIKÓW LOGITU")
    print("============================")
    print(f"Intercept (β0): {intercept:.4f}")
    print(f"Liczba cech: {n_total}")
    print(f"  • beta > 0  (positive): {n_pos}")
    print(f"  • beta < 0  (negative): {n_neg}")
    print(f"  • beta = 0  (zero):     {n_zero}")

    if n_pos == 0 and n_neg > 0:
        print("\n✅ Wszystkie niezerowe bety są ujemne – kierunek wpływu jest spójny z WoE.")
    elif n_neg == 0 and n_pos > 0:
        print("\n⚠️ Wszystkie niezerowe bety są dodatnie – to oznacza odwrotną konwencję WoE.")
    else:
        print("\n⚠️ Mamy mieszane znaki beta – warto sprawdzić, które cechy mają 'dziwny' kierunek.")
        print("   (np. problem z binningiem, korelacjami lub zmiennymi pomocniczymi).")


def save_coefficients(df_coef):
    out_path = os.path.join(INTERP_DIR, "coefficients_logit.csv")
    df_coef.to_csv(out_path, index=False)
    print(f"\n💾 Zapisano tabelę współczynników do: {out_path}")


# ============================================================
#                POBRANIE I PRZETWORZENIE DANYCH
# ============================================================

def load_and_prepare_data(preproc_logit, logit):
    """Wczytuje pełne dane, dzieli na X, y, przepuszcza przez pipeline WoE
    i zwraca (X_woe_df, y_series)."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    X_woe = preproc_logit.transform(X)

    # wymuszamy DataFrame z nazwami cech jak w logit
    feature_names = logit.feature_names_in_
    X_woe_df = pd.DataFrame(X_woe, columns=feature_names)

    return X_woe_df, y


# ============================================================
#                       PROFILE WoE
# ============================================================

def plot_woe_profile(X_woe, y, feature, save_path):
    """
    Tworzy profil WoE dla danej cechy:
       - oś X: wartość WoE (po binningu)
       - oś Y: częstość defaultów w danym binie
       - linia przerywana: średni default rate
       - pionowa linia w WoE = 0
    """
    df_tmp = pd.DataFrame({
        "woe": X_woe[feature],
        "y": y
    })

    # grupujemy po unikalnych wartościach WoE (to de facto biny)
    grp = (
        df_tmp
        .groupby("woe")
        .agg(events=("y", "sum"), total=("y", "count"))
        .reset_index()
        .sort_values("woe")
    )
    grp["dr"] = grp["events"] / grp["total"]

    mean_dr = y.mean()

    plt.figure(figsize=(7, 5))
    plt.plot(grp["woe"], grp["dr"], "o-", label="default rate")
    plt.axhline(mean_dr, color="tab:blue", linestyle="--",
                label=f"Średni default (train) = {mean_dr:.3f}")
    plt.axvline(0.0, color="tab:blue", linestyle=":",
                label="WoE = 0")

    plt.xlabel("Wartość WoE (po binningu)")
    plt.ylabel("Częstość defaultów w binie")
    plt.title(f"Profil WoE – {feature}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   ➜ zapisano profil WoE: {save_path}")


def generate_woe_profiles(df_coef, X_woe, y):
    """Rysuje:
       • profil dla jednej cechy z beta > 0 (jeśli istnieje),
       • profile dla 5 cech z największym |beta| i znakiem negative.
    """
    # cecha z dodatnią betą (jeśli jest)
    df_pos = df_coef[df_coef["sign"] == "positive"]
    if len(df_pos) > 0:
        pos_feat = df_pos.iloc[0]["feature"]
        save_path = os.path.join(
            INTERP_DIR,"woe_profils", f"woe_profile_positive_beta_{pos_feat}.png"
        )
        print(f"\n📈 Profil WoE dla cechy z dodatnią betą: {pos_feat}")
        plot_woe_profile(X_woe, y, pos_feat, save_path)
    else:
        print("\nℹ️ Brak cech z dodatnią betą – nie rysuję osobnego profilu dla beta > 0.")

    # top-5 cech z ujemną betą wg |beta|
    df_neg = df_coef[df_coef["sign"] == "negative"].head(5)
    print("\n📈 Profile WoE dla 5 cech z największym |beta| (beta < 0):")
    for feat in df_neg["feature"]:
        save_path = os.path.join(INTERP_DIR,"woe_profils", f"woe_profile_top_negative_{feat}.png")
        plot_woe_profile(X_woe, y, feat, save_path)

# ============================================================
#      DIAGNOSTYKA LICZEBNOŚCI BINÓW DLA PROFILI WOE
# ============================================================

from sklearn.model_selection import train_test_split


def get_train_split():
    """
    Odwzorowuje dokładnie ten sam podział 60/20/20,
    którego używaliśmy do trenowania modeli.
    Zwraca X_train, y_train.
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    # Val i test są nam tutaj niepotrzebne – patrzymy tylko na rozkład w trainie
    return X_train, y_train


def load_woe_transformer():
    """
    Ładuje pipeline preprocessingowy dla logitu i wyciąga z niego krok 'woe'.
    """
    preproc_path = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Nie znaleziono pipeline'u WoE pod ścieżką: {preproc_path}")

    preproc = joblib.load(preproc_path)
    if "woe" not in preproc.named_steps:
        raise ValueError("W pipeline'ie nie ma kroku o nazwie 'woe'.")

    return preproc.named_steps["woe"]


def compute_bin_table_for_feature(feature, X_train, y_train, woe_tr, min_count=50):
    """
    Buduje tabelę:
        bin (wg granic z WoE),
        count_total,
        count_good,
        count_bad,
        default_rate
    i zwraca ją jako DataFrame.
    """

    if feature not in woe_tr.bin_edges_:
        raise KeyError(f"Brak granic binów dla cechy '{feature}' w woe.bin_edges_")

    edges = np.array(woe_tr.bin_edges_[feature])

    # Dzielimy X_train na biny według granic z WoE
    bins = pd.cut(
        X_train[feature],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    # Tabelka liczności good/bad
    ctab = pd.crosstab(bins, y_train).rename(columns={0: "good", 1: "bad"})
    if "good" not in ctab.columns:
        ctab["good"] = 0
    if "bad" not in ctab.columns:
        ctab["bad"] = 0

    ctab["total"] = ctab["good"] + ctab["bad"]
    ctab["default_rate"] = ctab["bad"] / ctab["total"].replace(0, np.nan)

    ctab = ctab.reset_index().rename(columns={X_train[feature].name: "bin"})

    # Flaga małej liczności
    ctab["low_count_flag"] = ctab["total"] < min_count

    return ctab

def diagnose_bin_sizes(df_coef, n_top=5, min_count=50):
    """
    Sprawdza, czy dziwne zachowanie na końcach profili WoE
    może wynikać z bardzo małej liczności skrajnych binów.

    Dla:
      - top n_top cech wg |beta| (ujemne),
      - wszystkich cech z beta > 0
    zapisuje tabelki liczności do CSV.
    """

    print("\n🔍 Diagnostyka liczności binów WoE...")

    # 1. Pobieramy train i WoETransformera
    X_train, y_train = get_train_split()
    woe_tr = load_woe_transformer()

    # 2. Top cechy wg |beta| (ujemne)
    top_neg = df_coef[df_coef["sign"] == "negative"].head(n_top)["feature"].tolist()
    pos_feats = df_coef[df_coef["sign"] == "positive"]["feature"].tolist()

    features_to_check = top_neg + pos_feats

    all_tables = []

    for feat in features_to_check:
        try:
            tbl = compute_bin_table_for_feature(feat, X_train, y_train, woe_tr, min_count=min_count)
        except KeyError as e:
            print(f"⚠️ [WARN] Pomijam {feat}: {e}")
            continue

        tbl["feature"] = feat
        all_tables.append(tbl)

        # Krótkie podsumowanie w konsoli
        print(f"\n📊 Cechy binów – {feat}:")
        print(tbl[["bin", "total", "good", "bad", "default_rate", "low_count_flag"]].to_string(index=False))

        # Zapis osobnego pliku CSV dla tej cechy
        out_path_single = os.path.join(
            INTERP_DIR,
            f"woe_bin_counts_{feat}.csv"
        )
        tbl.to_csv(out_path_single, index=False)

    if all_tables:
        full = pd.concat(all_tables, ignore_index=True)
        out_path_all = os.path.join(INTERP_DIR, "woe_bin_counts_all_checked_features.csv")
        full.to_csv(out_path_all, index=False)
        print(f"\n💾 Zapisano zbiorczą tabelę liczności binów do: {out_path_all}")
    else:
        print("\n⚠️ Nie udało się zbudować żadnej tabeli binów – sprawdź nazwy cech i WoE.")



# ============================================================
#                RANKING CECH I CONTRIBUTION PLOT
# ============================================================

def plot_beta_importance(df_coef, top_n=20):
    """Wykres słupkowy top_n cech wg |beta|."""
    df_top = df_coef.head(top_n).iloc[::-1]  # od najmniejszej do największej na osi Y

    plt.figure(figsize=(8, 6))
    colors = ["tab:red" if b > 0 else "tab:green" for b in df_top["beta"]]
    plt.barh(df_top["feature"], df_top["beta"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Wartość współczynnika β")
    plt.title(f"Top {top_n} cech wg |β|")
    plt.tight_layout()

    out_path = os.path.join(WYKRESY_DIR, f"beta_importance_top{top_n}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n📊 Zapisano wykres ważności cech: {out_path}")


def plot_contribution_for_top_case(df_coef, intercept, X_woe, y, logit):
    """Contribution plot dla obserwacji o najwyższym PD."""
    proba = logit.predict_proba(X_woe)[:, 1]
    top_idx = np.argmax(proba)

    x_row = X_woe.iloc[top_idx]
    beta = df_coef.set_index("feature")["beta"]

    contrib = x_row * beta
    df_contrib = contrib.to_frame("contribution")
    df_contrib["abs_contrib"] = df_contrib["contribution"].abs()
    df_contrib = df_contrib.sort_values("abs_contrib", ascending=False).head(15)
    df_contrib = df_contrib.iloc[::-1]  # do barh

    # logit i PD dla tej obserwacji
    logit_val = intercept + (x_row * beta).sum()
    pd_val = 1 / (1 + np.exp(-logit_val))
    base_pd = 1 / (1 + np.exp(-intercept))

    plt.figure(figsize=(8, 6))
    colors = ["tab:red" if c > 0 else "tab:green" for c in df_contrib["contribution"]]
    plt.barh(df_contrib.index, df_contrib["contribution"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Wkład do logitu (β_j * x_j)")
    plt.title(
        f"Contribution plot – top case (PD={pd_val:.3f}, base PD={base_pd:.3f})"
    )
    plt.tight_layout()

    out_path = os.path.join(WYKRESY_DIR, "contribution_top_case.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 Zapisano contribution plot: {out_path}")


# ============================================================
#                     PDP i ICE DLA TOP CECH
# ============================================================

def compute_pdp_ice_for_feature(X_woe, y, logit, feature,
                                grid_size=20, ice_samples=50, random_state=42):
    """
    Liczy PDP i ICE dla pojedynczej cechy:
      - PDP: średnie PD po zastąpieniu danej cechy różnymi wartościami z gridu
      - ICE: dla wybranych obserwacji śledzimy PD w funkcji tej cechy
    Zwraca (grid, pdp_values, ice_curves), gdzie:
      - grid: wartości cechy
      - pdp_values: mean PD dla każdego z gridu
      - ice_curves: lista np.array o długości grid_size (każda to krzywa dla 1 obserwacji)
    """
    rng = np.random.RandomState(random_state)
    x_vals = X_woe[feature].values

    # ograniczamy się do "środka" rozkładu
    low, high = np.percentile(x_vals, [5, 95])
    grid = np.linspace(low, high, grid_size)

    # PDP
    pdp_values = []
    for v in grid:
        X_mod = X_woe.copy()
        X_mod[feature] = v
        pdp_values.append(logit.predict_proba(X_mod)[:, 1].mean())
    pdp_values = np.array(pdp_values)

    # ICE
    n_samples = min(ice_samples, len(X_woe))
    sample_idx = rng.choice(len(X_woe), size=n_samples, replace=False)

    ice_curves = []
    for idx in sample_idx:
        row = X_woe.iloc[idx:idx+1].copy()
        preds = []
        for v in grid:
            row_mod = row.copy()
            row_mod[feature] = v
            preds.append(logit.predict_proba(row_mod)[:, 1][0])
        ice_curves.append(np.array(preds))

    return grid, pdp_values, ice_curves


def plot_pdp(grid, pdp_values, feature):
    plt.figure(figsize=(7, 5))
    plt.plot(grid, pdp_values, "-o")
    plt.xlabel(f"{feature} (WoE)")
    plt.ylabel("Średnie PD")
    plt.title(f"PDP – {feature}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(PDP_DIR, f"pdp_{feature}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   ➜ zapisano PDP: {out_path}")


def plot_ice(grid, ice_curves, feature):
    plt.figure(figsize=(7, 5))
    for curve in ice_curves:
        plt.plot(grid, curve, alpha=0.2, color="tab:blue")
    plt.xlabel(f"{feature} (WoE)")
    plt.ylabel("PD")
    plt.title(f"ICE – {feature}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(ICE_DIR, f"ice_{feature}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   ➜ zapisano ICE: {out_path}")


def generate_pdp_ice_for_top_features(df_coef, X_woe, y, logit, top_n=5):
    """PDP + ICE dla top_n cech wg |beta| (niezależnie od znaku)."""
    df_top = df_coef.head(top_n)
    print(f"\n📈 PDP i ICE dla top {top_n} cech wg |beta|:")

    for feat in df_top["feature"]:
        print(f"   • {feat}")
        grid, pdp_vals, ice_curves = compute_pdp_ice_for_feature(
            X_woe, y, logit, feature=feat
        )
        plot_pdp(grid, pdp_vals, feat)
        plot_ice(grid, ice_curves, feat)



# ============================================================
#             LOKALNA INTERPRETACJA – 9 PRZYPADKÓW
# ============================================================

INTERP_LOCAL_DIR = os.path.join(INTERP_DIR, "interpretowalnosc_lokalna")
os.makedirs(INTERP_LOCAL_DIR, exist_ok=True)

LOGIT_PREPROC_PATH = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_logit_preproc():
    """
    Ładuje pipeline preprocessingowy dla logitu (WoE).
    """
    if not os.path.exists(LOGIT_PREPROC_PATH):
        raise FileNotFoundError(f"Nie znaleziono pipeline'u logitu pod: {LOGIT_PREPROC_PATH}")
    return joblib.load(LOGIT_PREPROC_PATH)


def get_data_splits_for_local():
    """
    Odwzorowuje podział 60/20/20 używany w projekcie.
    Zwraca: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_to_feature_df(preproc_logit, logit, X):
    """
    Przepuszcza X przez preproc_logit i zwraca DataFrame
    z kolumnami w tej samej kolejności, co feature_names_in_ modelu logit.
    """
    X_tr = preproc_logit.transform(X)

    if isinstance(X_tr, pd.DataFrame):
        # upewniamy się, że kolumny są w tej samej kolejności
        return X_tr.loc[:, logit.feature_names_in_]

    # jeśli np. jest to ndarray:
    return pd.DataFrame(X_tr, columns=logit.feature_names_in_)


def select_9_cases_evenly_by_pd(X_test_tr, y_test, logit):
    """
    Wybiera 9 obserwacji z testu, rozłożonych równomiernie po skali PD.
    Korzysta z kwantyli przewidzianych PD (0.05, 0.15, ..., 0.95).
    Zwraca listę indeksów X_test (oryginalnych).
    """
    p_test = logit.predict_proba(X_test_tr)[:, 1]
    s = pd.Series(p_test, index=y_test.index)

    quantiles = np.linspace(0.05, 0.95, 9)
    selected_idx = []

    for q in quantiles:
        target = s.quantile(q)
        # obserwacja, której PD jest najbliżej wybranego kwantyla
        idx = (s - target).abs().sort_values().index[0]
        # jeśli już mamy tę obserwację, szukamy kolejnej najbliższej
        if idx in selected_idx:
            for alt_idx in (s - target).abs().sort_values().index:
                if alt_idx not in selected_idx:
                    idx = alt_idx
                    break
        selected_idx.append(idx)

    return selected_idx, s


def decompose_logit_for_case(idx, x_row, y_true, beta, intercept, feature_names):
    """
    Rozkłada logit dla pojedynczej obserwacji na wkłady cech.
    Zwraca:
      - df_contrib: DataFrame z wkładami dla wszystkich cech (do sortowania)
      - meta: dict z logitem, PD i y_true
    """
    x_vals = x_row.values.astype(float)
    beta_vals = beta.astype(float)

    contrib = x_vals * beta_vals
    logit_val = intercept + contrib.sum()
    pd_val = sigmoid(logit_val)

    df_contrib = pd.DataFrame({
        "feature": feature_names,
        "x_value": x_vals,
        "beta": beta_vals,
        "contribution": contrib,
    })
    df_contrib["abs_contribution"] = df_contrib["contribution"].abs()

    # sortujemy od najbardziej wpływowych cech
    df_contrib = df_contrib.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

    meta = {
        "index": int(idx),
        "y_true": int(y_true),
        "logit": float(logit_val),
        "pd": float(pd_val),
    }
    return df_contrib, meta


def compute_local_decomposition_for_9_cases(logit, df_coef):
    """
    Główna funkcja:
      - ładuje preproc i dane,
      - wybiera 9 case'ów rozłożonych po skali PD,
      - dla każdego case'a rozkłada logit na wkłady cech,
      - zapisuje:
          * local_cases_meta.csv – 9 wierszy (case_id, index, y_true, logit, pd)
          * local_cases_top10_contributions.csv – top 10 cech dla każdego case'a
    """
    print("\n🧩 Liczę lokalną interpretację (9 przypadków)...")

    preproc_logit = load_logit_preproc()
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits_for_local()
    X_test_tr = transform_to_feature_df(preproc_logit, logit, X_test)

    beta = logit.coef_.ravel()
    intercept = float(logit.intercept_[0])
    feature_names = np.array(logit.feature_names_in_)

    # wybieramy 9 przypadków
    selected_idx, pd_series = select_9_cases_evenly_by_pd(X_test_tr, y_test, logit)
    print(f"   Wybrane indeksy testu: {selected_idx}")

    meta_rows = []
    all_top10_rows = []

    for case_id, idx in enumerate(selected_idx, start=1):
        x_row = X_test_tr.loc[idx, :]
        y_true = y_test.loc[idx]

        df_contrib, meta = decompose_logit_for_case(
            idx=idx,
            x_row=x_row,
            y_true=y_true,
            beta=beta,
            intercept=intercept,
            feature_names=feature_names,
        )

        # łączymy z globalnymi informacjami o współczynnikach (np. abs_beta, sign)
        df_contrib = df_contrib.merge(
            df_coef[["feature", "abs_beta", "sign"]],
            how="left",
            on="feature",
        )

        # bierzemy top 10 cech
        df_top10 = df_contrib.head(10).copy()
        df_top10["case_id"] = case_id
        df_top10["original_index"] = int(idx)
        df_top10["rank"] = np.arange(1, len(df_top10) + 1)

        all_top10_rows.append(df_top10)

        meta["case_id"] = case_id
        meta["original_index"] = int(idx)
        meta_rows.append(meta)

    # zapis meta
    df_meta = pd.DataFrame(meta_rows)[
        ["case_id", "original_index", "y_true", "logit", "pd"]
    ]
    meta_path = os.path.join(INTERP_LOCAL_DIR, "local_cases_meta.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"💾 Zapisano podsumowanie 9 przypadków → {meta_path}")

    # zapis top10 contributions (long format)
    df_all_top10 = pd.concat(all_top10_rows, ignore_index=True)
    contrib_path = os.path.join(INTERP_LOCAL_DIR, "local_cases_top10_contributions.csv")
    df_all_top10.to_csv(contrib_path, index=False)
    print(f"💾 Zapisano top 10 wkładów cech dla 9 przypadków → {contrib_path}")


# ============================================================
#                           MAIN
# ============================================================

def main():
    
    print("📂 Ładowanie modelu logit (WoE)...")
    logit = load_logit_model()
    preproc_logit = load_logit_preprocessor()

    print("📊 Ekstrakcja współczynników...")
    df_coef, intercept = extract_coefficients(logit)

    summarize_signs(df_coef, intercept)
    save_coefficients(df_coef)

    print("\nTop 10 cech wg |beta|:")
    print(df_coef.head(10).to_string(index=False))

    print("\n📂 Przygotowywanie danych (WoE)...")
    X_woe, y = load_and_prepare_data(preproc_logit, logit)

    # ---------- Profile WoE ----------
    generate_woe_profiles(df_coef, X_woe, y)

    # ---------- Ranking cech ----------
    plot_beta_importance(df_coef, top_n=20)

    # ---------- Contribution plot ----------
    plot_contribution_for_top_case(df_coef, intercept, X_woe, y, logit)

    # ---------- PDP + ICE ----------
    generate_pdp_ice_for_top_features(df_coef, X_woe, y, logit, top_n=5)

    print("\n✅ Zakończono generowanie wykresów interpretowalności logitu.")
    
    diagnose_bin_sizes(df_coef, n_top=5, min_count=50)
    
    # Lokalna interpretacja – 5 obserwacji
    compute_local_decomposition_for_9_cases(logit, df_coef)

if __name__ == "__main__":
    main()
