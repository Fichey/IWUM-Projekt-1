import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
#           KONFIGURACJA ŚCIEŻEK I FOLDERÓW WYJŚCIOWYCH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../interpretowalnosc_lokalna

META_PATH = os.path.join(BASE_DIR, "local_cases_meta.csv")
CONTRIB_PATH = os.path.join(BASE_DIR, "local_cases_top10_contributions.csv")

PLOTS_CASE_DIR = os.path.join(BASE_DIR, "wykresy_case")
PLOTS_GRID_DIR = os.path.join(BASE_DIR, "wykresy_zbiorcze")

os.makedirs(PLOTS_CASE_DIR, exist_ok=True)
os.makedirs(PLOTS_GRID_DIR, exist_ok=True)


# ============================================================
#                Wczytanie danych lokalnych
# ============================================================

def load_local_data():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku meta: {META_PATH}")
    if not os.path.exists(CONTRIB_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku z wkładami: {CONTRIB_PATH}")

    df_meta = pd.read_csv(META_PATH)
    df_contrib = pd.read_csv(CONTRIB_PATH)

    # Upewniamy się, że case_id jest int
    df_meta["case_id"] = df_meta["case_id"].astype(int)
    df_contrib["case_id"] = df_contrib["case_id"].astype(int)

    return df_meta, df_contrib


# ============================================================
#         Wykresy pojedyncze – top10 wkładów dla case
# ============================================================

def plot_single_case_bar(df_case, meta_row, save_path):
    """
    Rysuje wykres słupkowy wkładów cech do logitu dla pojedynczego case'a.
    df_case – wiersze dla jednego case_id (top 10 cech),
    meta_row – Series z informacjami: case_id, original_index, y_true, logit, pd
    """
    # sortujemy tak, aby najbardziej wpływowe cechy były na górze
    df_plot = df_case.sort_values("abs_contribution", ascending=True).copy()

    features = df_plot["feature"]
    contrib = df_plot["contribution"]

    # kolor: czerwony dla dodatnich wkładów, niebieski dla ujemnych
    colors = np.where(contrib >= 0, "#d62728", "#1f77b4")  # red / blue

    plt.figure(figsize=(8, 5))
    plt.barh(features, contrib, color=colors)
    plt.axvline(0, color="black", linewidth=1)

    case_id = int(meta_row["case_id"])
    idx = int(meta_row["original_index"])
    y_true = int(meta_row["y_true"])
    logit_val = meta_row["logit"]
    pd_val = meta_row["pd"]

    plt.title(
        f"Case {case_id} (idx={idx}, y={y_true})\n"
        f"logit={logit_val:.3f}, PD={pd_val:.3%}"
    )
    plt.xlabel("Wkład do logitu (beta * x)")
    plt.ylabel("Cecha")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_per_case_plots(df_meta, df_contrib):
    """
    Generuje po jednym wykresie dla każdego z 9 case'ów (top 10 cech).
    """
    for _, meta_row in df_meta.sort_values("case_id").iterrows():
        case_id = int(meta_row["case_id"])
        idx = int(meta_row["original_index"])

        df_case = df_contrib[df_contrib["case_id"] == case_id].copy()

        fname = f"case_{case_id}_idx_{idx}_top10_contributions.png"
        save_path = os.path.join(PLOTS_CASE_DIR, fname)

        plot_single_case_bar(df_case, meta_row, save_path)
        print(f"💾 Zapisano wykres dla case {case_id} → {save_path}")


# ============================================================
#     Zbiorczy wykres 3×3 – 9 case'ów, kolor = gradient po PD
# ============================================================

def plot_grid_cases(df_meta, df_contrib, n_per_case=10):
    """
    Tworzy zbiorczy wykres 3x3:
      - każdy subplot to top 10 wkładów dla danego case'a,
      - kolor słupków jest kolorem z mapy barw zależnym od PD (gradient).
    """
    # przygotowanie mapy kolorów po PD
    pd_vals = df_meta["pd"].values
    pd_min, pd_max = pd_vals.min(), pd_vals.max()
    norm = plt.Normalize(pd_min, pd_max)
    cmap = plt.cm.get_cmap("viridis")

    cases_sorted = df_meta.sort_values("case_id").reset_index(drop=True)
    n_cases = len(cases_sorted)

    n_rows, n_cols = 3, 3  # zakładamy 9 case'ów
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.ravel()

    for i, (_, meta_row) in enumerate(cases_sorted.iterrows()):
        ax = axes[i]
        case_id = int(meta_row["case_id"])
        idx = int(meta_row["original_index"])
        y_true = int(meta_row["y_true"])
        logit_val = meta_row["logit"]
        pd_val = meta_row["pd"]

        df_case = df_contrib[df_contrib["case_id"] == case_id].copy()
        df_case = df_case.sort_values("abs_contribution", ascending=True).tail(n_per_case)

        features = df_case["feature"]
        contrib = df_case["contribution"]

        # kolor całego case'a – jeden kolor z gradientu po PD
        color_case = cmap(norm(pd_val))

        ax.barh(features, contrib, color=color_case)
        ax.axvline(0, color="black", linewidth=0.8)

        ax.set_title(
            f"Case {case_id} (idx={idx}, y={y_true})\n"
            f"PD={pd_val:.2%}"
        )
        ax.set_xlabel("Wkład do logitu")
        ax.set_ylabel("Cecha")
        ax.grid(axis="x", alpha=0.3)

    # jeśli case'ów mniej niż 9, ukrywamy puste osie
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # dodajemy pasek koloru opisujący PD
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("PD (prawdopodobieństwo defaultu)")

    fig.suptitle("Lokalna interpretacja – top 10 wkładów (9 przypadków)", fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.95])

    out_path = os.path.join(PLOTS_GRID_DIR, "grid_3x3_cases_top10_contributions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"💾 Zapisano zbiorczy wykres 3x3 → {out_path}")


# ============================================================
#                                MAIN
# ============================================================

def main():
    print("📂 Wczytywanie danych lokalnej interpretacji...")
    df_meta, df_contrib = load_local_data()

    print("🖼️ Rysuję wykresy pojedynczych case'ów...")
    generate_per_case_plots(df_meta, df_contrib)

    print("🖼️ Rysuję zbiorczy wykres 3x3 z gradientem po PD...")
    plot_grid_cases(df_meta, df_contrib, n_per_case=10)

    print("✅ Gotowe – lokalna interpretacja zwizualizowana.")


if __name__ == "__main__":
    main()
