# IWUM-Projekt-1
Pierwszy projekt z przedmiotu interpretowalność i wyjaśnialność uczenia maszynowego

Poki co odpalamy to w nastepujacej kolejnosci:
1.EDA/dopasowanie_pipeline.py
2.Modele_interpretowalne/modele_interpretowalne
3.Modele_interpretowalne/ocena_jakosci_modelow_wykresy (chociaz wykresy sa juz itak wrzucone na githuba)
4.Modele_interpretowalne/interpretowalnosc_regresja_logistyczna

# 📌 Dokumentacja skryptów (moduł interpretowalnego modelu)

---

### **1. `EDA/transformers.py`**
Zestaw własnych transformerów wykorzystywanych w preprocessing’u:

- **InfinityReplacer** — zamienia wartości `±inf → NaN`
- **HighMissingDropper** — usuwa kolumny z dużym udziałem braków
- **MissingIndicator** — generuje flagi braków
- **CustomImputer** — imputacja braków (num + cat)
- **Winsorizer** — przycinanie skrajnych wartości (winsoryzacja)
- **LowVarianceDropper** — usuwa kolumny o niskiej wariancji
- **HighCorrelationDropper** — usuwa kolumny o wysokiej korelacji
- **WoETransformer** — wykonuje binning + liczy WoE + IV
- **WoEDirectionalityFilter** — usuwa cechy, których WoE ma nielogiczny kierunek (rosnący WoE przy rosnącym default rate)
- **DropColumnsTransformer** — usuwa cechy, dla których model logistyczny wyliczył dodatnie bety (bazując na liście z pliku `features_to_drop_positive_beta.txt`)

To jest **biblioteka wszystkich customowych transformacji** używanych w projekcie.

---

### **2. `EDA/dopasowanie_pipeline.py`**
Skrypt budujący pipeline’y preprocessingowe:

- wykonuje podział danych **train/val/test (60/20/20)**
- trenuje dwa pipeline’y:
  - `preprocessing_tree.pkl` — pipeline pod model drzewa
  - `preprocessing_logit_woe.pkl` — pipeline pod logit WoE (z filtrami kierunku i usuwaniem dodatnich bet)
- zapisuje pipeline’y do folderu:  
  **`EDA/preprocesing_pipelines/`**

To jest **skrypt treningowy preprocessing’u**, uruchamiany przed trenowaniem modeli.

---

### **3. `Modele_interpretowalne/modele_interpretacyjne.py`**
Skrypt odpowiedzialny za trenowanie modeli interpretowalnych:

- wczytuje dane i pipeline’y z EDA
- wykonuje **GridSearchCV** dla:
  - regresji logistycznej (WoE)
  - drzewa decyzyjnego (płytkie, interpretowalne)
- wybiera najlepsze modele na podstawie **ROC-AUC**
- liczy metryki:
  - ROC-AUC  
  - PR-AUC  
  - KS statistic  
  - log-loss  
  - Brier score
- zapisuje finalne modele do:
  **`Modele_interpretowalne/models/`**

To jest **główny skrypt trenowania modeli interpretowalnych**.

---

### **4. `Modele_interpretowalne/ocena_jakosci_modelow_wykresy.py`**
Skrypt generujący wykresy jakości modeli:

- krzywe **ROC** (val + test)
- krzywe **Precision–Recall** (val + test)
- **Calibration plot**
- **Histogramy PD** (rozkład predykcji dla good/bad)

Wszystkie wykresy zapisywane są do:
**`Modele_interpretowalne/wykresy_oceny_jakosci/`**

To jest **wizualne porównanie jakości logitu i drzewa**.

---

### **5. `Modele_interpretowalne/interpretowalnosc_regresja_logistyczna.py`**
Główny skrypt interpretowalności modelu logistycznego:

#### Co robi:
- ładuje `best_logistic_regression_woe.pkl`
- wyciąga współczynniki **beta**, liczy:
  - `abs_beta`
  - `odds_ratio = exp(beta)`
  - znak beta
- zapisuje tabelę współczynników do:
  **`interpretowalnosc_logit/coefficients_logit.csv`**

#### Generuje wykresy:
- **profile WoE** (default rate vs WoE)
- diagnostyka liczności binów (good/bad/total)
- **PDP** (średnia zmiana predykcji)
- **ICE** (indywidualne krzywe dla obserwacji)

Zapisywane do folderów:
- `interpretowalnosc_logit/woe_profiles/`
- `interpretowalnosc_logit/bin_diagnostics/`
- `interpretowalnosc_logit/PDP/`
- `interpretowalnosc_logit/ICE/`

To jest **kompletny moduł interpretowalności globalnej modelu logistycznego**.

---

### **6. `Modele_interpretowalne/interpretowalnosc_logit/diagnoza_modelu_logstycznego.py`**
⚠️ **ARCHIWALNY SKRYPT – NIE URUCHAMIAĆ**

Działał **wyłącznie** na poprzednim modelu logistycznym, który:
- miał **32 dodatnie bety**,  
- nie zawierał filtra kierunku WoE,  
- nie był interpretowalny.

Aktualny projekt korzysta tylko z:
- `modele_interpretacyjne.py`
- `interpretowalnosc_regresja_logistyczna.py`

Na górze pliku znajduje się ostrzeżenie:

lu logistycznego

