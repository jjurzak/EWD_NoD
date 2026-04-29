# Automatyczne wykrywanie liczby klastrów w danych

Projekt badawczy realizowany w ramach przedmiotu **Eksploracja i Wizualizacja Danych**. Głównym celem pracy jest porównanie skuteczności i wiarygodności popularnych heurystyk w określaniu optymalnej liczby klastrów w danych przy użyciu analizy nienadzorowanej (klasteryzacji).

Autorzy: **Jakub Jurzak, Marcin Osojca**

---

## 🎯 Cel projektu
Ocena skuteczności trzech popularnych metod detekcji optymalnego $k$ (liczby klastrów):
1. **Elbow Method** (Metoda łokcia)
2. **Silhouette Score** (Współczynnik sylwetki)
3. **Gap Statistic** (Statystyka luki)

Problem został zbadany na wielowymiarowym zbiorze danych, w którym klastry (odpowiadające różnym klasom) silnie na siebie nakładają. Uzyskane przez algorytmy wyniki ($k$) zostały zweryfikowane względem znanej liczby klas (ground truth) z wykorzystaniem metryk walidacji zewnętrznej: **Adjusted Rand Index (ARI)** oraz **Normalized Mutual Information (NMI)**.

## 📊 Zbiór danych
W projekcie wykorzystano [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) z repozytorium UCI Machine Learning:
* **Liczba obserwacji:** 13 611
* **Liczba cech:** 16 zmiennych numerycznych (opisujących kształt i wymiary morfologiczne fasoli)
* **Klasy:** 7 różnych gatunków (etykieta wykorzystywana wyłącznie do walidacji końcowej)

Zbiór cechuje się niezbalansowanym rozkładem klas i wysoką korelacją wybranych cech wymiarowych.

## ⚙️ Wymagania i uruchomienie
Projekt został zaimplementowany w języku Python. 

### Narzędzia:
* Python 3.10+
* Główne biblioteki: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`

### Instrukcja uruchomienia:
1. Pobierz repozytorium na dysk:
   ```bash
   git clone https://github.com/jjurzak/EWD_NoD.git
   cd EWD_NoD
   ```
2. Upewnij się, że masz zainstalowane wszystkie pakiety. Możesz je zainstalować używając `pip`:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl
   ```
3. Uruchom skrypt główny:
   ```bash
   python main.py
   ```
4. Wygenerowane wykresy i tabele (w formacie PNG oraz CSV) zostaną zapisane w katalogu `output/`.

## 📁 Struktura projektu
* `main.py` - Główny skrypt analizy danych (wczytanie, EDA, standaryzacja, PCA, klasteryzacja, ewaluacja).
* `sprawozdanie.pdf` - Wygenerowane sprawozdanie końcowe w LaTeX ze szczegółowym omówieniem wyników.
* `sprawozdanie.tex` - Kod źródłowy raportu w LaTeX.
* `dataset/` - Folder z plikami zbioru danych (m.in. .xlsx, .arff).
* `output/` - Zbiór wygenerowanych wykresów, takich jak:
  * Macierze korelacji i rozkłady klas.
  * Wizualizacje metod detekcji: krzywe Elbow, wykresy Silhouette i Gap Statistic.
  * Wizualizacje skupień po redukcji wymiarowości (PCA).

## 🏆 Wyniki
Żadna ze standardowych heurystyk nie wskazała idealnie naturalnej liczby klas ($k=7$), co wynika z bardzo małych różnic morfologicznych między niektórymi odmianami fasoli. 
Najdokładniejszą ocenę struktury wewnętrznej danych dostarczyła **Gap Statistic**, sugerująca $k=5$, co korespondowało z najwyższymi wynikami metryk ewaluacyjnych (ARI i NMI). Z kolei **Silhouette Score** znacząco zaniżył tę wartość ($k=3$), faworyzując mniejszą liczbę bardziej oddalonych od siebie skupień.

Szczegółowe tabele porównawcze i omówienie zjawiska nakładających się klastrów znajduje się w dokumencie **`sprawozdanie.pdf`**.
