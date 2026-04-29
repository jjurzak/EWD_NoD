import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATASET_PATH = 'dataset/Dry_Bean_Dataset.xlsx'
K_RANGE = range(2, 16)
N_GAP_REFS = 20
RANDOM_STATE = 42


#WCZYTANIE DANYCH 
print("[1/8] Wczytywanie danych...")
df = pd.read_excel(DATASET_PATH)
labels_true = df['Class']
features = df.drop(columns=['Class'])
feature_names = features.columns.tolist()
n_classes = labels_true.nunique()
print(f"     Obserwacje: {len(df)}, Cechy: {len(feature_names)}, Klasy: {n_classes}")


# EKSPLORACJA DANYCH (EDA)
print("[2/8] Eksploracja danych...")

# Statystyki opisowe -> CSV
desc = features.describe().T
desc.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_stats.csv'))

# Rozkład klas
class_counts = labels_true.value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
class_counts.plot(kind='bar', color=sns.color_palette('Set2', n_classes), edgecolor='black', ax=ax)
ax.set_title('Rozkład klas w zbiorze Dry Bean Dataset')
ax.set_xlabel('Klasa')
ax.set_ylabel('Liczba obserwacji')
ax.bar_label(ax.containers[0], fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
plt.close(fig)

# Macierz korelacji
fig, ax = plt.subplots(figsize=(12, 10))
corr = features.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Macierz korelacji cech')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
plt.close(fig)

# Histogramy wybranych cech
selected_feats = ['Area', 'Perimeter', 'Compactness', 'Eccentricity', 'Solidity', 'roundness']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.ravel(), selected_feats):
    for cls in labels_true.unique():
        ax.hist(features.loc[labels_true == cls, feat], bins=40, alpha=0.5, label=cls)
    ax.set_title(feat)
    ax.set_xlabel('')
axes[0, 2].legend(fontsize=7, loc='upper right')
plt.suptitle('Rozkłady wybranych cech wg klasy', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'))
plt.close(fig)


# STANDARYZACJA + PCA
print("[3/8] Standaryzacja i PCA...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA — pełna
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_,
       color='steelblue', alpha=0.7, label='Indywidualna')
ax.step(range(1, len(cumvar) + 1), cumvar, where='mid',
        color='firebrick', linewidth=2, label='Skumulowana')
ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, label='95% wariancji')
ax.set_xlabel('Numer składowej głównej')
ax.set_ylabel('Wyjaśniona wariancja')
ax.set_title('PCA — wariancja wyjaśniona')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pca_variance.png'))
plt.close(fig)

n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"     Składowych PCA dla 95% wariancji: {n_components_95}")

# PCA do klasteryzacji
pca = PCA(n_components=n_components_95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# PCA 2D — wizualizacja ground-truth
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette('tab10', n_classes)
for i, cls in enumerate(sorted(labels_true.unique())):
    mask = labels_true == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.5,
               color=palette[i], label=cls)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA 2D — klasy rzeczywiste (ground truth)')
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pca_2d_ground_truth.png'))
plt.close(fig)


# ELBOW METHOD
print("[4/8] Elbow Method...")

inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    km.fit(X_pca)
    inertias.append(km.inertia_)

# Wyznaczenie punktu łokcia (max drugiej pochodnej)
diffs1 = np.diff(inertias)
diffs2 = np.diff(diffs1)
elbow_k = list(K_RANGE)[np.argmax(np.abs(diffs2)) + 2]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(K_RANGE), inertias, 'bo-', linewidth=2, markersize=8)
ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=1.5,
           label=f'Łokieć: k={elbow_k}')
ax.set_xlabel('Liczba klastrów (k)')
ax.set_ylabel('Inercja (WCSS)')
ax.set_title('Metoda łokcia (Elbow Method)')
ax.set_xticks(list(K_RANGE))
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'elbow_method.png'))
plt.close(fig)
print(f"     Elbow k = {elbow_k}")


# SILHOUETTE SCORE 
print("[5/8] Silhouette Score...")

sil_scores = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    lab = km.fit_predict(X_pca)
    sil_scores.append(silhouette_score(X_pca, lab))

sil_best_k = list(K_RANGE)[np.argmax(sil_scores)]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['red' if k == sil_best_k else 'steelblue' for k in K_RANGE]
ax.bar(list(K_RANGE), sil_scores, color=colors, edgecolor='black', alpha=0.85)
ax.set_xlabel('Liczba klastrów (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score dla różnych k')
ax.set_xticks(list(K_RANGE))
ax.axhline(y=max(sil_scores), color='red', linestyle='--', linewidth=1,
           label=f'Maks: k={sil_best_k} ({max(sil_scores):.4f})')
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'silhouette_scores.png'))
plt.close(fig)
print(f"     Silhouette best k = {sil_best_k}")


# ── 6. GAP STATISTIC ────────────────────────────────────────────────────
print("[6/8] Gap Statistic (B={})...".format(N_GAP_REFS))


def compute_gap_statistic(X, k_range, n_refs=20, rng=42):
    """Gap Statistic wg Tibshirani et al. (2001)."""
    rng = np.random.RandomState(rng)
    gaps = []
    s_k = []
    W_obs = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        w_k = km.inertia_
        W_obs.append(np.log(w_k))

        w_refs = []
        for _ in range(n_refs):
            X_ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=42)
            km_ref.fit(X_ref)
            w_refs.append(np.log(km_ref.inertia_))

        gap = np.mean(w_refs) - np.log(w_k)
        gaps.append(gap)
        sdk = np.std(w_refs) * np.sqrt(1 + 1.0 / n_refs)
        s_k.append(sdk)

    # Kryterium: gap(k) >= gap(k+1) - s(k+1)
    gaps = np.array(gaps)
    s_k = np.array(s_k)
    best_k = list(k_range)[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - s_k[i + 1]:
            best_k = list(k_range)[i]
            break
    return gaps, s_k, best_k


gaps, gap_sds, gap_best_k = compute_gap_statistic(X_pca, K_RANGE, N_GAP_REFS, RANDOM_STATE)

fig, ax = plt.subplots(figsize=(9, 5))
k_list = list(K_RANGE)
ax.errorbar(k_list, gaps, yerr=gap_sds, fmt='go-', linewidth=2, markersize=8,
            capsize=4, capthick=1.5)
ax.axvline(x=gap_best_k, color='red', linestyle='--', linewidth=1.5,
           label=f'Optymalne k={gap_best_k}')
ax.set_xlabel('Liczba klastrów (k)')
ax.set_ylabel('Gap Statistic')
ax.set_title('Gap Statistic')
ax.set_xticks(k_list)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'gap_statistic.png'))
plt.close(fig)
print(f"     Gap best k = {gap_best_k}")


# ── 7. PORÓWNANIE METOD ─────────────────────────────────────────────────
print("[7/8] Porównanie metod i ewaluacja...")

results = {
    'Metoda': ['Elbow Method', 'Silhouette Score', 'Gap Statistic'],
    'Sugerowane k': [elbow_k, sil_best_k, gap_best_k],
    'Ground Truth k': [n_classes, n_classes, n_classes],
}

# ARI i NMI dla każdego sugerowanego k
ari_list = []
nmi_list = []
for k in [elbow_k, sil_best_k, gap_best_k]:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    pred = km.fit_predict(X_pca)
    ari_list.append(adjusted_rand_score(labels_true, pred))
    nmi_list.append(normalized_mutual_info_score(labels_true, pred))

results['ARI'] = [f'{v:.4f}' for v in ari_list]
results['NMI'] = [f'{v:.4f}' for v in nmi_list]

# ARI/NMI dla ground-truth k=7
km_gt = KMeans(n_clusters=n_classes, n_init=10, random_state=RANDOM_STATE)
pred_gt = km_gt.fit_predict(X_pca)
ari_gt = adjusted_rand_score(labels_true, pred_gt)
nmi_gt = normalized_mutual_info_score(labels_true, pred_gt)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_results.csv'), index=False)
print(results_df.to_string(index=False))
print(f"     k=7 (ground truth): ARI={ari_gt:.4f}, NMI={nmi_gt:.4f}")

# Tabela jako wykres
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')
table = ax.table(cellText=results_df.values,
                 colLabels=results_df.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#D9E2F3' if row % 2 == 0 else 'white')
ax.set_title('Porównanie metod wykrywania optymalnego k', fontsize=13, pad=20)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'comparison_table.png'))
plt.close(fig)

# Wykres zbiorczy — 3 metody obok siebie
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow
axes[0].plot(k_list, inertias, 'bo-', linewidth=2, markersize=6)
axes[0].axvline(x=elbow_k, color='red', linestyle='--', label=f'k={elbow_k}')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('WCSS')
axes[0].legend()

# Silhouette
axes[1].bar(k_list, sil_scores,
            color=['red' if k == sil_best_k else 'steelblue' for k in K_RANGE],
            edgecolor='black')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')

# Gap
axes[2].errorbar(k_list, gaps, yerr=gap_sds, fmt='go-', capsize=3)
axes[2].axvline(x=gap_best_k, color='red', linestyle='--', label=f'k={gap_best_k}')
axes[2].set_title('Gap Statistic')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Gap')
axes[2].legend()

plt.suptitle('Porównanie trzech metod wykrywania optymalnej liczby klastrów', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'summary_comparison.png'))
plt.close(fig)


# ── 8. WIZUALIZACJE FINALNE ─────────────────────────────────────────────
print("[8/8] Wizualizacje finalne...")

# PCA 2D — klastry KMeans k=7
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Ground truth
palette = sns.color_palette('tab10', n_classes)
for i, cls in enumerate(sorted(labels_true.unique())):
    mask_cls = labels_true == cls
    axes[0].scatter(X_2d[mask_cls, 0], X_2d[mask_cls, 1], s=6, alpha=0.4,
                    color=palette[i], label=cls)
axes[0].set_title('Ground Truth (7 klas)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(markerscale=3, fontsize=8)

# KMeans k=7
for i in range(n_classes):
    mask_c = pred_gt == i
    axes[1].scatter(X_2d[mask_c, 0], X_2d[mask_c, 1], s=6, alpha=0.4,
                    color=palette[i], label=f'Klaster {i}')
axes[1].set_title(f'KMeans k={n_classes} (ARI={ari_gt:.3f})')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(markerscale=3, fontsize=8)

plt.suptitle('Porównanie: klasy rzeczywiste vs klastry KMeans', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'clusters_vs_ground_truth.png'))
plt.close(fig)

# Silhouette plot dla najlepszego k
from sklearn.metrics import silhouette_samples

fig, ax = plt.subplots(figsize=(10, 7))
km_sil = KMeans(n_clusters=sil_best_k, n_init=10, random_state=RANDOM_STATE)
sil_labels = km_sil.fit_predict(X_pca)
sil_vals = silhouette_samples(X_pca, sil_labels)

y_lower = 10
cmap = plt.cm.get_cmap('tab10')
for i in range(sil_best_k):
    ith_vals = sil_vals[sil_labels == i]
    ith_vals.sort()
    size_i = ith_vals.shape[0]
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                     facecolor=cmap(i / sil_best_k), alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=10)
    y_lower = y_upper + 10

ax.axvline(x=sil_scores[sil_best_k - list(K_RANGE)[0]], color='red',
           linestyle='--', linewidth=1.5)
ax.set_title(f'Silhouette Plot (k={sil_best_k})')
ax.set_xlabel('Silhouette coefficient')
ax.set_ylabel('Klaster')
ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'silhouette_plot_detail.png'))
plt.close(fig)

print("\n" + "=" * 60)
print("PODSUMOWANIE WYNIKÓW")
print("=" * 60)
print(f"  Elbow Method       -> k = {elbow_k}")
print(f"  Silhouette Score   -> k = {sil_best_k}")
print(f"  Gap Statistic      -> k = {gap_best_k}")
print(f"  Ground Truth       -> k = {n_classes}")
print(f"\n  KMeans(k=7): ARI = {ari_gt:.4f}, NMI = {nmi_gt:.4f}")
print("=" * 60)
print(f"\nWszystkie wykresy zapisano w: {os.path.abspath(OUTPUT_DIR)}")
