"""
Ejemplos visuales de clustering con datos de juguete.

Genera las imágenes que se usan en el README:
  01_datos_juguete.png
  02_kmeans_pasos.png
  03_comparacion_algoritmos.png
  04_dendrograma.png
  05_dbscan_core_border_noise.png
  06_codo_y_silueta.png

Uso:  python clustering_visual.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Estilo: paleta categórica validada (primeros 3 huecos) + gris para el ruido
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # azul, naranja, aguamarina
MARKERS = ["o", "s", "^"]  # segunda codificación: forma, además del color
NOISE_COLOR = MUTED

OUT = Path("images")
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)

SEED = 42


def limpia(ax, ticks=False):
    """Quita bordes superior y derecho y, opcionalmente, los números de los ejes."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def pinta(ax, X, etiquetas, tamano=22):
    """Dibuja los puntos coloreados por grupo. La etiqueta -1 es ruido (DBSCAN)."""
    for e in sorted(set(etiquetas)):
        m = etiquetas == e
        if e == -1:
            ax.scatter(X[m, 0], X[m, 1], s=tamano + 8, c=NOISE_COLOR, marker="x", linewidths=1.2)
        else:
            ax.scatter(
                X[m, 0],
                X[m, 1],
                s=tamano,
                c=COLORS[e % 3],
                marker=MARKERS[e % 3],
                edgecolors=SURFACE,
                linewidths=0.6,
            )


# ---------------------------------------------------------------------------
# Datos de juguete
# ---------------------------------------------------------------------------
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=SEED)
X_moons, _ = make_moons(n_samples=300, noise=0.06, random_state=SEED)

# ---------------------------------------------------------------------------
# 01. Datos de juguete sin colorear (lo que ve el algoritmo: puntos sin etiqueta)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, X, titulo in zip(axs, (X_blobs, X_moons), ("make_blobs: grupos redondeados", "make_moons: dos medias lunas")):
    ax.scatter(X[:, 0], X[:, 1], s=22, c=INK_2, edgecolors=SURFACE, linewidths=0.6)
    ax.set_title(titulo)
    limpia(ax)
fig.suptitle("Los datos sin etiquetas: así los ve el algoritmo", x=0.02, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "01_datos_juguete.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 02. K-Means paso a paso (algoritmo de Lloyd escrito a mano)
# ---------------------------------------------------------------------------
_, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=SEED)
# Inicio deliberadamente malo: dos centroides en el mismo grupo y uno en otro,
# para que se vea cómo se van moviendo hasta encontrar los tres grupos.
centros = np.array(
    [
        X_blobs[y_blobs == 0][0],
        X_blobs[y_blobs == 0][1],
        X_blobs[y_blobs == 1][0],
    ]
)
estados = []
for _ in range(15):
    dist = np.linalg.norm(X_blobs[:, None, :] - centros[None, :, :], axis=2)
    etiquetas = dist.argmin(axis=1)
    estados.append((centros.copy(), etiquetas.copy()))
    nuevos = np.array([X_blobs[etiquetas == k].mean(axis=0) if (etiquetas == k).any() else centros[k] for k in range(3)])
    if np.allclose(nuevos, centros):
        break
    centros = nuevos
print("RESUMEN kmeans pasos hasta converger:", len(estados))
elegidos = [0, 1, 2, len(estados) - 1]
pasos = [estados[i] for i in elegidos]
titulos = ["Inicio: 3 centroides mal colocados", "Iteración 1", "Iteración 2", "Final: los centroides ya no se mueven"]

fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (c, e), t in zip(axs, pasos, titulos):
    pinta(ax, X_blobs, e, tamano=16)
    ax.scatter(c[:, 0], c[:, 1], s=130, c=INK, marker="X", edgecolors=SURFACE, linewidths=1.5, zorder=5)
    ax.set_title(t)
    limpia(ax)
fig.suptitle(
    "K-Means: asignar cada punto al centroide más cercano y mover el centroide al centro de su grupo (X)",
    x=0.01,
    ha="left",
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "02_kmeans_pasos.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 03. Comparación de los 4 algoritmos en los 2 conjuntos de datos
# ---------------------------------------------------------------------------
def algoritmos(nombre):
    if nombre == "blobs":
        return [
            ("K-Means (K=3)", KMeans(n_clusters=3, n_init=10, random_state=SEED)),
            ("Jerárquico (Ward, 3)", AgglomerativeClustering(n_clusters=3, linkage="ward")),
            ("DBSCAN (eps=0.35)", DBSCAN(eps=0.35, min_samples=6)),
            ("GMM (3 componentes)", GaussianMixture(n_components=3, random_state=SEED)),
        ]
    return [
        ("K-Means (K=2)", KMeans(n_clusters=2, n_init=10, random_state=SEED)),
        ("Jerárquico (Ward, 2)", AgglomerativeClustering(n_clusters=2, linkage="ward")),
        ("DBSCAN (eps=0.3)", DBSCAN(eps=0.3, min_samples=5)),
        ("GMM (2 componentes)", GaussianMixture(n_components=2, random_state=SEED)),
    ]


fig, axs = plt.subplots(2, 4, figsize=(14, 7))
resumen = []
for fila, (nombre, X) in enumerate((("blobs", X_blobs), ("moons", X_moons))):
    Xs = StandardScaler().fit_transform(X)
    for col, (titulo, modelo) in enumerate(algoritmos(nombre)):
        etiquetas = modelo.fit_predict(Xs)
        ax = axs[fila, col]
        pinta(ax, X, etiquetas)
        ax.set_title(titulo)
        limpia(ax)
        n_grupos = len(set(etiquetas) - {-1})
        n_ruido = int((etiquetas == -1).sum())
        resumen.append((nombre, titulo, n_grupos, n_ruido))
axs[0, 0].set_ylabel("make_blobs", fontsize=11, color=INK, fontweight="bold")
axs[1, 0].set_ylabel("make_moons", fontsize=11, color=INK, fontweight="bold")
fig.suptitle("Mismos datos, cuatro algoritmos: la forma de los grupos decide quién acierta", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "03_comparacion_algoritmos.png", dpi=150)
plt.close(fig)
for r in resumen:
    print("RESUMEN", r)

# ---------------------------------------------------------------------------
# 04. Dendrograma (jerárquico aglomerativo) con un subconjunto pequeño
# ---------------------------------------------------------------------------
X_peq = X_blobs[np.random.default_rng(1).choice(len(X_blobs), 30, replace=False)]
Z = linkage(X_peq, method="ward")
altura_corte = (Z[-3, 2] + Z[-2, 2]) / 2  # corte a mitad de camino que deja 3 grupos

fig, ax = plt.subplots(figsize=(9, 4.2))
dendrogram(
    Z,
    ax=ax,
    color_threshold=0,
    above_threshold_color=COLORS[0],
    no_labels=True,
)
ax.axhline(altura_corte, color=COLORS[1], linestyle="--", linewidth=1.5)
ax.text(
    ax.get_xlim()[1],
    altura_corte,
    "  corte → 3 grupos",
    color=INK,
    va="bottom",
    ha="right",
    fontweight="bold",
)
ax.set_ylabel("distancia entre grupos al unirse")
ax.set_xlabel("puntos (30 muestras)")
ax.set_title("Dendrograma (Ward): cortar a otra altura da otro número de grupos")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
limpia(ax, ticks=True)
ax.set_xticks([])
fig.tight_layout()
fig.savefig(OUT / "04_dendrograma.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 05. DBSCAN: puntos core, frontera y ruido
# ---------------------------------------------------------------------------
Xm, _ = make_moons(n_samples=200, noise=0.07, random_state=SEED)
atipicos = np.array([[-0.9, 1.0], [2.0, 1.1], [0.5, -0.9], [1.9, -0.6]])
Xd = np.vstack([Xm, atipicos])
EPS, MIN_S = 0.2, 5
db = DBSCAN(eps=EPS, min_samples=MIN_S).fit(Xd)
etq = db.labels_
es_core = np.zeros(len(Xd), dtype=bool)
es_core[db.core_sample_indices_] = True

fig, ax = plt.subplots(figsize=(8, 5))
for e in sorted(set(etq) - {-1}):
    core = (etq == e) & es_core
    borde = (etq == e) & ~es_core
    ax.scatter(Xd[core, 0], Xd[core, 1], s=34, c=COLORS[e % 3], marker=MARKERS[e % 3], edgecolors=SURFACE, linewidths=0.6)
    ax.scatter(Xd[borde, 0], Xd[borde, 1], s=34, facecolors=SURFACE, edgecolors=COLORS[e % 3], marker=MARKERS[e % 3], linewidths=1.4)
ruido = etq == -1
ax.scatter(Xd[ruido, 0], Xd[ruido, 1], s=60, c=NOISE_COLOR, marker="x", linewidths=1.6)

# círculo de radio eps alrededor de un punto core, para ver qué significa eps
idx = db.core_sample_indices_[10]
ax.add_patch(plt.Circle(Xd[idx], EPS, fill=False, edgecolor=INK, linestyle="--", linewidth=1.2))
leyenda = [
    Line2D([0], [0], color=INK, linestyle="--", linewidth=1.2, label=f"radio eps = {EPS} alrededor de un punto"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[0], markeredgecolor=SURFACE, markersize=8, label="core (zona densa)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=COLORS[0], markeredgewidth=1.4, markersize=8, label="frontera (border)"),
    Line2D([0], [0], marker="x", color="none", markeredgecolor=NOISE_COLOR, markeredgewidth=1.6, markersize=8, label="ruido (noise)"),
]
ax.legend(handles=leyenda, loc="lower left", frameon=False, fontsize=9)
ax.set_title(f"DBSCAN (eps={EPS}, min_samples={MIN_S}): core, frontera y ruido")
ax.set_aspect("equal", adjustable="datalim")
limpia(ax)
fig.tight_layout()
fig.savefig(OUT / "05_dbscan_core_border_noise.png", dpi=150)
plt.close(fig)
print("RESUMEN dbscan", len(set(etq) - {-1}), "grupos,", int(ruido.sum()), "ruido,", int(es_core.sum()), "core")

# ---------------------------------------------------------------------------
# 06. Método del codo y coeficiente de silueta (make_blobs, K de 2 a 8)
# ---------------------------------------------------------------------------
Ks = list(range(2, 9))
inercias, siluetas = [], []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_blobs)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_blobs, km.labels_))
mejor_k = Ks[int(np.argmax(siluetas))]

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, y, titulo, ylab in (
    (axs[0], inercias, "Método del codo", "inercia"),
    (axs[1], siluetas, "Coeficiente de silueta", "silueta media"),
):
    ax.plot(Ks, y, color=COLORS[0], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_title(titulo)
    ax.set_xlabel("número de clusters (K)")
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    limpia(ax, ticks=True)
    ax.axvline(3, color=COLORS[1], linestyle="--", linewidth=1.2)
axs[0].text(3.1, max(inercias) * 0.85, "codo en K=3", color=INK, fontweight="bold")
axs[1].text(4.2, max(siluetas) - 0.04, f"máximo en K={mejor_k}", color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "06_codo_y_silueta.png", dpi=150)
plt.close(fig)
print("RESUMEN mejor K por silueta:", mejor_k, [round(s, 3) for s in siluetas])
print("Listo: imágenes en", OUT.resolve())
