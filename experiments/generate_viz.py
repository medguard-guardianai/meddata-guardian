import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# ── Dataset: week-by-week topics for 9 labeled courses ────────────────────────
# Format: (short_name, country, weeks_list_of_topic_sets)
courses = [
    ("CMU 10-601", "USA", [
        ["intro"],
        ["decision trees"],
        ["K nearest neighbors", "performance evaluation", "perceptrons"],
        ["linear regression", "optimization"],
        ["optimization", "logistic regression", "feature engineering", "regularization"],
        ["neural networks"],
        ["neural networks", "ethics for machine learning"],
        ["learning theory", "maximum likelihood"],
        ["convolutional neural networks", "recurrent neural networks", "transformers"],
        ["reinforcement learning"],
        ["reinforcement learning"],
        ["recommender systems", "ensembles"],
        ["clustering", "dimensionality reduction", "generative models"],
    ]),
    ("Stanford CS 229", "USA", [
        ["intro", "supervised learning", "linear regression"],
        ["logistic regression", "statistical learning"],
        ["Bayesian learning", "classification"],
        ["kernels", "support vector machines", "neural networks"],
        ["neural networks", "regularization", "performance evaluation"],
        ["decision trees", "ensembles", "clustering", "probabilistic reasoning"],
        ["EM algorithm", "dimensionality reduction"],
        ["unsupervised learning", "reinforcement learning"],
        ["reinforcement learning"],
        ["ethics for machine learning", "explainable AI"],
    ]),
    ("UIUC CS 446", "USA", [
        ["intro", "K nearest neighbors", "Bayesian learning"],
        ["linear regression", "logistic regression"],
        ["support vector machines", "kernels"],
        ["decision trees", "ensembles"],
        ["learning theory"],
        ["perceptrons", "neural networks"],
        ["convolutional neural networks", "recurrent neural networks"],
        ["transformers"],
        ["dimensionality reduction", "clustering"],
        ["probabilistic reasoning"],
        ["encoder-decoder architectures", "generative models"],
        ["diffusion models", "self-supervised learning"],
        ["language models", "reinforcement learning"],
        ["reinforcement learning"],
    ]),
    ("UMD CMSC 422", "USA", [
        ["intro", "decision trees"],
        ["ensembles", "K nearest neighbors"],
        ["K nearest neighbors", "perceptrons"],
        ["classification"],
        ["gradient descent"],
        ["Bayesian learning", "logistic regression"],
        ["logistic regression"],
        ["classification", "neural networks"],
        ["neural networks"],
        ["neural networks"],
        ["convolutional neural networks"],
        ["convolutional neural networks", "clustering"],
        ["dimensionality reduction"],
        ["support vector machines"],
    ]),
    ("U Toronto CSC 311", "Canada", [
        ["intro", "K nearest neighbors"],
        ["decision trees", "learning theory"],
        ["linear regression"],
        ["logistic regression"],
        ["neural networks"],
        ["neural networks"],
        ["probabilistic reasoning"],
        ["probabilistic reasoning", "Bayesian learning"],
        ["dimensionality reduction", "recommender systems"],
        ["ethics for machine learning", "recommender systems"],
        ["clustering", "EM algorithm"],
        ["reinforcement learning"],
    ]),
    ("UBC CPSC 340", "Canada", [
        ["intro"],
        ["decision trees", "learning theory", "Bayesian learning", "K nearest neighbors"],
        ["K nearest neighbors", "ensembles"],
        ["clustering"],
        ["linear regression"],
        ["gradient descent"],
        ["feature engineering", "regularization"],
        ["perceptrons", "support vector machines", "feature engineering"],
        ["convolutional neural networks", "kernels", "optimization"],
        ["ensembles"],
        ["maximum likelihood", "dimensionality reduction"],
        ["recommender systems", "dimensionality reduction"],
        ["neural networks", "deep learning"],
        ["convolutional neural networks", "encoder-decoder architectures", "recurrent neural networks", "transformers"],
    ]),
    ("McGill COMP 451", "Canada", [
        ["intro", "K nearest neighbors"],
        ["perceptrons"],
        ["maximum likelihood", "Bayesian learning"],
        ["Bayesian learning", "logistic regression"],
        ["learning theory", "gradient descent"],
        ["linear regression", "overfitting"],
        ["regularization"],
        ["decision trees"],
        ["clustering", "probabilistic reasoning"],
        ["feature engineering", "dimensionality reduction"],
        ["generative models", "neural networks"],
        ["neural networks", "convolutional neural networks"],
        ["recurrent neural networks", "encoder-decoder architectures"],
    ]),
    ("U Alberta CMPUT 466", "Canada", [
        ["intro"],
        ["maximum likelihood"],
        ["supervised learning"],
        ["linear regression", "regularization", "learning theory"],
        ["learning theory", "optimization"],
        ["optimization"],
        ["logistic regression", "performance evaluation"],
        ["classification", "Bayesian learning", "generative models"],
        ["Bayesian learning", "neural networks"],
        ["performance evaluation"],
        ["performance evaluation", "neural networks"],
        ["dense vector embeddings", "neural networks"],
        ["neural networks", "generative models"],
    ]),
    ("U Waterloo CS 480", "Canada", [
        ["intro", "K nearest neighbors"],
        ["linear regression", "maximum likelihood"],
        ["maximum likelihood", "Bayesian learning", "probabilistic reasoning"],
        ["logistic regression", "perceptrons", "neural networks"],
        ["neural networks", "kernels"],
        ["deep learning", "convolutional neural networks"],
        ["probabilistic reasoning"],
        ["recurrent neural networks", "transformers"],
        ["graphical models", "encoder-decoder architectures"],
        ["generative models", "encoder-decoder architectures"],
        ["diffusion models", "ensembles"],
        ["ensembles", "support vector machines"],
    ]),
]

# Canonicalize some topic names
ALIASES = {
    "probabilistic reasoning/inference/models": "probabilistic reasoning",
    "probabilistic reasoning": "probabilistic reasoning",
    "statistical learning": "learning theory",
    "unsupervised learning": "clustering",
    "supervised learning": "supervised learning",
    "explainable AI": "explainable AI",
    "classification": "classification",
    "deep learning": "deep learning",
    "overfitting": "overfitting",
    "graphical models": "graphical models",
    "dense vector embeddings": "dense vector embeddings",
    "self-supervised learning": "self-supervised learning",
    "language models": "language models",
    "diffusion models": "diffusion models",
}

def canon(t):
    return ALIASES.get(t.strip(), t.strip())

# ── 1. TOP TOPICS BAR CHART ───────────────────────────────────────────────────
all_topics = []
for _, _, weeks in courses:
    for week in weeks:
        for t in week:
            all_topics.append(canon(t))

counts = Counter(all_topics)
# remove catch-alls
for skip in ["intro", "review", "performance evaluation", "classification",
             "supervised learning", "deep learning", "overfitting",
             "graphical models", "dense vector embeddings", "self-supervised learning",
             "language models", "diffusion models", "explainable AI", "learning theory"]:
    counts.pop(skip, None)

top_n = 18
top = counts.most_common(top_n)
labels, vals = zip(*top)

ORANGE  = "#E55A00"
DARK    = "#1A1A2E"
LGRAY   = "#F4F4F4"
MID     = "#888888"
ORANGE2 = "#FF8C2F"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

bars = ax.barh(range(len(labels)), vals, color=ORANGE, height=0.65, zorder=3)
# highlight top 3
for i in range(3):
    bars[i].set_color(ORANGE2)
    bars[i].set_linewidth(0)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, color=LGRAY, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Number of course-weeks", color=MID, fontsize=11)
ax.xaxis.label.set_color(MID)
ax.tick_params(colors=MID, axis="x")
ax.spines[:].set_visible(False)
ax.grid(axis="x", color="#333355", linewidth=0.7, zorder=0)

# value labels
for bar, v in zip(bars, vals):
    ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", color=LGRAY, fontsize=10)

ax.set_title("Top ML Topics Across 9 Labeled Syllabuses",
             color=LGRAY, fontsize=15, fontweight="bold", pad=14)
ax.text(0.99, 1.01, "week-count = topic appearances summed across all course weeks",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MID)

plt.tight_layout()
plt.savefig("/Users/shrivarshininarayanan/meddata-guardian-1/experiments/top_topics.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("Saved top_topics.png")

# ── 2. HEATMAP: TOPICS × UNIVERSITIES ────────────────────────────────────────
# pick top 20 topics by total week-count for rows
top20 = [t for t, _ in counts.most_common(20)]

# build matrix: rows=topics, cols=courses
short_names = [c[0] for c in courses]
matrix = np.zeros((len(top20), len(courses)))
for ci, (_, _, weeks) in enumerate(courses):
    for week in weeks:
        for t in week:
            t = canon(t)
            if t in top20:
                matrix[top20.index(t), ci] += 1

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

cmap = plt.get_cmap("YlOrRd")
cmap.set_under("#1A1A2E")
im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.5, vmax=matrix.max())

# axes
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=35, ha="right", color=LGRAY, fontsize=10)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20, color=LGRAY, fontsize=10)

# cell labels
for i in range(len(top20)):
    for j in range(len(short_names)):
        v = int(matrix[i, j])
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=9, color="white" if v >= 2 else "#333333", fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
cbar.set_label("Weeks spent on topic", color=LGRAY, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=LGRAY)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LGRAY)

# Canada label markers
canada_idx = [i for i, c in enumerate(courses) if c[1] == "Canada"]
for ci in canada_idx:
    ax.get_xticklabels()[ci].set_color(ORANGE2)

ax.set_title("Topic Coverage Heatmap  ·  Topics × Universities\n"
             "(orange = Canadian school)",
             color=LGRAY, fontsize=13, fontweight="bold", pad=10)

# grid lines
ax.set_xticks(np.arange(-0.5, len(short_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(top20), 1), minor=True)
ax.grid(which="minor", color="#2A2A44", linewidth=0.8)
ax.tick_params(which="minor", length=0)

plt.tight_layout()
plt.savefig("/Users/shrivarshininarayanan/meddata-guardian-1/experiments/topic_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("Saved topic_heatmap.png")

# ── 3. US + CANADA MAP ───────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

uni_coords = {
    "CMU 10-601":      (40.44, -79.99, "USA"),
    "Stanford CS 229": (37.43, -122.17, "USA"),
    "UIUC CS 446":     (40.11, -88.23, "USA"),
    "UMD CMSC 422":    (38.99, -76.94, "USA"),
    "U Toronto CSC 311":    (43.66, -79.39, "Canada"),
    "UBC CPSC 340":         (49.26, -123.25, "Canada"),
    "McGill COMP 451":      (45.50, -73.58, "Canada"),
    "U Alberta CMPUT 466":  (53.52, -113.53, "Canada"),
    "U Waterloo CS 480":    (43.47, -80.54, "Canada"),
}

if HAS_CARTOPY:
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=45))
    ax.set_extent([-130, -60, 24, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#1E1E30")
    ax.add_feature(cfeature.OCEAN, facecolor="#111122")
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="#444466")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="#333355")
else:
    # fallback: simple scatter on blank axes
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#111122")
    ax.set_xlim(-135, -55)
    ax.set_ylim(23, 62)
    ax.set_xlabel("Longitude", color=MID)
    ax.set_ylabel("Latitude", color=MID)
    ax.tick_params(colors=MID)
    ax.spines[:].set_color("#333355")

fig.patch.set_facecolor(DARK)

for name, (lat, lon, country) in uni_coords.items():
    color = ORANGE2 if country == "Canada" else ORANGE
    if HAS_CARTOPY:
        ax.plot(lon, lat, "o", color=color, markersize=12,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lon + 0.5, lat + 0.8, name.split()[0] + "\n" + name.split()[1],
                fontsize=7.5, color="white", transform=ccrs.PlateCarree(),
                zorder=6, ha="left")
    else:
        ax.plot(lon, lat, "o", color=color, markersize=12, zorder=5)
        short = name
        ax.text(lon + 0.5, lat + 0.5, short, fontsize=7.5, color="white", zorder=6)

legend_els = [
    mpatches.Patch(color=ORANGE, label="US universities (4)"),
    mpatches.Patch(color=ORANGE2, label="Canadian universities (5)"),
]
ax.legend(handles=legend_els, loc="lower right",
          facecolor=DARK, edgecolor=ORANGE, labelcolor=LGRAY, fontsize=10)

ax.set_title("Universities in Labeled Dataset  ·  9 of 50 shown",
             color=LGRAY, fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/shrivarshininarayanan/meddata-guardian-1/experiments/uni_map.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("Saved uni_map.png")

print("\nAll done! Files saved to experiments/")
