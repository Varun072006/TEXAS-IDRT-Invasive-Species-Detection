"""
Unsupervised clustering of tree-crown visual embeddings.

This is an exploratory step towards future species identification:
before any labeled species data exists, clustering visually similar
crowns together helps a domain expert spot candidate groups (e.g. a
suspected invasive species standing out from native canopy) for
targeted field verification and labeling.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    HDBSCAN_AVAILABLE = False


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: int = 5,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """Cluster a set of crown embeddings and return an integer label per row.

    algorithm="kmeans": fixed number of clusters, fast, good default.
    algorithm="hdbscan": density-based, discovers cluster count automatically
        and flags outliers as -1 (useful for spotting rare/unusual crowns).
    """
    scaled = StandardScaler().fit_transform(embeddings)

    if algorithm == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is not installed; `pip install hdbscan` or use algorithm='kmeans'")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(scaled)

    if algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        return clusterer.fit_predict(scaled)

    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def project_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """PCA projection of high-dimensional embeddings to 2D for visualization."""
    scaled = StandardScaler().fit_transform(embeddings)
    return PCA(n_components=2, random_state=42).fit_transform(scaled)
