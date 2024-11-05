from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import hdbscan
import numpy as np
import pandas as pd


NOISE_LABEL = -1


class Clusterer(ABC):
    @abstractmethod
    def cluster(self, points: np.ndarray, params: Dict[str, Any]) -> Any:
        """
        Clusters the given array of points based on the provided parameters.

        :param points: A np.ndarray of points to be clustered.
        :param params: A dictionary of parameters for the clustering algorithm.
        :return: The result of clustering, the nature of which depends on the specific implementation.
        """
        pass


class HDBSCANClusterer(Clusterer):
    def cluster(
        self, points: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        clusterer = hdbscan.HDBSCAN(**params)
        clusterer.fit(points)

        df = pd.DataFrame(points, columns=["x", "y"])
        df["labels"] = clusterer.labels_
        df["probabilities"] = clusterer.probabilities_

        # Calculate the weighted average for each cluster
        weighted_mean = lambda x: np.average(
            x, weights=df.loc[x.index, "probabilities"]
        )
        cluster_centers = (
            df[df["labels"] != NOISE_LABEL]
            .groupby("labels")
            .agg({"x": weighted_mean, "y": weighted_mean})
            .reset_index()
        )
        return clusterer.labels_, clusterer.probabilities_, cluster_centers

# Only HDBSCAN is supported at the moment