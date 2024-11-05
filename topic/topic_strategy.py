from abc import ABC, abstractmethod
import asyncio
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# constants
DEFAULT_VOCAB_CONCATENATOR = ";"
NOISE_LABEL = -1


class TopicStrategyName(Enum):
    FREQUENCY = "frequency"
    TFIDF = "tfidf"
    K_NEIGHBOR_TFIDF = "k_neighbor_tfidf"
    SIBLINGS_TFIDF = "siblings_tfidf"


class TopicStrategy(ABC):
    @abstractmethod
    def get_cluster_topics(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        **kwargs: Any,
    ) -> Dict[int, str]:
        pass


class FrequencyTopicStrategy(TopicStrategy):
    def get_cluster_topics(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
    ) -> Dict[int, str]:
        unique_labels = set(
            labels
        )  # TODO: refactor to use clusterer.labels_.max() instead (increments from 0, may include -1)
        unique_labels.discard(NOISE_LABEL)

        cluster_terms = defaultdict(list)
        for index, label in enumerate(labels):
            cluster_terms[label].extend(
                [
                    t
                    for t in points_df.iloc[
                        index, points_df.columns.get_loc(topic_vocabulary_fieldname)
                    ].split(DEFAULT_VOCAB_CONCATENATOR)
                ]
            )

        cluster_topics = {}
        for l in unique_labels:
            term_counter = Counter(cluster_terms[l])
            topic_terms = []
            for t, cnt in term_counter.most_common(20):
                if t not in stopwords:  # naive stopword filter
                    topic_terms.append(t)
                if len(topic_terms) >= terms_per_topic:
                    break
            cluster_topics[l] = term_concat_delimiter.join(topic_terms)
        return cluster_topics


class TFIDFTopicStrategy(TopicStrategy):
    CUSTOM_TOKENIZER_DELIMITER = ";"

    def _custom_tokenizer(self, text: str) -> List[str]:
        return text.split(self.CUSTOM_TOKENIZER_DELIMITER)

    def get_cluster_topics(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        cluster_terms = defaultdict(list)
        for label, terms in zip(labels, points_df[topic_vocabulary_fieldname]):
            if label != NOISE_LABEL and isinstance(terms, str):  # check type as it can be NaN
                cluster_terms[label].append(terms)

        documents = [
            self.CUSTOM_TOKENIZER_DELIMITER.join(terms)
            for terms in cluster_terms.values()
        ]

        vectorizer = TfidfVectorizer(tokenizer=self._custom_tokenizer, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(documents)
        original_terms = {v: k for k, v in vectorizer.vocabulary_.items()}

        cluster_topics = {}
        for label, scores in zip(cluster_terms.keys(), tfidf_matrix.toarray()):
            topic_terms = []
            for idx in scores.argsort()[
                ::-1
            ]:  # Sort indices by score in descending order
                candidate = original_terms[idx]
                if candidate in stopwords or (dynamic_stopwords is not None and candidate in dynamic_stopwords(label)):
                    continue

                topic_terms.append(candidate)

                if len(topic_terms) >= terms_per_topic:
                    break

            cluster_topics[label] = term_concat_delimiter.join(topic_terms)

        return cluster_topics


class KNeighborsTFIDFTopicStrategy(TopicStrategy):
    CUSTOM_TOKENIZER_DELIMITER = ";"

    def _custom_tokenizer(self, text: str) -> List[str]:
        return text.split(self.CUSTOM_TOKENIZER_DELIMITER)

    def get_cluster_topics(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        centers: pd.DataFrame,
        k: int,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._get_cluster_topics_helper(
                labels,
                points_df,
                centers,
                k,
                stopwords,
                topic_vocabulary_fieldname,
                terms_per_topic,
                term_concat_delimiter,
                dynamic_stopwords,
            )
        )

    async def _get_cluster_topics_helper(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        centers: pd.DataFrame,
        k: int,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        """
        Get the cluster topics using the k-nearest neighbor TF-IDF algorithm. We use asyncio to parallelize the processing of each cluster group.
        """
        cluster_terms = defaultdict(list)
        for label, terms in zip(labels, points_df[topic_vocabulary_fieldname]):
            if label != NOISE_LABEL and isinstance(terms, str):
                cluster_terms[label].append(terms)

        point_to_k_closest, k_group_to_points = self._calculate_nearest_neighbors(
            centers, k
        )

        tasks = [
            self._get_cluster_topics_within_k_neighbors(
                k_neighbors,
                set(cluster_labels_to_define),
                cluster_terms,
                stopwords,
                terms_per_topic,
                term_concat_delimiter,
                dynamic_stopwords,
            )
            for k_neighbors, cluster_labels_to_define in k_group_to_points.items()
        ]

        results = await asyncio.gather(*tasks)
        return dict(item for result in results for item in result.items())

    def _calculate_nearest_neighbors(self, df, k):
        points_array = df[["x", "y"]].to_numpy()
        distances = np.linalg.norm(points_array[:, np.newaxis] - points_array, axis=-1)

        k_closest = np.argsort(distances, axis=1)[:, :k]
        point_to_k_closest = {
            df.iloc[i]["labels"]: df.iloc[k_closest[i]]["labels"].tolist()
            for i in range(len(df))
        }

        # Invert the map to prevent redundant calculations
        k_groups_to_points = defaultdict(list)
        for point, k_group in point_to_k_closest.items():
            k_groups_to_points[tuple(sorted(k_group))].append(point)

        return point_to_k_closest, dict(k_groups_to_points)

    async def _get_cluster_topics_within_k_neighbors(
        self,
        k_neighbor_labels: List[int],
        cluster_labels_to_define: Set[int],
        cluster_terms: Dict[int, List[str]],
        stopwords: Set[str],
        terms_per_topic: int,
        term_concat_delimiter: str,
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        vectorizer = TfidfVectorizer(tokenizer=self._custom_tokenizer, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(
            [
                self.CUSTOM_TOKENIZER_DELIMITER.join(cluster_terms[label])
                for label in k_neighbor_labels
            ]
        )
        original_terms = vectorizer.get_feature_names_out()

        cluster_topics = {}
        for label, scores in zip(k_neighbor_labels, tfidf_matrix.toarray()):
            if (
                label not in cluster_labels_to_define
            ):  # TODO: improve efficiency (i.e., running for-loop over all neighbors is inefficient)
                continue

            topic_terms = [
                term
                for term in original_terms[scores.argsort()[::-1]]
                if term not in stopwords and (dynamic_stopwords is None or term not in dynamic_stopwords(label))
            ][:terms_per_topic]
            cluster_topics[label] = term_concat_delimiter.join(topic_terms)
        return cluster_topics


class SiblingsTFIDFTopicStrategy(TopicStrategy):
    CUSTOM_TOKENIZER_DELIMITER = ";"

    def _custom_tokenizer(self, text: str) -> List[str]:
        return text.split(self.CUSTOM_TOKENIZER_DELIMITER)

    def get_cluster_topics(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        topic_tree: TopicTree,
        treenode_prefix: str,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._get_cluster_topics_helper(
                labels,
                points_df,
                topic_tree,
                treenode_prefix,
                stopwords,
                topic_vocabulary_fieldname,
                terms_per_topic,
                term_concat_delimiter,
                dynamic_stopwords,
            )
        )

    def _get_sibling_groups(self, labels: List[int], topic_tree: TopicTree, treenode_prefix: str) -> List[List[int]]:
        sibling_groups, found = [], set()
        for label in labels:
            if label in found:
                continue

            sibling_group = [
                int(label[len(treenode_prefix):]) 
                for label in topic_tree.get_siblings_inclusive(
                    f"{treenode_prefix}{label}",
                )
            ]

            sibling_groups.append(sibling_group)
            found.update(sibling_group)

        res = [group for group in sibling_groups if len(group) > 1]

        singletons = [group[0] for group in sibling_groups if len(group) == 1 and group[0] != NOISE_LABEL]
        res.append(singletons)                
        return res

    async def _get_cluster_topics_helper(
        self,
        labels: List[int],
        points_df: pd.DataFrame,
        topic_tree: TopicTree,
        treenode_prefix: str,
        stopwords: Set[str] = set(),
        topic_vocabulary_fieldname: str = "mesh_terms",
        terms_per_topic: int = 1,
        term_concat_delimiter: str = "; ",
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        """
        Get the cluster topics using the siblings TF-IDF algorithm. We use asyncio to parallelize the processing of each cluster group.
        """
        cluster_terms = defaultdict(list)
        for label, terms in zip(labels, points_df[topic_vocabulary_fieldname]):
            if label != NOISE_LABEL and isinstance(terms, str):
                cluster_terms[label].append(terms)

        sibling_groups = self._get_sibling_groups(labels, topic_tree, treenode_prefix)

        print(f"Sibling groups: {sibling_groups}")

        tasks = [
            self._get_cluster_topics(
                labels,
                cluster_terms,
                stopwords,
                terms_per_topic,
                term_concat_delimiter,
                dynamic_stopwords,
            )
            for labels in sibling_groups
        ]

        results = await asyncio.gather(*tasks)
        return dict(item for result in results for item in result.items())

    async def _get_cluster_topics(
        self,
        labels: List[int],
        cluster_terms: Dict[int, List[str]],
        stopwords: Set[str],
        terms_per_topic: int,
        term_concat_delimiter: str,
        dynamic_stopwords: Callable[[int], Set[str]] | None = None,
    ) -> Dict[int, str]:
        # TODO: centralize redundant code
        vectorizer = TfidfVectorizer(tokenizer=self._custom_tokenizer, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(
            [
                self.CUSTOM_TOKENIZER_DELIMITER.join(cluster_terms[label])
                for label in labels
            ]
        )
        original_terms = vectorizer.get_feature_names_out()

        cluster_topics = {}
        for label, scores in zip(cluster_terms.keys(), tfidf_matrix.toarray()):
            topic_terms = [
                term
                for term in original_terms[scores.argsort()[::-1]]
                if term not in stopwords and (dynamic_stopwords is None or term not in dynamic_stopwords(label))
            ][:terms_per_topic]
            cluster_topics[label] = term_concat_delimiter.join(topic_terms)
        return cluster_topics
