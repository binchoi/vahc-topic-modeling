
from collections import Counter, defaultdict
from dataclasses import asdict
from enum import Enum
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from topic.builder import TopicJSONBuilder
from topic.clusterer import Clusterer, HDBSCANClusterer
from topic.config import TopicAnnotatorConfig
from topic.topic_strategy import FrequencyTopicStrategy, KNeighborsTFIDFTopicStrategy, TFIDFTopicStrategy, TopicStrategyName
from topic.topic_tree import TopicTree


NOISE_LABEL = -1


class TopicAnnotator:
    """
    Identifies and labels regions of the provided semantic map with relevant topics
    """
    clusterer: Clusterer = HDBSCANClusterer()
    configs: Dict[int, TopicAnnotatorConfig] = {}
    builder: TopicJSONBuilder = TopicJSONBuilder()

    def __init__(self, configs: Dict[int, TopicAnnotatorConfig], stopwords=set(), clusterer: Clusterer = HDBSCANClusterer()):
        self.clusterer = clusterer
        self.configs = configs
        self.stopwords = stopwords
        self.topic_cache : Dict[str, List[str]] = {}  # topic_key -> List[topic_names] (e.g., L0_38 -> ['cancer', 'tumor'])
        self.topic_tree : TopicTree = TopicTree()

    def _get_delimiter(self, filename):
        if filename.endswith(".tsv"):
            return "\t"
        elif filename.endswith(".csv"):
            return ","
        else:
            print(
                f"* ERROR: couldn't determine the delimiter for the embeddings file. Using default delimiter: \t"
            )
            return "\t"

    def generate_topic_annotations_and_export(self, path_tsv, path_embd, output_path):
        delimiter = self._get_delimiter(path_tsv)

        points = self._load_2d_points(path_embd)
        points_df = self._get_points_df(path_tsv, delimiter, points)

        labels_cache, self.topic_cache, self.topic_tree = {}, {}, TopicTree()
        levels = []
        for level_key, config in sorted(self.configs.items(), key=lambda x: x[0], reverse=True):
            if not self._fulfills_prereq_condition(points, config.prereq_condition):
                print(f"* prereq condition for this topic level is not met: {config.prereq_condition}; skipped")
                continue

            print(f"* begin clustering: {config.clustering_configs}")
            labels, _, centers = self.clusterer.cluster(
                points, config.clustering_configs
            )
            labels_cache[level_key] = labels
            print(f"* clustering successful")
            
            self._update_topic_tree(labels_cache, level_key)
            print(f"* topic tree updated")

            print(f"* begin topic modeling: {config.topic_determination_configs}")
            topic_tree_stopwords = lambda label: self._get_topic_tree_stopwords(level_key, label)
            cluster_topics = self._get_cluster_topics(
                labels,
                centers,
                points_df,
                self.stopwords,
                config.topic_determination_configs,
                config.labeling_configs,
                topic_tree_stopwords,
                level_key,
            )
            self._update_topic_cache(level_key, cluster_topics, config.labeling_configs)
            print(f"* topic modeling successful")

            level_data = self.builder.get_topic_level_dict(
                level_key, asdict(config), labels, centers, points_df, cluster_topics
            )
            levels.append(level_data)

        levels = sorted(levels, key=lambda x: x["level"])

        print(f"* dumping topic json to {output_path}")
        with open(output_path, "w") as f:
            json.dump(
                {"levels": levels},
                f,
                indent=2,
                default=lambda obj: obj.value if isinstance(obj, Enum) else obj,
            )

        print(f"* generated {output_path}")

    def _fulfills_prereq_condition(self, points, condition):
        if "min_length" in condition:
            return len(points) >= condition["min_length"]
        return True
    
    def _update_topic_cache(self, level_key, cluster_topics, labeling_configs):
        self.topic_cache.update({
            self._global_topic_key(level_key, label): topic.split(labeling_configs.get("term_concat_delimiter", "; ")) 
            for label, topic in cluster_topics.items()
        })
    
    def _global_topic_key(self, level_key, label_idx):
        return f"L{level_key}_{label_idx}"
    
    def _get_topic_tree_stopwords(self, level_key, label):
        """
        Dynamically identfies stopwords based on the topic cluster's lineage in the topic tree.
        Currently, it retrieves the stopwords from the parent topic clusters (ancestors).
        """
        ancestors = self.topic_tree.get_ancestors(self._global_topic_key(level_key, label))
        stopwords = set(topic for ancestor in ancestors for topic in self.topic_cache.get(ancestor, []))
        return stopwords
    
    def _update_topic_tree(self, labels_cache, level_key):
        LINEAGE_THRESHOLD = 0.5
        child_level_key, parent_level_key = level_key, level_key+1

        child_level_labels, parent_level_labels = labels_cache.get(child_level_key), labels_cache.get(parent_level_key)
        if any([child_level_labels is None, parent_level_labels is None]):
            return
        
        child_to_parent_candidates = defaultdict(Counter)
        for child_label, parent_label in zip(child_level_labels, parent_level_labels):
            if child_label == NOISE_LABEL:
                continue
            child_to_parent_candidates[child_label][parent_label] += 1
    
        new_edges = [
            (
                self._global_topic_key(child_level_key, child_label),
                self._global_topic_key(parent_level_key, parent_candidates.most_common(1)[0][0])
            )
            for child_label, parent_candidates in child_to_parent_candidates.items()
            if (
                parent_candidates.most_common(1)[0][0] != NOISE_LABEL and
                parent_candidates.most_common(1)[0][1] / sum(parent_candidates.values()) > LINEAGE_THRESHOLD
            )
        ]

        self.topic_tree.add_edges(new_edges)

    def _get_cluster_topics(
        self,
        labels,
        centers,
        points_df,
        stopwords,
        topic_determination_configs,
        labeling_configs,
        dynamic_stopwords = None,
        level_key = None,
    ):
        if topic_determination_configs["strategy"] == TopicStrategyName.FREQUENCY:
            return FrequencyTopicStrategy().get_cluster_topics(
                labels=labels,
                points_df=points_df,
                stopwords=stopwords,
                terms_per_topic=labeling_configs["terms_per_topic"],
                term_concat_delimiter=labeling_configs.get(
                    "term_concat_delimiter", "; "
                ),
            )
        elif topic_determination_configs["strategy"] == TopicStrategyName.K_NEIGHBOR_TFIDF:
            return KNeighborsTFIDFTopicStrategy().get_cluster_topics(
                labels=labels,
                points_df=points_df,
                centers=centers,
                k=topic_determination_configs["k"],
                stopwords=stopwords,
                terms_per_topic=labeling_configs["terms_per_topic"],
                term_concat_delimiter=labeling_configs.get(
                    "term_concat_delimiter", "; "
                ),
                dynamic_stopwords=dynamic_stopwords,
            )
        elif topic_determination_configs["strategy"] == TopicStrategyName.SIBLINGS_TFIDF:
            if level_key is None:
                raise ValueError("level_key is required for SIBLINGS_TFIDF strategy")
            return SiblingsTFIDFTopicStrategy().get_cluster_topics(
                labels=labels,
                points_df=points_df,
                topic_tree=self.topic_tree,
                treenode_prefix=self._global_topic_key(level_key, ""),
                stopwords=stopwords,
                terms_per_topic=labeling_configs["terms_per_topic"],
                term_concat_delimiter=labeling_configs.get(
                    "term_concat_delimiter", "; "
                ),
                dynamic_stopwords=dynamic_stopwords,
            )
        # TopicStrategyName.TFIDF as default
        return TFIDFTopicStrategy().get_cluster_topics(
            labels=labels,
            points_df=points_df,
            stopwords=stopwords,
            terms_per_topic=labeling_configs["terms_per_topic"],
            term_concat_delimiter=labeling_configs.get("term_concat_delimiter", "; "),
            dynamic_stopwords=dynamic_stopwords,
        )

    def _load_2d_points(self, filename):
        points = np.load(filename)
        print('* loaded coordinates from %s' % filename)
        print('* printing head of embd coordinates')
        print(points[:5])
        return points

    def _get_points_df(
        self,
        path_tsv,
        delimiter,
        points,
    ):
        df = pd.read_csv(
            path_tsv,
            sep=delimiter
        )
        print("* loaded df")

        # merge embds
        df['x'] = points[:, 0]
        df['y'] = points[:, 1]
        
        # drop unnecessary columns
        df.drop(columns=['title', 'journal', 'year', 'abstract'], inplace=True)  

        print('* merged embds to df')
        return df

