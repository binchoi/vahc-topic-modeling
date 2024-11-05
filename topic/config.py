from dataclasses import dataclass


@dataclass
class TopicAnnotatorConfig:
    prereq_condition: dict  # e.g. {'min_length': 0}
    clustering_configs: dict  # e.g. {'min_cluster_size': 20, 'min_samples': 10}
    topic_determination_configs: (
        dict  # e.g. {'strategy': 'tfidf"}
    )
    labeling_configs: dict  # e.g. {'terms_per_topic': 1, 'term_concat_delimiter': '; '}
