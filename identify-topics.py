import logging
from topic.config import TopicAnnotatorConfig
from topic.topic_annotator import TopicAnnotator
from topic.topic_strategy import TopicStrategyName

"""
Topic Modeling

This file contains functions used for topic modeling and generating the JSON topic model file used to enable the exploration of topic clusters in the semantic space.

Currently, it takes the 2D dimension reduced embeddings as the input and generates the topic model file.
In the future, it may compute the clusters required for the topic modeling from a higher dimensional space.
"""

# constants
GENERIC_STOPWORDS = {
    "Human",
    "Humans",
    "Mice",
    "Female",
    "Male",
    "Aged",
    "Middle Aged",
    "Animals",
    "Adult",
    "United States",
    "",
}


PARAMETER_TUNED_CONFIGS = {
    0: TopicAnnotatorConfig(
        prereq_condition={"min_length": 0},
        clustering_configs={"min_cluster_size": 10, "min_samples": 7},
        topic_determination_configs={
            "strategy": TopicStrategyName.K_NEIGHBOR_TFIDF,
            "k": 5,
        },
        labeling_configs={"terms_per_topic": 2, "term_concat_delimiter": "; "},
    ),
    1: TopicAnnotatorConfig(
        prereq_condition={"min_length": 500},
        clustering_configs={"min_cluster_size": 20, "min_samples": 10},
        topic_determination_configs={"strategy": TopicStrategyName.TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
    2: TopicAnnotatorConfig(
        prereq_condition={"min_length": 2000},
        clustering_configs={"min_cluster_size": 150, "min_samples": 30},
        topic_determination_configs={"strategy": TopicStrategyName.TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
    3: TopicAnnotatorConfig(
        prereq_condition={"min_length": 100000},
        clustering_configs={"min_cluster_size": 500, "min_samples": 400},
        topic_determination_configs={"strategy": TopicStrategyName.TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
}


SIB_PARAMETER_TUNED_CONFIGS = {
    0: TopicAnnotatorConfig(
        prereq_condition={"min_length": 0},
        clustering_configs={"min_cluster_size": 10, "min_samples": 7},
        topic_determination_configs={
            "strategy": TopicStrategyName.SIBLINGS_TFIDF,
        },
        labeling_configs={"terms_per_topic": 2, "term_concat_delimiter": "; "},
    ),
    1: TopicAnnotatorConfig(
        prereq_condition={"min_length": 500},
        clustering_configs={"min_cluster_size": 20, "min_samples": 10},
        topic_determination_configs={"strategy": TopicStrategyName.SIBLINGS_TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
    2: TopicAnnotatorConfig(
        prereq_condition={"min_length": 2000},
        clustering_configs={"min_cluster_size": 150, "min_samples": 30},
        topic_determination_configs={"strategy": TopicStrategyName.SIBLINGS_TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
    3: TopicAnnotatorConfig(
        prereq_condition={"min_length": 100000},
        clustering_configs={"min_cluster_size": 500, "min_samples": 400},
        topic_determination_configs={"strategy": TopicStrategyName.SIBLINGS_TFIDF},
        labeling_configs={"terms_per_topic": 1},
    ),
}

"""
TODO: The updated method on production takes only the configuration for the lowest level of the hierarchy 
and dynamically determines the configurations for subsequent levels based on child-to-parent thresholds. 
This improves the flexibility and robustness of our method across varying semantic map structures. 

A patch containing this update will be available in the next release.
"""


def identify_topics(dataset, path_tsv, path_embd, path_out=None):
    print("* begin topic identification")
    logging.log(
        dataset=dataset,
        stage='identify_topics',
        message='Identifying topic regions within the semantic map...',
    )
    path_out = path_out or path_tsv.replace('.raw', '_topics.json')  # TODO: improve naming convention

    annotator = TopicAnnotator(configs=PARAMETER_TUNED_CONFIGS, stopwords=GENERIC_STOPWORDS)
    annotator.generate_topic_annotations_and_export(path_tsv, path_embd, path_out)

    print('* saved topics to %s' % path_out)
    logging.log(
        dataset=dataset,
        stage='identify_topics',
        message='Topic regions successfully identified',
    )
    return path_out
