from collections import defaultdict


NOISE_LABEL = -1


class TopicJSONBuilder:
    def get_topic_level_dict(
        self, level_key, meta, labels, centers, points_df, cluster_topics
    ):
        cluster_point_indices = defaultdict(list)
        for i, label in enumerate(labels):
            if label != NOISE_LABEL:
                cluster_point_indices[label].append(i)

        topics = [
            {
                "name": cluster_topics.get(label, ""),
                "x": float(center["x"]),
                "y": float(center["y"]),
                "pmid_count": len(cluster_pmids),
                "pmids": cluster_pmids,
            }
            for label, indices in cluster_point_indices.items()
            if (center := centers.loc[centers["labels"] == label].iloc[0]) is not None
            and (cluster_pmids := points_df.iloc[indices]["pmid"].tolist()) is not None
        ]

        return {"level": level_key, "meta": meta, "topics": topics}

