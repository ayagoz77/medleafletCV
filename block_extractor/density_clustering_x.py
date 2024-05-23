import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from typing import List, Tuple


def find_min_x_by_line(points, tolerance=20):
    x_groups = {}
    for x in points:
        if not x_groups:
            x_groups[x] = 1
            continue

        found_group = False
        for x2 in x_groups.keys():
            if abs(x - x2) < tolerance:
                x_groups[x2] += 1
                found_group = True
                break

        if not found_group:
            x_groups[x] = 1
    line = sorted(x_groups.items(), key=lambda x: x[1], reverse=True)[0]
    return line[0]


class TextBBoxCluster:
    def __init__(self, eps=40, min_samples=2) -> None:
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    def __call__(self, bboxes: List[Tuple]):
        X = np.array([[bbox[0], bbox[1]] for bbox in bboxes])

        dbscan = self.clusterer.fit(X)
        labels = dbscan.labels_

        clusters = defaultdict(list)
        for bbox, label in zip(bboxes, labels):
            if label != -1:  # -1 is noise in DBSCAN
                clusters[label].append(bbox)

        for cluster_id in clusters:
            clusters[cluster_id] = sorted(
                clusters[cluster_id], key=lambda bbox: bbox[1]
            )

        cluster_stats = {
            cluster_id: (
                len(bboxes),
                find_min_x_by_line(sorted(np.array(bboxes)[:, 0])),
            )
            for cluster_id, bboxes in clusters.items()
        }
        sorted_cl = sorted(
            cluster_stats.items(), key=lambda item: item[1][0], reverse=True
        )
        final_clusters_x = [sorted_cl[0][1]]

        for s in sorted_cl[1:]:
            if s[1][0] > final_clusters_x[-1][0] / 2:
                final_clusters_x.append(s[1])
        print(final_clusters_x)
        unique_sort_cl = sorted(list(set(np.array(final_clusters_x)[:, 1])))
        return unique_sort_cl

    def cluster_by_x_coordinate(self, bboxes, cluster_starts):
        clusters = {i[1]: [] for i in cluster_starts}
        print(clusters)
        for bbox in bboxes:
            x_center = bbox[0] + bbox[2]
            for i, start_x in cluster_starts:
                if x_center <= start_x:
                    clusters[start_x].append(bbox)
                    break
            else:
                clusters[cluster_starts[-1][1]].append(bbox)

        return clusters
