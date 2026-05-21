import time

import numpy as np
from usearch.index import Index

from ..base.module import BaseANN


def translate_metric(metric):
    if metric == "angular":
        return "ip"
    if metric == "euclidean":
        return "l2sq"
    raise ValueError("zeelin-usearch supports angular and euclidean distances")


class ZeelinUSearch(BaseANN):
    def __init__(self, metric, index_param):
        self.metric = translate_metric(metric)
        self.normalize = metric == "angular"
        self.index_param = index_param
        self.name = "zeelin-usearch"

    def fit(self, X):
        start = time.time()
        X = np.asarray(X, dtype=np.float32)
        if self.normalize:
            norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            X = X / norms
        self.index = Index(
            ndim=X.shape[1],
            metric=self.metric,
            dtype=self.index_param.get("dtype", "f32"),
            connectivity=int(self.index_param["connectivity"]),
            expansion_add=int(self.index_param["expansion_add"]),
            expansion_search=int(self.index_param.get("expansion_search", 40)),
        )
        self.index.add(np.arange(len(X), dtype=np.uint64), X, threads=1)
        self.build_time = time.time() - start

    def set_query_arguments(self, expansion_search):
        self.expansion_search = int(expansion_search)
        self.index.expansion_search = self.expansion_search
        self.name = "zeelin-usearch (%s, expansion_search=%s)" % (
            self.index_param,
            self.expansion_search,
        )

    def query(self, v, n):
        v = np.asarray(v, dtype=np.float32)
        if self.normalize:
            norm = np.linalg.norm(v)
            if norm != 0:
                v = v / norm
        matches = self.index.search(v, n, threads=1)
        return matches.keys

    def batch_query(self, X, n):
        X = np.asarray(X, dtype=np.float32)
        if self.normalize:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.res = [self.index.search(q, n, threads=1).keys for q in X]

    def get_batch_results(self):
        return self.res
