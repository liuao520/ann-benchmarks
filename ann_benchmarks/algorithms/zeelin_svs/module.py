import time

import numpy as np
import svs

from ..base.module import BaseANN


def distance_type(metric):
    if metric == "angular":
        return svs.DistanceType.Cosine
    if metric == "euclidean":
        return svs.DistanceType.L2
    raise ValueError("zeelin-svs supports angular and euclidean distances")


class ZeelinSVS(BaseANN):
    def __init__(self, metric, index_param):
        self.metric = distance_type(metric)
        self.index_param = index_param
        self.name = "zeelin-svs"

    def fit(self, X):
        start = time.time()
        X = np.asarray(X, dtype=np.float32)
        params = svs.VamanaBuildParameters(
            alpha=float(self.index_param["alpha"]),
            graph_max_degree=int(self.index_param["graph_max_degree"]),
            window_size=int(self.index_param["window_size"]),
            max_candidate_pool_size=int(self.index_param.get("max_candidate_pool_size", 0)) or 2
            * int(self.index_param["window_size"]),
            prune_to=int(self.index_param.get("prune_to", 0)) or int(self.index_param["graph_max_degree"]),
            use_full_search_history=bool(self.index_param.get("use_full_search_history", True)),
        )
        ids = np.arange(X.shape[0], dtype=np.uint64)
        self.index = svs.DynamicVamana.build(params, X, ids, self.metric, 1)
        self.index.num_threads = 1
        self.build_time = time.time() - start

    def set_query_arguments(self, search_window_size):
        self.search_window_size = int(search_window_size)
        self.index.search_window_size = self.search_window_size
        self.name = "zeelin-svs (%s, search_window_size=%s)" % (
            self.index_param,
            self.search_window_size,
        )

    def query(self, v, n):
        ids, _ = self.index.search(np.asarray(v, dtype=np.float32), n)
        return ids[0]

    def batch_query(self, X, n):
        self.res = self.index.search(np.asarray(X, dtype=np.float32), n)[0]

    def get_batch_results(self):
        return self.res
