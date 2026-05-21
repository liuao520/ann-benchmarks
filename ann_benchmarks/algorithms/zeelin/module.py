import time

import numpy as np
import scann
from sklearn import preprocessing

from ..base.module import BaseANN


class Zeelin(BaseANN):
    def __init__(self, metric, index_param):
        if metric != "angular":
            raise ValueError("zeelin is tuned for angular distance")
        # Store the index-time parameters selected by config.yml.
        # 保存 config.yml 传入的建索引参数。
        self.index_param = index_param
        self.name = "zeelin"

    def fit(self, X):
        start = time.time()
        # Angular distance is evaluated through dot product after L2 normalization.
        # 对 angular 数据先做 L2 归一化，然后用 dot_product 搜索。
        X = np.asarray(X, dtype=np.float32)
        X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
        X = preprocessing.normalize(X, norm="l2", axis=1)

        # ScaNN tree + asymmetric hashing + exact reordering gives the best
        # single-threaded recall/QPS tradeoff found on glove-100-angular.
        # 组合使用分区树、各向异性量化和精排，提升单线程 recall-QPS 曲线。
        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, "dot_product")
            .tree(
                self.index_param["num_leaves"],
                1,
                training_sample_size=len(X),
                spherical=True,
                quantize_centroids=True,
                soar_lambda=self.index_param.get("soar_lambda"),
                overretrieve_factor=self.index_param.get("overretrieve_factor"),
            )
            .score_ah(
                self.index_param["dims_per_block"],
                anisotropic_quantization_threshold=self.index_param["anisotropic_quantization_threshold"],
            )
            .reorder(1)
            .build()
        )
        self.build_time = time.time() - start

    def set_query_arguments(self, leaves_to_search, reorder_k=None):
        # The benchmark passes each query-time pair from config.yml here.
        # leaves_to_search 控制粗筛范围，reorder_k 控制最终精排候选数。
        if reorder_k is None:
            leaves_to_search, reorder_k = leaves_to_search
        self.leaves_to_search = leaves_to_search
        self.reorder_k = reorder_k
        self.name = "zeelin (%s, leaves_to_search=%s, reorder_k=%s)" % (
            self.index_param,
            leaves_to_search,
            reorder_k,
        )

    def query(self, v, n):
        # Normalize each query with the same transform used for the index.
        # 查询向量也要做同样的 L2 归一化，保证 angular -> dot_product 等价。
        v = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm != 0:
            v = v / norm
        return self.searcher.search(v, n, self.reorder_k, self.leaves_to_search)[0]

    def batch_query(self, X, n):
        # Keep batch mode available for ann-benchmarks, although the official
        # single-threaded comparison uses normal query() timing.
        # 保留批量查询接口；官方单线程对比仍以 query() 为准。
        X = np.asarray(X, dtype=np.float32)
        X = preprocessing.normalize(X, norm="l2", axis=1)
        self.res = self.searcher.search_batched(
            X,
            final_num_neighbors=n,
            pre_reorder_num_neighbors=self.reorder_k,
            leaves_to_search=self.leaves_to_search,
        )[0]

    def get_batch_results(self):
        return self.res
