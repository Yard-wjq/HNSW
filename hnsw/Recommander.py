import os
import json

from hnsw.ImageVectorizer import ImageVectorizer
from hnsw.HNSW import HNSW

class Recommender:

    def __init__(self, HNSW_path, data_path):
        """
        初始化卡通头像推荐系统
        :param max_elements: 最大元素数量
        """
        self.vectorizer = ImageVectorizer(data_path)

        # 初始化HNSW索引
        self.HNSW = HNSW.load_index(HNSW_path)

    def recommend(self, query_image_path, k=8 , ef_search=128):
        """
        推荐相似卡通头像
        :param query_image_path: 查询图像路径
        :param k: 返回的推荐数量
        :param ef_search: 搜索时的动态候选列表大小
        :return: 相似图像路径和距离列表
        """
        query_vector = self.vectorizer.image_to_vector(query_image_path)
        ids, distances = self.HNSW.knn_query(query_vector, k=k, ef_search=ef_search)
        # print(ids, distances)
        return [(self.HNSW.node_dict[id], 1 - distance) for id, distance in zip(ids, distances)]