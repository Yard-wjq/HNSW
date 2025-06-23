import math
import os
import numpy as np
import random
import heapq
import pickle
import time
from collections import defaultdict

import torchvision.transforms as transforms

data_dir = "../archive/3/images/"

class HNSW:
    def __init__(self, max_elements, M=16, ef_construction=128, m_L=1.0):
        """
        初始化HNSW图

        参数:
        - max_elements: 最大元素数量
        - M: 每个节点的最大连接数
        - ef_construction: 构建时的动态候选列表大小，越大找寻的最近值越精准
        - m_L: 控制层数的参数
        """
        self.max_elements = max_elements
        self.M = M
        self.M_max = M * 2
        self.ef_construction = ef_construction
        self.m_L = m_L

        """
        data = [ [0.1,0.2], [0.1,0.3] ... ] 
        """
        self.data = []

        """ 
        图结构
        graph = [
            layer0: { 
            node1: [node2, node3] 
            node2: [node4, node5] 
            node3: [node6, node7] 
            }
            layer1: {
            node1: [node8, node9] 
            node2: [node10, node11] 
            node3: [node12, node13] 
            }
        ]
        """
        self.graph = []
        # 入口点
        self.entry_point = None
        # 当前最大层
        self.ep_layer = -1
        self.distance = self.cosine_distance
        self.node_dict = {}

    def get_random_layer(self):
        # 指数分布随机层数分配
        return math.floor(-math.log(random.random()) * self.m_L)

    @staticmethod
    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def l2_distance(a, b):
        return np.sum((a - b) ** 2)

    def search_layer(self, q, ep, ef, lc):
        """
        在特定层搜索最近邻
        :param q: 查询向量
        :param ep: 入口点
        :param ef: 需要返回的候选数量
        :param lc: 层数
        """
        # 初始化数据结构
        visited = set(ep)  # 已访问元素集合
        candidates = []  # 候选堆 (最小堆，按距离q的距离排序)
        result = []  # 结果堆 (最大堆，保存最近的ef个元素)

        # 将入口点加入候选集和结果集
        for e in ep:
            dist = self.distance(self.data[e], q)
            heapq.heappush(candidates, (dist, e))
            heapq.heappush(result, (-dist, e))  # 使用负距离模拟最大堆
        # 主循环
        while candidates:

            # 取出距离q最近的候选元素
            dist_c, c = heapq.heappop(candidates)

            # 获取当前结果集中距离q最远的元素
            if result:
                max_dist_neg, f = result[0]
                max_dist = -max_dist_neg
                # 由于是大于那么至少会访问完全部的result集合
                # 此时说明result的neighbors无法距离q向量更近了
                if dist_c > max_dist:
                    break  # 终止条件

            # 处理当前元素的邻居
            for neighbor in self.graph[lc][c]:
                if neighbor not in visited:
                    visited.add(neighbor)

                    dist = self.distance(q, self.data[neighbor])

                    # 获取当前结果集中最远距离
                    current_max_dist = -result[0][0] if result else float('inf')

                    # 如果结果集未满或新元素更近
                    if len(result) < ef or dist < current_max_dist:

                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(result, (-dist, neighbor))

                        # 保持结果集大小不超过ef
                        if len(result) > ef:
                            heapq.heappop(result)

        # 返回结果 (按距离从小到大排序)
        result = sorted([(-dist, node_idx) for dist, node_idx in result], key=lambda x: x[0])
        return [e for _, e in result]

    def select_neighbors(self, q, candidates, M):
        """
        从候选列表中选择最近的M个邻居
        """
        if len(candidates) <= M:
            return candidates

        # 按距离排序并选择前M个
        sorted_candidates = sorted(candidates, key=lambda x: self.distance(q, self.data[x]))
        return sorted_candidates[:M]
    
    def select_neighbors_heuristic(self, q, candidates, M, lc, self_idx = None, extendCandidates = False , keepPrunedConnections = False):
        R = []
        W = candidates
        W_d = []
        # 可能通过q发现了新大陆
        if extendCandidates:
            # 不能把自己添进去 neighbour != self_idx
            # 这里是论文原文的实现方法：
            for node_idx in candidates:
                for neighbour in self.graph[lc][node_idx]:
                    if neighbour not in W and neighbour != self_idx:
                        W.append(neighbour)
            # 但是数量级可达 ef_construction * M 因此可以做一下简化，只将Q的邻居纳入候选队列 ef_construction + M 尚可承受
            # q_idx = self.graph[lc][self_idx][-1]
            # for neighbour in self.graph[lc][q_idx]:
            #         if neighbour not in W and neighbour != self_idx:
            #             W.append(neighbour)
        heap = []
        for e in W:
            dist = self.distance(q, self.data[e])  
            heapq.heappush(heap, (dist, e))

        while heap and len(R) < M:
            dist_e, e = heapq.heappop(heap)
            # 检查 e 是否比 R 中所有元素更近（若 R 为空则直接加入）
            # 到 q 的距离小于 到 R 中任意一点的 任意距离，可以增加R中节点的多样性
            if not R or (all(dist_e < self.distance(self.data[e], self.data[r])) for r in R):
                R.append(e)
            else:
                W_d.append(e)

        # 步骤15-17：保留部分被丢弃的候选
        if keepPrunedConnections and len(R) < M:
            # 按距离排序被丢弃的候选
            W_d_sorted = sorted(W_d, key=lambda x: self.distance(q, self.data[x]))
            for e in W_d_sorted:
                if len(R) >= M:
                    break
                R.append(e)

        return R

    def add_point(self, q, path):
        """
        向图中添加新元素
        :param q: 新向量
        """
        if len(self.data) >= self.max_elements:
            return
        # 在数据中加入该节点，并获取编号
        node_idx = len(self.data)
        self.data.append(q)
        self.node_dict[node_idx] = path
        # print(node_idx)
        # 确定新点的层数
        level = self.get_random_layer()
        # print(level)
        # 如level大于最高图层，则扩展图层
        while len(self.graph) <= level:
            self.graph.append(defaultdict(list))
            # 触发一下，方便看log
            self.graph[len(self.graph) - 1][node_idx]

        # 从顶层开始搜索入口点,  ep_layer ---> level + 1 , 只寻找每层最近的节点, 因此此时 ef = 1
        ep = [self.entry_point] if self.entry_point is not None else []
        for lc in range(self.ep_layer, level, -1):
            ep = self.search_layer(q=q, ep=ep, ef=1, lc=lc)

        # 逐层插入    min(level, ep_layer) ---> 0 开始寻找M个扩展节点
        for lc in range(min(level, self.ep_layer), -1, -1):     
            M = self.M if lc != 0 else self.M_max
            candidates = self.search_layer(q=q, ep=ep, ef=self.ef_construction, lc=lc)
            # 直接选择最近节点
            # neighbors = self.select_neighbors(q, candidates, M)
            # 启发式选择
            neighbors = self.select_neighbors_heuristic(q,candidates,M,lc)
            # 添加双向连接
            for neighbor in neighbors:
                self.graph[lc][node_idx].append(neighbor)
                self.graph[lc][neighbor].append(node_idx)

                # 保持每个节点的连接数不超过M
                if len(self.graph[lc][neighbor]) > M:
                    # 直接选择最近节点
                    # self.graph[lc][neighbor] = self.select_neighbors(
                        # self.data[neighbor], self.graph[lc][neighbor], M)
                    # 启发式选择
                    self.graph[lc][neighbor] = self.select_neighbors_heuristic(self.data[neighbor], self.graph[lc][neighbor], M, lc)

            ep = candidates

        # 如果新元素在最高层，更新入口点
        if level > self.ep_layer:
            self.entry_point = node_idx
            self.ep_layer = level

        # print(self.graph)

    def knn_query(self, q, k=8, ef_search=10):
        """
        k近邻查询
        :param q: 查询向量
        :param k: 返回的最近邻数量
        """

        if not self.data:
            return [], []

        # 从顶层开始搜索入口点
        ep = [self.entry_point]

        # 从顶层到底层逐层搜索
        for layer in range(self.ep_layer - 1, 0, -1):
            result = self.search_layer(q, ep, ef=1, lc=layer)
            ep = [result[0]]  # 更新入口点为当前层最近邻

        # 在最底层进行搜索
        candidates = self.search_layer(q, ep, ef_search, 0)

        # 选择最近的k个
        sorted_candidates = sorted(candidates, key=lambda x: self.distance(q, self.data[x]))
        top_k = sorted_candidates[:k]

        distances = [self.distance(q, self.data[node]) for node in top_k]
        return top_k, distances

    def save_index(self, file_path):
        """
        保存HNSW索引到文件
        :param file_path: 保存路径
        """
        # 创建保存目录
        os.path.dirname(file_path)

        # 准备需要保存的数据
        index_data = {
            'max_elements': self.max_elements,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'm_L': self.m_L,
            'data': self.data,
            'graph': [dict(layer) for layer in self.graph],
            'ep_layer': self.ep_layer,
            'entry_point': self.entry_point,
            'node_dict': self.node_dict,
        }

        # 使用pickle保存二进制数据
        with open(file_path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_index(cls, file_path):
        """
        从文件加载HNSW索引
        :param file_path: 文件路径
        :return: 加载的HNSW实例
        """
        with open(file_path, 'rb') as f:
            index_data = pickle.load(f)

        # 创建新实例
        HNSW = cls(
            max_elements=index_data['max_elements'],
            M=index_data['M'],
            ef_construction=index_data['ef_construction'],
            m_L=index_data['m_L'],
        )

        # 恢复状态
        HNSW.data = index_data['data']
        HNSW.M_max = index_data['M'] * 2
        HNSW.graph = [defaultdict(list, layer) for layer in index_data['graph']]
        HNSW.ep_layer = index_data['ep_layer']
        HNSW.entry_point = index_data['entry_point']
        HNSW.distance = HNSW.cosine_distance
        HNSW.node_dict = index_data['node_dict']
        return HNSW

    def brute_force_search(self, q, k=8):
        """
        暴力搜索查询向量q的topk最近邻
        :param q: 查询向量
        :param k: 返回的最近邻数量
        :return: (topk索引列表, 对应的距离列表)
        """
        if not self.data:
            return [], []
        
        # 计算q与所有数据点的距离
        distances = [(i, self.distance(q, point)) for i, point in enumerate(self.data)]
        
        # 按距离排序并取前k个
        distances.sort(key=lambda x: x[1])
        top_k = [idx for idx, dist in distances[:k]]
        top_distances = [dist for idx, dist in distances[:k]]
        
        return top_k, top_distances

    def calculate_recall(self, k=8, ef_search=10, sample_size=None):
        # 固定随机数，使结果可靠
        random.seed(2025611)
        """
        计算 HNSW 搜索结果的召回率（Recall）
        :param k: 查询的最近邻数量
        :param ef_search: HNSW 搜索时的动态候选列表大小
        :param sample_size: 随机采样多少查询点计算召回率（None 表示全部计算）
        :return: 平均召回率
        """
        if not self.data:
            return 0.0

        # 随机采样部分查询点（如果 sample_size 指定）
        query_indices = range(len(self.data))
        if sample_size is not None and sample_size < len(self.data):
            query_indices = random.sample(query_indices, sample_size)
        # query_indices = [i for i in range(sample_size)]
        total_recall = 0.0
        num_queries = 0

        brute_force_time = 0
        HNSW_time = 0
        for i in query_indices:
            q = self.data[i]

            # 1. 暴力搜索得到 Ground Truth
            start_time = time.time()
            true_topk, _ = self.brute_force_search(q, k=k)
            # print(true_topk)
            true_set = set(true_topk)
            end_time = time.time()  # 记录结束时间
            brute_force_time += end_time - start_time  # 计算耗时

            # 2. HNSW 搜索得到近似结果
            start_time = time.time()
            hnsw_topk, _ = self.knn_query(q, k=k, ef_search=ef_search)
            # print(hnsw_topk)
            hnsw_set = set(hnsw_topk)
            end_time = time.time()  # 记录结束时间
            HNSW_time += end_time - start_time  # 计算耗时

            # 3. 计算召回率 = 交集数量 / k
            intersection = true_set & hnsw_set
            recall = len(intersection) / k
            total_recall += recall
            num_queries += 1

            if num_queries % 100 == 0:
                print(num_queries) 

        # 计算平均召回率
        avg_recall = total_recall / sample_size
        print(f"采样{sample_size}个点的暴力搜索用时: {brute_force_time:.6f}秒")
        print(f"采样{sample_size}个点的HNSW搜索用时: {HNSW_time:.6f}秒")
        print(f"采样{sample_size}个点的平均召回率: {avg_recall:.4f}")
        return avg_recall
        


def construct(HNSW_path, image_path, elements, M=16, ef_construction=128, m_L=1.0):
    """
    构建HNSW索引，并保存到path路径
    :param k: 查询的最近邻数量
    :param ef_search: HNSW 搜索时的动态候选列表大小
    :param sample_size: 随机采样多少查询点计算召回率（None 表示全部计算）
    :return: 平均召回率
    """
    hnsw = HNSW(max_elements=elements,M=M, ef_construction=ef_construction, m_L=m_L)
    from ImageVectorizer import ImageVectorizer
    import time
    dir = image_path
    ImageVectorizer = ImageVectorizer(path = dir)
    image_paths  = os.listdir(dir)
    start_time = time.time()
    vectors = ImageVectorizer.images_to_vector(elements)
    i = 0
    for vector in vectors:
        hnsw.add_point(q=vector, path=f"{image_paths[i]}")
        i += 1
        if i % 100 == 0:
            print(i)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"耗时: {elapsed_time:.6f} 秒")
    v1 = ImageVectorizer.image_to_vector(f"{data_dir}{image_paths[0]}")
    top_k, distances = hnsw.knn_query(v1, k=16)
    print("top_k:", top_k)
    print("distances:", distances)
    images = [hnsw.node_dict[node_idx] for node_idx in top_k]
    print(images)
    hnsw.save_index(HNSW_path)
    return hnsw



if __name__ == '__main__':
    elements=60000
    M = 20
    ef_construction = 200
    hnsw = construct(f"../output/HNSW_{elements}_resnet34_heuristric_{M}_{ef_construction}",
                     data_dir, elements=elements, M=M, ef_construction=ef_construction, m_L=1.0)
    # hnsw = HNSW.load_index(f"../output/HNSW_{elements}_resnet34_simple.index")
    sample_size = 1000
    recall_sampled = hnsw.calculate_recall(k=16, ef_search=100, sample_size=sample_size)

    # v2 = ImageVectorizer.image_to_vector("../archive/data/2.png")
    # v3 = ImageVectorizer.image_to_vector("../archive/data/3.png")
    # print(v1.shape)
    # hnsw.add_point(v1)
    # hnsw.add_point(v2)
    # hnsw.add_point(v3)
