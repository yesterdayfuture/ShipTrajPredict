#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRACLUS 算法完整示例（单文件可运行）
--------------------------------
参考文献：
Lee J G, Han J, Whang K Y. Trajectory clustering: a partition-and-group framework.
Proceedings of the 2007 ACM SIGMOD international conference on Management of data. 2007.

整体流程：
1. 轨迹分段（Partitioning）
2. 线段聚类（Grouping）
3. 可选：代表性轨迹生成（Classifying，本篇仅给出思路）
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from ast import literal_eval


# ========================= 1. 数据结构 =========================
class Point:
    """二维轨迹点"""
    __slots__ = ("x", "y", "t") #显式声明类的实例属性,限制实例只能拥有x、y、t三个属性，避免动态属性字典__dict__的创建

    def __init__(self, x: float, y: float, t: float = None):
        self.x = float(x)
        self.y = float(y)
        self.t = t  # 时间戳，可选，本实现里仅用于可视化

    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"


class Trajectory:
    """一条轨迹 = 点的序列"""
    def __init__(self, pts: List[Point]):
        self.pts = pts
        self.data = []

    def __repr__(self):
        for point in self.pts:
            self.data.append((point.x, point.y))
        return str(self.data)



class LineSegment:
    """轨迹分段后的线段"""
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __repr__(self):
        return f"LineSegment({self.p1} -> {self.p2})"


# ========================= 2. 轨迹分段 =========================
def min_dist_angle_criterion(traj: Trajectory,
                             dist_thresh: float = 0.5,
                             angle_thresh: float = np.radians(45)) -> List[LineSegment]:
    """
    基于“最小描述长度”思想，用“垂直距离 + 角度距离”进行轨迹分段。
    每当累计误差超过阈值时产生一个分段点。

    参数
    ----
    dist_thresh : 垂直距离阈值（米）
    angle_thresh : 角度距离阈值（弧度）

    返回
    ----
    分段后的线段列表
    """
    if len(traj.pts) < 2:
        return []

    seg_start = 0
    segments = []

    for i in range(2, len(traj.pts)):
        # 当前轨迹段子区间 [seg_start, i]，用直线首尾近似
        p_s = traj.pts[seg_start]
        p_e = traj.pts[i]

        # 计算区间每一点到直线的垂直距离 & 角度误差
        max_dist = 0.0
        max_angle = 0.0

        for j in range(seg_start + 1, i):
            p = traj.pts[j]

            # 2.1 垂直距离
            # 向量叉积法求点到直线距离
            vec_se = np.array([p_e.x - p_s.x, p_e.y - p_s.y])
            vec_sp = np.array([p.x - p_s.x, p.y - p_s.y])
            cross = abs(np.cross(vec_se, vec_sp))
            perp_dist = cross / (np.linalg.norm(vec_se) + 1e-8)

            # 2.2 角度误差
            # 使用向量夹角
            vec_prev = np.array([p.x - traj.pts[j - 1].x,
                                 p.y - traj.pts[j - 1].y])
            cos_theta = np.clip(np.dot(vec_se, vec_prev) /
                                (np.linalg.norm(vec_se) * np.linalg.norm(vec_prev) + 1e-8),
                                -1.0, 1.0)
            angle = np.arccos(cos_theta)

            max_dist = max(max_dist, perp_dist)
            max_angle = max(max_angle, angle)

        # 若任一误差超标，则把上一个点作为分段点
        if max_dist > dist_thresh or max_angle > angle_thresh:
            segments.append(LineSegment(traj.pts[seg_start], traj.pts[i - 1]))
            seg_start = i - 1

    # 末尾剩余部分
    segments.append(LineSegment(traj.pts[seg_start], traj.pts[-1]))
    return segments


# ========================= 3. 线段距离 =========================
def perpendicular_distance(l1: LineSegment, l2: LineSegment) -> float:
    """垂直距离"""
    def point_to_line(p: Point, line: LineSegment):
        # 线段 line 的向量
        v = np.array([line.p2.x - line.p1.x, line.p2.y - line.p1.y])
        w = np.array([p.x - line.p1.x, p.y - line.p1.y])
        # 投影长度
        t = max(0.0, min(1.0, np.dot(w, v) / (np.dot(v, v) + 1e-8)))
        proj = line.p1.x + t * v[0], line.p1.y + t * v[1]
        return np.linalg.norm([p.x - proj[0], p.y - proj[1]])

    d1 = point_to_line(l1.p1, l2)
    d2 = point_to_line(l1.p2, l2)
    d3 = point_to_line(l2.p1, l1)
    d4 = point_to_line(l2.p2, l1)
    return min(d1, d2, d3, d4)


def parallel_distance(l1: LineSegment, l2: LineSegment) -> float:
    """平行距离：线段首尾到另一线段的投影距离"""
    def proj_len(p: Point, line: LineSegment):
        v = np.array([line.p2.x - line.p1.x, line.p2.y - line.p1.y])
        w = np.array([p.x - line.p1.x, p.y - line.p1.y])
        return np.dot(w, v) / (np.linalg.norm(v) + 1e-8)

    l1_vec = np.array([l1.p2.x - l1.p1.x, l1.p2.y - l1.p1.y])
    l2_vec = np.array([l2.p2.x - l2.p1.x, l2.p2.y - l2.p1.y])

    # 计算投影区间
    l1_proj_s = proj_len(l1.p1, l2)
    l1_proj_e = proj_len(l1.p2, l2)
    l2_proj_s = proj_len(l2.p1, l1)
    l2_proj_e = proj_len(l2.p2, l1)

    # 区间长度差
    l1_len = np.linalg.norm(l1_vec)
    l2_len = np.linalg.norm(l2_vec)

    return min(abs(l1_proj_e - l1_proj_s) * l2_len / (l1_len + 1e-8),
               abs(l2_proj_e - l2_proj_s) * l1_len / (l2_len + 1e-8))


def angle_distance(l1: LineSegment, l2: LineSegment) -> float:
    """角度距离 = 线段夹角 * 长度"""
    v1 = np.array([l1.p2.x - l1.p1.x, l1.p2.y - l1.p1.y])
    v2 = np.array([l2.p2.x - l2.p1.x, l2.p2.y - l2.p1.y])
    cos_theta = np.clip(np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta * (np.linalg.norm(v1) + np.linalg.norm(v2))


def traclus_distance(l1: LineSegment, l2: LineSegment,
                     w_perp: float = 1.0,
                     w_par: float = 1.0,
                     w_ang: float = 1.0) -> float:
    """综合距离，权重可调"""
    return (w_perp * perpendicular_distance(l1, l2) +
            w_par * parallel_distance(l1, l2) +
            w_ang * angle_distance(l1, l2))


# ========================= 4. 线段聚类 =========================
def cluster_segments(segments: List[LineSegment],
                     eps: float = 3.0,
                     min_samples: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 DBSCAN 对线段进行聚类。
    将线段转成 4 维特征向量 [x1, y1, x2, y2] 后聚类。
    你也可以换成自定义距离矩阵的 DBSCAN（metric='precomputed'）。
    """
    if not segments:
        return np.array([]), np.array([])

    X = np.array([[seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y] for seg in segments])

    # 这里简单用欧氏距离，实际可替换为 traclus_distance 的矩阵
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    return X, labels


# ========================= 5. 可视化 =========================
def plot_trajectories(trajs: List[Trajectory], title="原始轨迹"):
    plt.figure(figsize=(6, 6))
    for traj in trajs:
        xs = [p.x for p in traj.pts]
        ys = [p.y for p in traj.pts]
        plt.plot(xs, ys, marker='o', markersize=2)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_segments(segments: List[LineSegment], labels: np.ndarray):
    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap("tab20", max(labels.max() + 1, 2))
    for seg, lbl in zip(segments, labels):
        color = "gray" if lbl == -1 else colors(lbl)
        plt.plot([seg.p1.x, seg.p2.x], [seg.p1.y, seg.p2.y], color=color)
    plt.title("线段聚类结果（-1=噪声）")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# ========== 7. 距离矩阵 + 聚类 ==========
def build_distance_matrix(segments: List[LineSegment],
                          w_perp=1.0, w_par=1.0, w_ang=1.0) -> np.ndarray:
    """
    计算 TRACLUS 距离矩阵（对称，n×n）
    """
    n = len(segments)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = traclus_distance(segments[i], segments[j],
                                 w_perp=w_perp, w_par=w_par, w_ang=w_ang)
            D[i, j] = d
            D[j, i] = d
    return D


def cluster_segments_traclus(segments: List[LineSegment],
                             eps: float = 3.0,
                             min_samples: int = 3,
                             **kwargs) -> np.ndarray:
    """
    使用 TRACLUS 距离 + DBSCAN
    """
    if not segments:
        return np.array([])

    D = build_distance_matrix(segments, **kwargs)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    return clustering.fit_predict(D)


# ========== 8. 代表性轨迹生成 ==========
from sklearn.decomposition import PCA

def build_representative_trajectories(segments: List[LineSegment],
                                      labels: np.ndarray) -> List[Trajectory]:
    """
    为每个簇（不含噪声 -1）生成一条代表性轨迹
    :return: List[Trajectory]，顺序与簇号一致
    """
    clusters = {}
    for seg, lbl in zip(segments, labels):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(seg)

    reps = []
    for cid, segs in sorted(clusters.items()):
        # 8.1 收集所有端点
        pts = []
        for s in segs:
            pts.extend([s.p1, s.p2])
        pts = np.array([[p.x, p.y] for p in pts])

        # 8.2 PCA -> 主方向直线
        pca = PCA(n_components=1).fit(pts)
        centroid = pca.mean_                      # 中心点
        direction = pca.components_[0]            # 单位主方向

        # 8.3 投影到直线上
        t_vals = ((pts - centroid) @ direction)   # 标量投影坐标

        # 8.4 在直线 L 上均匀插值 50 个点
        t_min, t_max = t_vals.min(), t_vals.max()
        if np.isclose(t_min, t_max):   # 只有一个点
            rep_pts = [Point(centroid[0], centroid[1])]
        else:
            ts = np.linspace(t_min, t_max, 50)
            rep_pts = [Point(*(centroid + t * direction)) for t in ts]

        reps.append(Trajectory(rep_pts))

    return reps


def plot_representatives(reps: List[Trajectory]):
    plt.figure(figsize=(6, 6))
    for rep in reps:
        xs = [p.x for p in rep.pts]
        ys = [p.y for p in rep.pts]
        plt.plot(xs, ys, linewidth=3)
    plt.title("代表性轨迹")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig("representatives.png")

# ========================= 6. 主流程 DEMO =========================
if __name__ == "__main__":
    # 6.1 构造几条合成轨迹
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 100)

    # 轨迹1：直线
    traj1 = Trajectory([Point(x, y) for x, y in zip(t, t + rng.normal(0, 0.3, size=t.size))])
    # 轨迹2：正弦
    traj2 = Trajectory([Point(x, y) for x, y in zip(t, 5 + np.sin(t) + rng.normal(0, 0.2, size=t.size))])
    # 轨迹3：带噪声的直线
    traj3 = Trajectory([Point(x, y) for x, y in zip(t, 0.5 * t + 2 + rng.normal(0, 0.4, size=t.size))])

    trajs = [traj1, traj2, traj3]
    # plot_trajectories(trajs)

    # 6.2 分段
    all_segments = []
    for traj in trajs:
        segs = min_dist_angle_criterion(traj, dist_thresh=0.5, angle_thresh=np.radians(70))
        all_segments.extend(segs)

    # print(f'分段后的数据长度是：{len(all_segments)}')

    # 6.3 聚类
    # _, labels = cluster_segments(all_segments, eps=1, min_samples=3)
    labels = cluster_segments_traclus(all_segments,
                                  eps=1.0,           # 根据 TRACLUS 距离重新调参
                                  min_samples=3,
                                  w_perp=1.0, w_par=1.0, w_ang=1.0)
    print(f'聚类后的结果是：{labels}')
    # plot_segments(all_segments, labels)

    print("聚类完成！噪声(-1)以外的每个颜色代表一个簇。")

    # 8. 生成并绘制代表性轨迹
    reps = build_representative_trajectories(all_segments, labels)
    print(reps)
    print(type(reps[0]))
    print(reps[0])
    print(type(reps[0].data))
    # ast_data = literal_eval("[(0.4744033093996196, 0.139464113105485), (0.5306502994346296, 0.208231052418812), (0.5868972894696396, 0.2769979917321388), (0.6431442795046497, 0.3457649310454658), (0.6993912695396595, 0.41453187035879235), (0.7556382595746697, 0.48329880967211947), (0.8118852496096797, 0.5520657489854464), (0.8681322396446896, 0.6208326882987731), (0.9243792296796995, 0.6895996276121), (0.9806262197147095, 0.7583665669254268), (1.0368732097497195, 0.8271335062387537), (1.0931201997847295, 0.8959004455520806)]")
    # print(ast_data)
    # print(type(ast_data))
    
    # plot_representatives(reps)
    print(f"已生成 {len(reps)} 条代表性轨迹。")