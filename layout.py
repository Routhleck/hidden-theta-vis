import networkx as nx
import numpy as np
from bokeh_sampledata.gapminder import regions

from data import HiddenGroup, Theta, Hidden

import logging
from tqdm import tqdm

# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def _process_params(G, center, dim):
    """Ensure graph and center coordinates are properly set."""
    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        raise ValueError("Length of center coordinates must match dimension of layout")

    return G, center


def generate_hex_grid(width, height, num):
    """
    生成一个蜂窝网格，并返回指定数量的正六边形的中心点坐标。

    :param width: 正方形区域的宽度
    :param height: 正方形区域的高度
    :param num: 需要的正六边形中心点数量
    :return: 每个正六边形的中心点坐标列表
    """
    # 计算合适的 hex_size 以确保生成的正六边形数量接近 num
    hex_area = (width * height) / num
    hex_size = np.sqrt(2 * hex_area / (3 * np.sqrt(3)))
    hex_radius = hex_size * np.sqrt(3) / 2  # 正六边形的内切圆半径

    # 计算网格的行数和列数
    num_rows = int(height / (3 * hex_radius))
    num_cols = int(width / (2 * hex_size))

    # 计算总中心点数量
    total_centers = num_rows * num_cols

    # 如果总中心点数量与 num 的误差大于 2，调整 hex_size 并重新计算
    max_attempts = 30
    attempts = 0
    while abs(total_centers - num) > (num * 0.2) or total_centers < num:
        if total_centers < num:
            hex_size *= 0.98
        else:
            hex_size *= 1.02
        hex_radius = hex_size * np.sqrt(3) / 2
        num_rows = int(height / (3 * hex_radius))
        num_cols = int(width / (2 * hex_size))
        total_centers = num_rows * num_cols
        attempts += 1
        if attempts >= max_attempts and total_centers > num:
            break

    # 生成正六边形网格的中心点
    hex_centers = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * 2 * hex_size + (row % 2) * hex_size
            y = row * 3 * hex_radius
            hex_centers.append((x, y))

    # 如果生成的中心点数量多于需要的数量，随机选择 num 个中心点
    if len(hex_centers) > num:
        hex_centers = np.array(hex_centers)
        hex_centers = hex_centers[np.random.choice(len(hex_centers), num, replace=False)]

    return hex_centers

def generate_circular_grid(num):
    """
    生成一个同心圆网格，并返回指定数量的中心点坐标。

    :param num: 需要的中心点数量
    :return: 每个中心点的坐标列表
    """
    # 计算每层的数量
    layers = []
    current_num = 0
    layer_index = 0

    # 最里层的数量最大是6，如果num小于6，则最里层的数量就是num
    if num <= 6:
        layers.append(num)
        current_num = num
    else:
        layers.append(6)
        current_num = 6
        layer_index = 1

    # 计算剩余层的数量
    while current_num < num:
        # 每层的数量增加6
        layer_size = 6 + layer_index * 6
        layers.append(layer_size)
        current_num += layer_size
        layer_index += 1

    # 如果最后一层的数量超过了num，调整最后一层的数量
    if current_num > num:
        layers[-1] = num - sum(layers[:-1])

    # 生成同心圆网格的中心点
    centers = []
    radius = 1
    for layer_size in layers:
        angle_step = 2 * np.pi / layer_size
        for i in range(layer_size):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            centers.append((x, y))
        radius += 1

    return centers

def hidden_theta_layout(G: nx.Graph,
                        hidden_groups: list[HiddenGroup],
                        thetas: list[Theta],
                        H2H=0.1,
                        T2H=0.1,
                        T2T=0.1,
                        n_iterations=50,
                        dim=2,
                        center=None,
                        width=100.,
                        height=100.,
                        alpha=0.05,
                        n_neighbors=10,
):
    """
    Custom layout for a graph with Hidden and Theta nodes, based on user-defined rules.

    Parameters:
        G (nx.Graph): Input graph.
        hidden_groups (list): List of HiddenGroup instances.
        thetas (list): List of Theta instances.
        H2H (float): Distance for Hidden-to-Hidden nodes.
        T2H (float): Distance for Theta-to-Hidden nodes.
        T2T (float): Distance for Theta-to-Theta nodes.
        n_iterations (int): Number of iterations to adjust node positions.
        dim (int): Number of dimensions (2 for 2D layouts).
        center (array-like): Coordinates for the layout's center.
        alpha (float): Weight for adjusting node positions.

    Returns:
        dict: A dictionary of positions keyed by node.
    """
    # 预处理图和中心点
    G, center = _process_params(G, center, dim)
    logging.info("Graph and center processed.")

    # 1. 计算每个区域的中心点
    num_groups = len(hidden_groups)
    # region_centers = generate_hex_grid(width, height, num_groups)
    region_centers = generate_circular_grid(num_groups)
    logging.info("Hex grid generated with %d centers.", num_groups)

    remaining_center = np.zeros(dim)

    # 2. 将第一个 target 为 None 的 Hidden 放置在其区域中心点上
    pos = {}
    for group, center_point in zip(hidden_groups, region_centers):
        for hidden in group.hiddens:
            if hidden.target is None:
                pos[f"Hidden_{hidden.id}"] = center_point
                break
    logging.info("Initial Hidden nodes positioned.")

    # 3. 将没有 target 的 Hidden 随机放在距离其 Group 中心点为 H2H 的位置
    for group, center_point in zip(hidden_groups, region_centers):
        for hidden in group.hiddens:
            if hidden.target is None and f"Hidden_{hidden.id}" not in pos:
                angle = np.arctan2(center_point[1], center_point[0]) + np.random.rand() * np.pi / 2
                pos[f"Hidden_{hidden.id}"] = center_point + np.array([H2H * np.cos(angle), H2H * np.sin(angle)])
    logging.info("Randomly positioned Hidden nodes without targets.")

    # 4. 将有 target 的 Hidden 放置在 target 距离为 H2H 的位置
    for group in hidden_groups:
        group_hiddens = [hidden for hidden in group.hiddens if isinstance(hidden.target, Hidden)]
        while group_hiddens:
            for hidden in group_hiddens:
                if hidden.target is not None:
                    target_id = f"Hidden_{hidden.target.id}"
                    if target_id in pos:
                        angle = np.arctan2(pos[target_id][1], pos[target_id][0]) + np.random.rand() * np.pi / 2
                        pos[f"Hidden_{hidden.id}"] = pos[target_id] + np.array([H2H * np.cos(angle), H2H * np.sin(angle)])
                        group_hiddens.remove(hidden)
    logging.info("Positioned Hidden nodes with targets.")

    # 5. 处理 Theta 的布局
    thetas_hidden = [theta for theta in thetas if isinstance(theta.target, Hidden)]
    thetas_theta = [theta for theta in thetas if isinstance(theta.target, Theta)]
    thetas_empty = [theta for theta in thetas if theta.target is None]

    for theta in thetas_hidden:
        target_id = f"Hidden_{theta.target.id}"
        if target_id in pos:
            angle = np.arctan2(pos[target_id][1], pos[target_id][0]) + np.random.rand() * np.pi / 2
            pos[f"Theta_{theta.id}"] = pos[target_id] + np.array([T2H * np.cos(angle), T2H * np.sin(angle)])
    logging.info("Positioned Theta nodes with Hidden targets.")

    for theta in thetas_empty:
        pos[f"Theta_{theta.id}"] = remaining_center + np.random.rand(dim) * 0.05
    logging.info("Positioned Theta nodes without targets.")

    while thetas_theta:
        for theta in thetas_theta:
            target_id = f"Theta_{theta.target.id}"
            if target_id in pos:
                angle = np.arctan2(pos[target_id][1], pos[target_id][0]) + np.random.rand() * np.pi / 2
                pos[f"Theta_{theta.id}"] = pos[target_id] + np.array([T2T * np.cos(angle), T2T * np.sin(angle)])
                thetas_theta.remove(theta)
    logging.info("Positioned Theta nodes with Theta targets.")

    # 6. 迭代调整所有节点位置
    for iteration in tqdm(range(n_iterations)):
        # 计算每个节点的最近邻居节点
        for node, coord in pos.items():
            # 计算所有节点之间的距离
            distances = np.linalg.norm(np.array(list(pos.values())) - coord, axis=1)
            # 排除自身
            distances[list(pos.keys()).index(node)] = np.inf

            # 找到最近的 n_neighbors 个邻居节点
            nearest_neighbor_indices = np.argsort(distances)[:n_neighbors]
            nearest_neighbors = np.array(list(pos.values()))[nearest_neighbor_indices]
            nearest_distances = distances[nearest_neighbor_indices]

            # 计算能量
            energies = 0.01 / nearest_distances * alpha

            # 如果距离超过 H2H, T2H, T2T 的最小值，能量为0
            energies[nearest_distances > min(H2H, T2H, T2T)] = 0

            # 移动节点以最小化能量
            if np.sum(energies) > 0:
                directions = coord - nearest_neighbors
                directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
                pos[node] += np.sum(directions * energies[:, np.newaxis], axis=0)

    # 将位置归一化到中心点
    pos = {node: center + coord for node, coord in pos.items()}

    # 确保返回的 pos 字典按 G 的节点顺序排列
    pos = dict(zip(G.nodes(), [pos.get(node, np.zeros(dim)) for node in G.nodes()]))

    return pos
