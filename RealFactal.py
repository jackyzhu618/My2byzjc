import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib import font_manager

# ===================================================================
# 算法一：为“无茎蕨叶”生成点云
# ===================================================================
def generate_barnsley_fern(num_points=50000):
    """
    使用迭代函数系统(IFS) / 混沌游戏算法生成无茎蕨叶的点云。
    返回: 一个Numpy数组，形状为 (N, 2)。
    """
    # 无茎蕨叶的变换规则
    rules = [
        (0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.90),
        (0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.05),
        (-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.05)
    ]

    total_prob = sum(r[6] for r in rules)
    probs = np.array([r[6] / total_prob for r in rules])
    cum_probs = np.cumsum(probs)

    points = np.zeros((num_points, 2), dtype=np.float32)
    p = np.array([0.0, 0.0], dtype=np.float32)

    # 为了让图形稳定，实际生成点数略多，并舍弃前20个点
    points_to_generate = num_points + 20
    generated_count = 0
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    for i in range(points_to_generate):
        r = np.random.rand()
        idx = np.searchsorted(cum_probs, r)
        a, b, c, d, e, f, _ = rules[idx]
        p = np.array([a * p[0] + b * p[1] + e, c * p[0] + d * p[1] + f], dtype=np.float32)

        if i >= 20:
            points[generated_count] = p
            generated_count += 1

    return points


# ===================================================================
# 算法二：为“谢尔宾斯基三角形”生成多边形集合
# ===================================================================
def generate_sierpinski_points(num_points=20000,depth=6):
    """
    使用递归分割算法生成谢尔宾斯基三角形的点云。

    参数:
        depth (int): 递归的深度。深度越大，点的数量越多，细节越丰富。

    返回:
        numpy.ndarray: 一个形状为 (N, 2) 的 NumPy 数组，
                       代表点云中所有唯一顶点的 (x, y) 坐标。
    """
    # 使用集合 (set) 来存储点，可以自动处理重复的点
    final_points = set()

    def _recursive_sierpinski(points, current_depth):
        # 当达到最深层级时，将这三个顶点加入到集合中
        if current_depth == 0:
            final_points.update(points)
            return
        else:
            p0, p1, p2 = points
            # 计算三条边的中点
            mid01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            mid12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid20 = ((p2[0] + p0[0]) / 2, (p2[1] + p0[1]) / 2)

            # 对三个角落的子三角形进行递归
            _recursive_sierpinski([p0, mid01, mid20], current_depth - 1)
            _recursive_sierpinski([p1, mid01, mid12], current_depth - 1)
            _recursive_sierpinski([p2, mid20, mid12], current_depth - 1)

    # 定义初始大三角形的顶点（确保是浮点数）
    initial_vertices = [(0.0, 0.0), (0.5, np.sqrt(3) / 2), (1.0, 0.0)]

    # 启动递归
    _recursive_sierpinski(initial_vertices, depth)

    # 将点的集合转换为 NumPy 数组并返回
    return np.array(list(final_points))


# ===================================================================
# 主绘图程序 (处理两种不同的数据类型)
# ===================================================================

# 1. 设置绘图环境

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 2. 绘制无茎蕨叶 (处理点云数据)
fern_points = generate_barnsley_fern()
ax1.scatter(fern_points[:, 0], fern_points[:, 1], s=0.2, c='green', alpha=0.7)
ax1.set_title('无茎蕨叶 (点云生成法)', fontsize=14)
ax1.set_aspect('equal')
ax1.axis('off')

# 3. 绘制谢尔宾斯基三角形 (处理多边形数据)
sierpinski_polygons = generate_sierpinski_points()
ax2.scatter(sierpinski_polygons[:, 0], sierpinski_polygons[:, 1], s=0.2, c='green', alpha=0.7)
ax2.set_title('谢尔宾斯基三角形 (递归分割法)', fontsize=14)
ax2.set_aspect('equal')
# Manually set limits for the polygon-based plot
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-0.1, 1.0)
ax2.axis('off')

# 4. 整体布局和显示
fig.suptitle('分形算法对比', fontsize=18, y=0.95)
plt.tight_layout(pad=3.0)
plt.show()

