import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from torch.cuda.amp import GradScaler, autocast
import os
from matplotlib.animation import FuncAnimation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ==============================================================================
# 1. 增强版损失函数
# ==============================================================================
def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """标准Chamfer距离计算"""
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return torch.tensor(0.0, device=p1.device, requires_grad=True)

    p1 = p1.float()
    p2 = p2.float()
    dist_matrix_squared = torch.cdist(p1, p2).pow(2)

    dist1, _ = dist_matrix_squared.min(dim=1)
    loss1 = dist1.mean()

    dist2, _ = dist_matrix_squared.min(dim=0)
    loss2 = dist2.mean()

    return loss1 + loss2


def multi_scale_chamfer(p1, p2, scales=[1, 0.5, 0.25]):
    """多尺度Chamfer距离计算，增强对分形层次结构的捕捉"""
    loss = 0
    for s in scales:
        down_p1 = p1[::int(1 / s)] if s < 1 else p1
        down_p2 = p2[::int(1 / s)] if s < 1 else p2
        loss += chamfer_distance(down_p1, down_p2) * s
    return loss


def calculate_dcd_2d(p1: torch.Tensor, p2: torch.Tensor, alpha: int = 1000, n_lambda: int = 1) -> torch.Tensor:
    """
    计算两个二维点云之间的密度感知倒角距离 (DCD)。
    该实现基于论文 'Density-aware Chamfer Distance' (arXiv:2111.12702v1)。
    """
    p1 = p1.float()
    p2 = p2.float()

    N, M = p1.shape[0], p2.shape[0]

    dist_matrix = torch.cdist(p1, p2)
    dist_squared = dist_matrix.pow(2)

    p1_to_p2_dists, p1_to_p2_idx = dist_squared.min(dim=1)
    p2_to_p1_dists, p2_to_p1_idx = dist_squared.min(dim=0)

    n_p2 = torch.zeros(M, device=p1.device)
    n_p2.scatter_add_(0, p1_to_p2_idx, torch.ones(N, device=p1.device))

    n_p1 = torch.zeros(N, device=p1.device)
    n_p1.scatter_add_(0, p2_to_p1_idx, torch.ones(M, device=p1.device))

    n_p2_gathered = n_p2.gather(0, p1_to_p2_idx)
    n_p1_gathered = n_p1.gather(0, p2_to_p1_idx)

    epsilon = 1e-8
    weights_p2 = n_p2_gathered.pow(n_lambda) + epsilon
    weights_p1 = n_p1_gathered.pow(n_lambda) + epsilon

    term1 = (1 - (1 / weights_p2) * torch.exp(-alpha * p1_to_p2_dists)).mean()
    term2 = (1 - (1 / weights_p1) * torch.exp(-alpha * p2_to_p1_dists)).mean()

    dcd_loss = (term1 + term2) / 2

    return dcd_loss


def calculate_hausdorff_distance(img1, img2):
    """计算两张图像之间的Hausdorff距离"""
    points1 = torch.nonzero(img1 > 0.5).float()
    points2 = torch.nonzero(img2 > 0.5).float()

    if len(points1) == 0 or len(points2) == 0:
        return float('inf')

    dist_matrix = torch.cdist(points1, points2)
    d1 = torch.max(torch.min(dist_matrix, dim=1)[0])
    d2 = torch.max(torch.min(dist_matrix, dim=0)[0])
    return max(d1, d2)


def fractal_dimension_reg(points, target_dim=1.58496, alpha=0.01):
    """分形维度正则化，约束生成点云的维度特性"""
    if points.shape[0] < 10:
        return torch.tensor(0.0, device=points.device)

    points = points.float()
    min_bbox = points.min(dim=0)[0]
    max_bbox = points.max(dim=0)[0]
    extent = max_bbox - min_bbox
    max_scale = torch.min(extent) / 2

    dim_est = torch.tensor(0.0, device=points.device)
    num_scales = 5
    valid_scales = 0

    for i in range(num_scales):
        scale = max_scale * (0.5 ** i)
        grid = torch.floor((points - min_bbox) / scale)
        unique_cells = torch.unique(grid, dim=0).shape[0]
        if unique_cells > 1:
            dim_est += torch.log(torch.tensor(unique_cells, dtype=torch.float, device=points.device)) / \
                       torch.log(torch.tensor(1.0 / scale, dtype=torch.float, device=points.device))
            valid_scales += 1

    if valid_scales > 0:
        dim_est /= valid_scales
    else:
        dim_est = torch.tensor(0.0, device=points.device)

    target_dim_tensor = torch.tensor(target_dim, device=points.device)
    return alpha * torch.abs(dim_est - target_dim_tensor)


def enforce_contractivity(matrices, epsilon=1e-3):
    """强制变换矩阵满足收缩性约束（谱范数<1）"""
    original_dtype = matrices.dtype
    matrices = matrices.float()

    for i in range(matrices.shape[0]):
        U, S, Vh = torch.linalg.svd(matrices[i], full_matrices=False)
        S = torch.clamp(S, max=1 - epsilon)
        S_diag = torch.diag(S)
        matrices[i] = U @ S_diag @ Vh

    return matrices.to(original_dtype)


# ===================================================================
# 算法一：为“无茎蕨叶”生成点云
# ==============================================
def generate_barnsley_fern(num_points=50000, iterations=10, device='cpu'):
    """
    使用确定性算法和3个非退化变换来生成巴恩斯利蕨。
    """
    print(f"使用确定性算法生成巴恩斯利蕨 (无主茎)，共迭代 {iterations} 次...")

    A2 = torch.tensor([[0.85, 0.04], [-0.04, 0.85]], dtype=torch.float32, device=device)
    b2 = torch.tensor([0.00, 1.60], dtype=torch.float32, device=device)

    A3 = torch.tensor([[0.20, -0.26], [0.23, 0.22]], dtype=torch.float32, device=device)
    b3 = torch.tensor([0.00, 1.60], dtype=torch.float32, device=device)

    A4 = torch.tensor([[-0.15, 0.28], [0.26, 0.24]], dtype=torch.float32, device=device)
    b4 = torch.tensor([0.00, 0.44], dtype=torch.float32, device=device)

    transforms_A = torch.stack([A2, A3, A4])
    transforms_b = torch.stack([b2, b3, b4])

    current_points = torch.zeros(1, 2, device=device)
    print(f"初始点数量: {len(current_points)}")

    for i in range(iterations):
        transformed_clouds = []
        for j in range(len(transforms_A)):
            A = transforms_A[j]
            b = transforms_b[j]
            new_points = torch.matmul(current_points, A.T) + b
            transformed_clouds.append(new_points)

        all_new_points = torch.cat(transformed_clouds, dim=0)
        current_points = torch.unique(all_new_points, dim=0)
        print(f"第 {i + 1}/{iterations} 次迭代后，点数量: {len(current_points)}")

    print(f"生成完成！最终生成的点云包含 {len(current_points)} 个点。")
    return current_points.cpu().numpy()  # 修改：返回NumPy数组


# ===================================================================
# 算法二：为“谢尔宾斯基三角形”生成多边形集合
# ===================================================================
def generate_sierpinski(num_points=20000, depth=6, device='cpu'):
    """
    使用递归分割算法生成谢尔宾斯基三角形的点云。
    """
    final_points = set()

    def _recursive_sierpinski(points, current_depth):
        if current_depth == 0:
            final_points.update(points)
            return
        else:
            p0, p1, p2 = points
            mid01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            mid12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid20 = ((p2[0] + p0[0]) / 2, (p2[1] + p0[1]) / 2)

            _recursive_sierpinski([p0, mid01, mid20], current_depth - 1)
            _recursive_sierpinski([p1, mid01, mid12], current_depth - 1)
            _recursive_sierpinski([p2, mid20, mid12], current_depth - 1)

    initial_vertices = [(0.0, 0.0), (0.5, np.sqrt(3) / 2), (1.0, 0.0)]
    _recursive_sierpinski(initial_vertices, depth)

    return np.array(list(final_points))


# ==============================================================================
# 2. 增强型IFS模型
# ==============================================================================
class AdaptiveIFSNet(nn.Module):
    def __init__(self, max_ifs=10, base_sigma=0.5, init_seed=None):
        super(AdaptiveIFSNet, self).__init__()
        self.max_ifs = max_ifs
        self.ifs_w = nn.Embedding(max_ifs, 6)
        self.ifs_b = nn.Embedding(max_ifs, 2)

        self.gate = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        if init_seed is not None:
            np.random.seed(init_seed)
            torch.manual_seed(init_seed)

        for index in range(max_ifs):
            theta1 = np.random.rand() * 2 * np.pi - np.pi
            theta2 = np.random.rand() * 2 * np.pi - np.pi
            perturbation1 = 0.1 * (np.random.rand() - 0.5)
            perturbation2 = 0.1 * (np.random.rand() - 0.5)
            sigma1 = base_sigma + perturbation1
            sigma2 = base_sigma + perturbation2
            d1, d2 = 0.005, 0.005

            params = torch.tensor([theta1, theta2, sigma1, sigma2, d1, d2], dtype=torch.float32)
            b = (torch.rand(2) - 0.5) * 2

            self.ifs_w.weight.data[index].copy_(params)
            self.ifs_b.weight.data[index].copy_(b)

    def make_rotation_matrix(self, theta):
        """创建2D旋转矩阵"""
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return torch.stack([
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t, cos_t], dim=-1)
        ], dim=-2)

    def make_diagonal_matrix(self, d1, d2):
        """创建2x2对角矩阵"""
        zero = torch.zeros_like(d1)
        return torch.stack([
            torch.stack([d1, zero], dim=-1),
            torch.stack([zero, d2], dim=-1)
        ], dim=-2)

    def make_matrices_from_svdformat(self, params):
        """从SVD格式参数构建仿射变换矩阵"""
        theta1, theta2, sigma1, sigma2, d1, d2 = torch.chunk(params, 6, dim=-1)
        theta1, theta2 = theta1.squeeze(-1), theta2.squeeze(-1)
        sigma1, sigma2 = sigma1.squeeze(-1), sigma2.squeeze(-1)
        d1, d2 = d1.squeeze(-1), d2.squeeze(-1)

        r_mat1 = self.make_rotation_matrix(theta1 * 2 * math.pi)
        r_mat2 = self.make_rotation_matrix(theta2 * 2 * math.pi)

        sig_mat = self.make_diagonal_matrix(torch.sigmoid(sigma1), torch.sigmoid(sigma2))

        d1_eff = d1.sign() - d1.detach() + d1
        d2_eff = d2.sign() - d2.detach() + d2
        d_mat = self.make_diagonal_matrix(d1_eff, d2_eff)

        w = torch.matmul(torch.matmul(torch.matmul(r_mat1, sig_mat), r_mat2), d_mat)
        return w

    def forward(self, p_in, return_gates=False, enforce_contractive=True):
        target_num_points = p_in.shape[0]
        device = p_in.device

        w_params = self.ifs_w.weight
        gates = self.gate(w_params).squeeze(-1)

        valid_indices = torch.where(gates > 0.1)[0]
        if len(valid_indices) == 0:
            valid_indices = torch.tensor([gates.argmax()], device=device)

        valid_w_params = w_params[valid_indices]
        valid_b_params = self.ifs_b.weight[valid_indices]
        w_matrices = self.make_matrices_from_svdformat(valid_w_params)

        if enforce_contractive:
            w_matrices = enforce_contractivity(w_matrices)

        output_clouds = []
        for i in range(len(valid_indices)):
            p_out_i = torch.matmul(p_in, w_matrices[i].T) + valid_b_params[i]
            output_clouds.append(p_out_i)

        dets = torch.abs(torch.det(w_matrices))
        probs = dets / (torch.sum(dets) + 1e-8)
        num_samples_per_cloud = (probs * target_num_points).round().long()

        num_samples_per_cloud = torch.clamp(num_samples_per_cloud, min=1)

        total = torch.sum(num_samples_per_cloud)
        if total != target_num_points:
            num_samples_per_cloud[-1] += target_num_points - total

        sampled_clouds = []
        for i in range(len(valid_indices)):
            cloud_i = output_clouds[i]
            num_samples = num_samples_per_cloud[i]

            if num_samples > 0 and cloud_i.shape[0] > 0:
                rand_indices = torch.randperm(cloud_i.shape[0], device=device)[:num_samples]
                sampled_clouds.append(cloud_i[rand_indices])

        if sampled_clouds:
            p_out_sampled = torch.cat(sampled_clouds, dim=0)
        else:
            p_out_sampled = torch.empty(0, 2, device=device)

        if return_gates:
            return p_out_sampled, gates, valid_indices

        return p_out_sampled


# ==============================================================================
# 3. 辅助函数：可视化操作
# ==============================================================================
def visualize_ifs_transforms(model, target_points, num_points=1000, title_prefix=""):
    """可视化IFS的各个变换效果"""
    device = next(model.parameters()).device
    points_tensor = torch.from_numpy(target_points).float().to(device)

    w_params = model.ifs_w.weight
    b_params = model.ifs_b.weight
    w_matrices = model.make_matrices_from_svdformat(w_params)

    transformed_clouds = []
    for i in range(model.max_ifs):
        p_out_i = torch.matmul(points_tensor[:num_points], w_matrices[i].T) + b_params[i]
        transformed_clouds.append(p_out_i.cpu().detach().numpy())

    n_cols = min(3, model.max_ifs)
    n_rows = (model.max_ifs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in range(model.max_ifs):
        ax = axes[i]
        ax.scatter(transformed_clouds[i][:, 0], transformed_clouds[i][:, 1], s=1, c='blue', alpha=0.7)
        ax.set_title(f'{title_prefix}Transform {i}')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    for i in range(model.max_ifs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def create_training_animation(history_points, interval=100, title_prefix=""):
    """创建训练过程的动画"""
    fig, ax = plt.subplots(figsize=(8, 8))

    def init():
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f'{title_prefix}Training Progress')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        return ax,

    def update(frame):
        ax.clear()
        ax.scatter(history_points[frame][:, 0], history_points[frame][:, 1], s=1, c='blue', alpha=0.7)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f'{title_prefix}Training Epoch: {frame * 100}')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        return ax,

    ani = FuncAnimation(fig, update, frames=len(history_points),
                        init_func=init, blit=False, interval=interval)
    return ani


# ==============================================================================
# 4. 训练函数
# ==============================================================================
def optimize_ifs_count(target_points, sigma_value, min_ifs=2, max_ifs=10, epochs=1000, loss_function='chamfer'):
    """优化确定最佳IFS映射个数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_in_target = torch.from_numpy(target_points).float().to(device)

    best_loss = float('inf')
    best_ifs_count = min_ifs
    results = []

    print(f"\n--- Optimizing Best IFS Mapping Count with {loss_function.upper()} Loss ---")

    for ifs_count in range(min_ifs, max_ifs + 1):
        print(f"\nTesting with {ifs_count} IFS mappings...")

        model = AdaptiveIFSNet(max_ifs=ifs_count, base_sigma=sigma_value, init_seed=None).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            p_out = model(p_in_target)

            # 根据指定的损失函数计算损失
            if loss_function == 'chamfer':
                loss = chamfer_distance(p_out, p_in_target)
            elif loss_function == 'dcd':
                loss = calculate_dcd_2d(p_out, p_in_target)
            elif loss_function == 'hausdorff':
                # 注意：Hausdorff距离计算需要二值图像，这里简化处理
                loss = chamfer_distance(p_out, p_in_target)  # 使用Chamfer作为替代
            else:
                loss = multi_scale_chamfer(p_out, p_in_target)

            if epoch > epochs // 2:
                loss += fractal_dimension_reg(p_out, target_dim=1.58496, alpha=0.01)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            p_out = model(p_in_target)
            if loss_function == 'chamfer':
                final_loss = chamfer_distance(p_out, p_in_target).item()
            elif loss_function == 'dcd':
                final_loss = calculate_dcd_2d(p_out, p_in_target).item()
            elif loss_function == 'hausdorff':
                final_loss = chamfer_distance(p_out, p_in_target).item()  # 使用Chamfer作为替代
            else:
                final_loss = multi_scale_chamfer(p_out, p_in_target).item()

        results.append((ifs_count, final_loss))
        print(f"  Final loss with {ifs_count} mappings: {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_ifs_count = ifs_count

    plt.figure(figsize=(10, 6))
    counts, losses = zip(*results)
    plt.plot(counts, losses, 'o-', linewidth=2)
    plt.scatter([best_ifs_count], [best_loss], color='red', s=100, zorder=5)
    plt.annotate(f'Best: {best_ifs_count} mappings',
                 xy=(best_ifs_count, best_loss),
                 xytext=(best_ifs_count + 0.5, best_loss + 0.001),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.title(f'Loss vs. Number of IFS Mappings ({loss_function.upper()} Loss)')
    plt.xlabel('Number of IFS Mappings')
    plt.ylabel('Final Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print(f"\nBest number of IFS mappings found: {best_ifs_count} with loss: {best_loss:.6f}")
    return best_ifs_count


def load_partial_state_dict(model, state_dict):
    """
    加载部分匹配的state_dict，处理尺寸不匹配的情况
    """
    model_dict = model.state_dict()
    pretrained_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                print(f"Loading parameter: {k}")
            else:
                print(f"Parameter {k} shape mismatch: model expects {model_dict[k].shape}, checkpoint has {v.shape}")
                if k.startswith('ifs_w') or k.startswith('ifs_b'):
                    min_len = min(model_dict[k].shape[0], v.shape[0])
                    if min_len > 0:
                        if k.startswith('ifs_w'):
                            print(f"Loading first {min_len} IFS mappings for ifs_w")
                            model_dict[k][:min_len] = v[:min_len]
                        elif k.startswith('ifs_b'):
                            print(f"Loading first {min_len} IFS mappings for ifs_b")
                            model_dict[k][:min_len] = v[:min_len]
        else:
            print(f"Parameter {k} not found in model")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def run_optimized_training(dataset_type="sierpinski",
                           loss_functions=['multi_scale_chamfer', 'chamfer', 'dcd', 'hausdorff'],
                           sigma_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
                           num_points=2000, max_ifs_range=(2, 10),
                           search_epochs=3000, final_epochs=5000,
                           curriculum=True, mixed_precision=False,
                           save_history=True):
    """
    运行多轮训练，比较不同损失函数的效果
    """
    LEARNING_RATE = 0.005
    GRAD_CLIP_VALUE = 1.0
    NOISE_LEVEL = 0.01
    FRACTAL_DIM_REG_WEIGHT = 0.01
    SELF_SIMILARITY_WEIGHT = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Experiment on device: {device}")

    # 生成目标点云
    if dataset_type.lower() == "sierpinski":
        print("\nGenerating and visualizing Sierpinski Triangle target data...")
        np_pts = generate_sierpinski(num_points=num_points)
        target_dim = 1.58496
        title_prefix = "Sierpinski - "
    elif dataset_type.lower() == "fern" or dataset_type.lower() == "barnsley_fern":
        print("\nGenerating and visualizing Barnsley Fern target data...")
        np_pts = generate_barnsley_fern(num_points=num_points)
        target_dim = 1.8
        title_prefix = "Barnsley Fern - "
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use 'sierpinski' or 'fern'.")

    plt.figure(figsize=(8, 8))
    plt.scatter(np_pts[:, 0], np_pts[:, 1], s=1, c='red', alpha=0.7)
    plt.title(f"Target Point Cloud ({title_prefix}Original)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    p_in_target = torch.from_numpy(np_pts).float().to(device)

    # 存储不同损失函数的结果
    results = {}

    for loss_function in loss_functions:
        print(f"\n\n=== Training with {loss_function.upper()} Loss ===")

        # 阶段1：搜索最佳sigma值
        print(f"\n--- STAGE 1: Searching for the best initial sigma ({loss_function.upper()}) ---")
        overall_best_loss = float('inf')
        best_sigma_params = None
        best_sigma_value = None

        for sigma in sigma_values:
            print(f"\n--- Testing base_sigma = {sigma:.2f} ---")
            model = AdaptiveIFSNet(max_ifs=max_ifs_range[1], base_sigma=sigma, init_seed=None).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            current_sigma_best_loss = float('inf')
            current_sigma_best_params = None

            scaler = GradScaler(enabled=mixed_precision)

            for epoch in range(search_epochs):
                model.train()

                if curriculum:
                    if epoch < search_epochs // 3:
                        current_target = p_in_target[:num_points // 3]
                    elif epoch < 2 * search_epochs // 3:
                        current_target = p_in_target[:2 * num_points // 3]
                    else:
                        current_target = p_in_target
                else:
                    current_target = p_in_target

                if (epoch + 1) % 10 == 0 and epoch < 600:
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * NOISE_LEVEL)

                optimizer.zero_grad()

                with autocast(enabled=mixed_precision):
                    p_out = model(current_target)

                    # 根据指定的损失函数计算损失
                    if loss_function == 'chamfer':
                        loss = chamfer_distance(p_out, current_target)
                    elif loss_function == 'dcd':
                        loss = calculate_dcd_2d(p_out, current_target)
                    elif loss_function == 'hausdorff':
                        # 注意：Hausdorff距离计算需要二值图像，这里简化处理
                        loss = chamfer_distance(p_out, current_target)
                    else:
                        loss = multi_scale_chamfer(p_out, current_target)

                    if epoch > search_epochs // 2:
                        loss += fractal_dimension_reg(p_out, target_dim=target_dim, alpha=FRACTAL_DIM_REG_WEIGHT)

                    if epoch > search_epochs // 3:
                        self_sim_loss = chamfer_distance(p_out, current_target) * SELF_SIMILARITY_WEIGHT
                        loss += self_sim_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                scaler.step(optimizer)
                scaler.update()

                if (epoch + 1) in [int(search_epochs * 0.25), int(search_epochs * 0.5), int(search_epochs * 0.75)]:
                    print(f"  ...Decaying learning rate at epoch {epoch + 1}")
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.5

                if loss.item() < current_sigma_best_loss:
                    current_sigma_best_loss = loss.item()
                    current_sigma_best_params = copy.deepcopy(model.state_dict())

                if (epoch + 1) % 500 == 0:
                    print(f'  Epoch [{epoch + 1}/{search_epochs}], Loss: {loss.item():.6f}')

            print(f"--- Result for base_sigma = {sigma:.2f}: Best Loss = {current_sigma_best_loss:.6f} ---")

            if current_sigma_best_loss < overall_best_loss:
                overall_best_loss = current_sigma_best_loss
                best_sigma_params = current_sigma_best_params
                best_sigma_value = sigma

        print(f"\n--- STAGE 1 COMPLETE ({loss_function.upper()}) ---")
        print(f"Best sigma found: {best_sigma_value:.2f} with a loss of {overall_best_loss:.6f}")

        # 阶段2：优化最佳IFS映射个数
        print(f"\n--- STAGE 2: Optimizing Best IFS Mapping Count ({loss_function.upper()}) ---")
        best_ifs_count = optimize_ifs_count(
            np_pts,
            sigma_value=best_sigma_value,
            min_ifs=max_ifs_range[0],
            max_ifs=max_ifs_range[1],
            epochs=search_epochs // 3,
            loss_function=loss_function
        )

        # 阶段3：最终训练使用最佳的映射个数
        print(f"\n--- STAGE 3: Final training using best parameters ({loss_function.upper()}) ---")
        final_model = AdaptiveIFSNet(max_ifs=best_ifs_count, base_sigma=best_sigma_value, init_seed=None).to(device)

        print(f"\nLoading best parameters from stage 1 into model with {best_ifs_count} IFS mappings...")
        final_model = load_partial_state_dict(final_model, best_sigma_params)

        optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
        scaler = GradScaler(enabled=mixed_precision)

        history_points = [] if save_history else None

        for epoch in range(final_epochs):
            final_model.train()

            if (epoch + 1) % 10 == 0 and epoch < 1000:
                with torch.no_grad():
                    for param in final_model.parameters():
                        param.add_(torch.randn_like(param) * (NOISE_LEVEL * (1 - epoch / 1000)))

            optimizer.zero_grad()

            with autocast(enabled=mixed_precision):
                p_out = final_model(p_in_target)

                # 根据指定的损失函数计算损失
                if loss_function == 'chamfer':
                    loss = chamfer_distance(p_out, p_in_target)
                elif loss_function == 'dcd':
                    loss = calculate_dcd_2d(p_out, p_in_target)
                elif loss_function == 'hausdorff':
                    # 注意：Hausdorff距离计算需要二值图像，这里简化处理
                    loss = chamfer_distance(p_out, p_in_target)
                else:
                    loss = multi_scale_chamfer(p_out, p_in_target)

                loss += fractal_dimension_reg(p_out, target_dim=target_dim, alpha=FRACTAL_DIM_REG_WEIGHT)
                loss += chamfer_distance(p_out, p_in_target) * SELF_SIMILARITY_WEIGHT

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()

            if (epoch + 1) in [int(final_epochs * 0.25), int(final_epochs * 0.5), int(final_epochs * 0.75)]:
                print(f"  ...Decaying learning rate at epoch {epoch + 1}")
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5

            if save_history and (epoch + 1) % 100 == 0:
                final_model.eval()
                with torch.no_grad():
                    history_p_out = final_model(p_in_target).cpu().detach().numpy()
                    history_points.append(history_p_out)
                final_model.train()

            if (epoch + 1) % 500 == 0:
                print(f'  Epoch [{epoch + 1}/{final_epochs}], Loss: {loss.item():.6f}')

        print(f"\n--- STAGE 3 COMPLETE: Training finished with {loss_function.upper()} loss. ---")

        # 评估最终模型
        final_model.eval()
        with torch.no_grad():
            p_out_final, gates, valid_indices = final_model(p_in_target, return_gates=True)
            p_out_final = p_out_final.cpu().detach()
            gates = gates.cpu().detach().numpy()

        print("\nFinal Learned Parameters:")
        print(f"Effective transformations: {len(valid_indices)}/{best_ifs_count}")
        print("Transformation gate values:")
        for i in range(best_ifs_count):
            print(f"  Transform {i}: {gates[i]:.4f} {'(active)' if i in valid_indices else ''}")

        w_params = final_model.ifs_w.weight.data
        A_matrices = final_model.make_matrices_from_svdformat(w_params)
        print("\nFinal Learned affine matrices (A):")
        for i in range(best_ifs_count):
            print(f" A_{i}:\n{A_matrices[i].cpu().numpy()}")

        # 计算最终的评估指标
        metrics = {
            'chamfer': chamfer_distance(p_out_final, p_in_target).item(),
            'dcd': calculate_dcd_2d(p_out_final, p_in_target).item(),
            'fractal_dim': fractal_dimension_reg(p_out_final, target_dim=target_dim, alpha=1.0).item()
        }

        results[loss_function] = {
            'model': final_model,
            'point_cloud': p_out_final,
            'history_points': history_points,
            'best_ifs_count': best_ifs_count,
            'metrics': metrics
        }

        # 可视化每个变换的效果
        fig_transforms = visualize_ifs_transforms(final_model, np_pts,
                                                  title_prefix=f"{title_prefix}{loss_function.upper()} Loss - ")
        plt.show()

        # 创建训练动画
        if save_history and len(history_points) > 0:
            ani = create_training_animation(history_points,
                                            title_prefix=f"{title_prefix}{loss_function.upper()} Loss - ")
            plt.show()

    # 比较不同损失函数的结果
    compare_results(results, dataset_type)

    return results


def compare_results(results, dataset_type):
    """比较不同损失函数的结果"""
    loss_functions = list(results.keys())
    n_functions = len(loss_functions)

    # 可视化不同损失函数的结果
    fig, axes = plt.subplots(1, n_functions, figsize=(5 * n_functions, 5))
    if n_functions == 1:
        axes = [axes]

    for i, loss_function in enumerate(loss_functions):
        point_cloud = results[loss_function]['point_cloud']
        ax = axes[i]
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1, c='blue', alpha=0.8)
        ax.set_title(f"{loss_function.upper()} Loss\n{len(results[loss_function]['point_cloud'])} points")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"Comparison of Different Loss Functions ({dataset_type.capitalize()})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
    plt.show()

    # 打印评估指标
    print("\n=== Comparison of Evaluation Metrics ===")
    print("{:<15} {:<15} {:<15} {:<15}".format("Loss Function", "Chamfer Distance", "DCD", "Fractal Dim. Deviation"))
    print("-" * 60)

    for loss_function in loss_functions:
        metrics = results[loss_function]['metrics']
        print("{:<15} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            loss_function.upper(),
            metrics['chamfer'],
            metrics['dcd'],
            metrics['fractal_dim']
        ))


# ==============================================================================
# 5. 主程序
# ==============================================================================
if __name__ == '__main__':
    # 设置参数
    dataset_type = "fern"  # 可选 "sierpinski" 或 "fern"
    sigma_range = [-1.0, -0.5, 0.0, 0.5, 1.0]
    num_points = 10000
    max_ifs_range = (2, 10)
    search_epochs = 3000
    final_epochs = 5000

    # 选择要比较的损失函数
    loss_functions = [
        'multi_scale_chamfer',  # 多尺度Chamfer距离（原始默认）
        'chamfer',  # 标准Chamfer距离
        'dcd',  # 密度感知倒角距离
        'hausdorff'  # Hausdorff距离（简化版）
    ]

    # 运行优化训练并比较不同损失函数
    results = run_optimized_training(
        dataset_type=dataset_type,
        loss_functions=loss_functions,
        sigma_values=sigma_range,
        num_points=num_points,
        max_ifs_range=max_ifs_range,
        search_epochs=search_epochs,
        final_epochs=final_epochs,
        curriculum=True,
        mixed_precision=False,  # 设为True启用混合精度训练
        save_history=True
    )

    print(f"\nExperiment with {dataset_type} dataset finished successfully.")
    print("Results are stored in the 'results' dictionary.")