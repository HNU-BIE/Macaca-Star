import torch

def masked_l1_loss(fake_B, real_B, target_mask=None):
    """
    联合加权 L1 损失函数：
    - 全局 L1：提供背景正则化，压制反卷积带来的高频条纹伪影。
    - 掩码 L1：针对脑组织区域提供高强度的像素级重构约束。
    """
    # 1. 计算基础的绝对误差矩阵
    diff = torch.abs(fake_B - real_B)

    # 2. 计算全局 L1 Loss (包含背景)
    loss_global = diff.mean()

    # 3. 计算局部 Masked L1 Loss (仅限脑组织)
    if target_mask is None:
        # 假设背景为 0，提取 > 0 的区域作为组织掩码
        target_mask = (real_B > 0.0).float()

    masked_diff = diff * target_mask
    valid_pixels = target_mask.sum()
    loss_masked = masked_diff.sum() / (valid_pixels + 1e-8)

    # 4. 加权融合并返回
    total_loss = (0.7 * loss_global) + (0.3 * loss_masked)
    return total_loss
    # """
    # 带掩码的 L1 损失：忽略纯黑背景，只对脑组织区域计算梯度
    # """
    # # 如果你在 Dataset 里没有单独提取 mask，可以直接通过像素值大于 0 动态生成
    # if target_mask is None:
    #     # 假设纯黑背景严格为 0 (或者可以设为 > -0.99 如果你做过 [-1, 1] 归一化)
    #     target_mask = (real_B > 0.0).float()
    #
    # # 计算绝对误差，并乘以 mask 抹平背景的误差
    # diff = torch.abs(fake_B - real_B) * target_mask
    #
    # # 核心：计算均值时，除以脑组织的有效像素数量，而不是整张图 (H*W)
    # # 加上 1e-8 防止全黑图像导致除以 0 报错
    # valid_pixels = target_mask.sum()
    # loss = diff.sum() / (valid_pixels + 1e-8)
    #
    # return loss