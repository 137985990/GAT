# domain_adaptation.py
"""
跨域适应模块

包含：
1. GradientReversalLayer (GRL) - 梯度反转层
2. DomainDiscriminator - 域判别器
3. MMD Loss - 最大均值差异损失
4. CORAL Loss - 相关性对齐损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, List


# ============== 梯度反转层 ==============
class GradientReversalFunction(Function):
    """
    梯度反转函数
    前向传播: 恒等变换
    反向传播: 梯度乘以 -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        return -lambda_ * grads, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层
    
    用于域对抗训练：
    - 前向传播时不改变特征
    - 反向传播时反转梯度，使得特征提取器学习域不变特征
    
    Args:
        lambda_: 梯度反转强度，可以随训练进度调整
    """
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """动态调整反转强度"""
        self.lambda_ = lambda_


# ============== 域判别器 ==============
class DomainDiscriminator(nn.Module):
    """
    域判别器网络
    
    输入: latent 特征向量
    输出: 域分类 logits
    
    Args:
        in_features: 输入特征维度
        hidden_dim: 隐藏层维度
        num_domains: 域的数量（数据集数量）
        num_layers: 隐藏层数量
        dropout: dropout 比例
        use_grl: 是否在内部使用 GRL
        grl_lambda: GRL 的 lambda 值
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, 
                 num_domains: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, use_grl: bool = True,
                 grl_lambda: float = 1.0):
        super().__init__()
        
        self.use_grl = use_grl
        if use_grl:
            self.grl = GradientReversalLayer(grl_lambda)
        
        layers = []
        current_dim = in_features
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, num_domains))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_features) 特征向量
        Returns:
            (batch_size, num_domains) 域分类 logits
        """
        if self.use_grl:
            x = self.grl(x)
        return self.classifier(x)
    
    def set_lambda(self, lambda_: float):
        """设置 GRL 的 lambda 值"""
        if self.use_grl:
            self.grl.set_lambda(lambda_)


# ============== MMD 损失 ==============
def compute_kernel(x: torch.Tensor, y: torch.Tensor, 
                   kernel_type: str = 'rbf',
                   kernel_mul: float = 2.0,
                   kernel_num: int = 5,
                   fix_sigma: Optional[float] = None) -> torch.Tensor:
    """
    计算核矩阵
    
    Args:
        x: (n, d) 源域特征
        y: (m, d) 目标域特征
        kernel_type: 核函数类型 ('rbf', 'linear')
        kernel_mul: 带宽乘数
        kernel_num: 多核数量
        fix_sigma: 固定带宽（如果设置则使用）
    
    Returns:
        (n+m, n+m) 核矩阵
    """
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)
    
    # 计算欧氏距离矩阵
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    L2_distance = ((total0 - total1) ** 2).sum(dim=2)
    
    if kernel_type == 'linear':
        return total @ total.t()
    
    # RBF 核
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    
    # 多核带宽
    bandwidth_list = [bandwidth * (kernel_mul ** (i - kernel_num // 2)) 
                      for i in range(kernel_num)]
    
    kernel_val = torch.zeros_like(L2_distance)
    for bandwidth in bandwidth_list:
        kernel_val += torch.exp(-L2_distance / (2 * bandwidth + 1e-8))
    
    return kernel_val / kernel_num


def mmd_loss(source: torch.Tensor, target: torch.Tensor,
             kernel_type: str = 'rbf',
             kernel_mul: float = 2.0,
             kernel_num: int = 5,
             fix_sigma: Optional[float] = None) -> torch.Tensor:
    """
    计算 MMD (Maximum Mean Discrepancy) 损失
    
    衡量两个分布之间的差异，最小化 MMD 使得源域和目标域特征分布对齐
    
    Args:
        source: (n, d) 源域特征
        target: (m, d) 目标域特征
        kernel_type: 核函数类型
        kernel_mul: 带宽乘数
        kernel_num: 多核数量
        fix_sigma: 固定带宽
    
    Returns:
        MMD 损失值 (标量)
    """
    n_source = source.size(0)
    n_target = target.size(0)
    
    if n_source == 0 or n_target == 0:
        return torch.tensor(0.0, device=source.device)
    
    kernels = compute_kernel(source, target, kernel_type, 
                            kernel_mul, kernel_num, fix_sigma)
    
    # 分割核矩阵
    XX = kernels[:n_source, :n_source]  # 源-源
    YY = kernels[n_source:, n_source:]  # 目标-目标
    XY = kernels[:n_source, n_source:]  # 源-目标
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return torch.clamp(mmd, min=0.0)


class MMDLoss(nn.Module):
    """
    MMD 损失模块
    
    支持多域对齐：计算所有域对之间的 MMD 并求平均
    """
    def __init__(self, kernel_type: str = 'rbf',
                 kernel_mul: float = 2.0,
                 kernel_num: int = 5,
                 fix_sigma: Optional[float] = None):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
    
    def forward(self, features: torch.Tensor, 
                domain_labels: torch.Tensor) -> torch.Tensor:
        """
        计算多域 MMD 损失
        
        Args:
            features: (batch_size, feature_dim) 特征
            domain_labels: (batch_size,) 域标签
        
        Returns:
            平均 MMD 损失
        """
        unique_domains = domain_labels.unique()
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device)
        
        total_mmd = torch.tensor(0.0, device=features.device)
        n_pairs = 0
        
        for i, d1 in enumerate(unique_domains):
            for d2 in unique_domains[i+1:]:
                feat1 = features[domain_labels == d1]
                feat2 = features[domain_labels == d2]
                
                if len(feat1) > 0 and len(feat2) > 0:
                    total_mmd += mmd_loss(feat1, feat2, 
                                         self.kernel_type,
                                         self.kernel_mul,
                                         self.kernel_num,
                                         self.fix_sigma)
                    n_pairs += 1
        
        return total_mmd / max(n_pairs, 1)


# ============== CORAL 损失 ==============
def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算 CORAL (Correlation Alignment) 损失
    
    对齐源域和目标域的二阶统计量（协方差矩阵）
    
    Args:
        source: (n, d) 源域特征
        target: (m, d) 目标域特征
    
    Returns:
        CORAL 损失值 (标量)
    """
    d = source.size(1)
    n_source = source.size(0)
    n_target = target.size(0)
    
    if n_source < 2 or n_target < 2:
        return torch.tensor(0.0, device=source.device)
    
    # 中心化
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    # 计算协方差矩阵
    cov_source = (source_centered.t() @ source_centered) / (n_source - 1)
    cov_target = (target_centered.t() @ target_centered) / (n_target - 1)
    
    # CORAL Loss = 1/(4*d^2) * ||C_s - C_t||_F^2
    loss = torch.sum((cov_source - cov_target) ** 2) / (4 * d * d)
    
    return loss


class CORALLoss(nn.Module):
    """
    CORAL 损失模块
    
    支持多域对齐
    """
    def forward(self, features: torch.Tensor,
                domain_labels: torch.Tensor) -> torch.Tensor:
        """
        计算多域 CORAL 损失
        
        Args:
            features: (batch_size, feature_dim) 特征
            domain_labels: (batch_size,) 域标签
        
        Returns:
            平均 CORAL 损失
        """
        unique_domains = domain_labels.unique()
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device)
        
        total_coral = torch.tensor(0.0, device=features.device)
        n_pairs = 0
        
        for i, d1 in enumerate(unique_domains):
            for d2 in unique_domains[i+1:]:
                feat1 = features[domain_labels == d1]
                feat2 = features[domain_labels == d2]
                
                if len(feat1) > 1 and len(feat2) > 1:
                    total_coral += coral_loss(feat1, feat2)
                    n_pairs += 1
        
        return total_coral / max(n_pairs, 1)


# ============== 域自适应包装器 ==============
class DomainAdaptationModule(nn.Module):
    """
    域自适应模块
    
    整合所有跨域组件，方便在训练中使用
    
    Args:
        feature_dim: 特征维度
        num_domains: 域数量
        use_adversarial: 是否使用对抗训练
        use_mmd: 是否使用 MMD 损失
        use_coral: 是否使用 CORAL 损失
        adversarial_weight: 对抗损失权重
        mmd_weight: MMD 损失权重
        coral_weight: CORAL 损失权重
    """
    def __init__(self, feature_dim: int, num_domains: int = 4,
                 use_adversarial: bool = True,
                 use_mmd: bool = True,
                 use_coral: bool = False,
                 adversarial_weight: float = 1.0,
                 mmd_weight: float = 0.1,
                 coral_weight: float = 0.1,
                 discriminator_hidden: int = 256,
                 discriminator_layers: int = 2,
                 discriminator_dropout: float = 0.1):
        super().__init__()
        
        self.use_adversarial = use_adversarial
        self.use_mmd = use_mmd
        self.use_coral = use_coral
        
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight
        
        if use_adversarial:
            self.domain_discriminator = DomainDiscriminator(
                in_features=feature_dim,
                hidden_dim=discriminator_hidden,
                num_domains=num_domains,
                num_layers=discriminator_layers,
                dropout=discriminator_dropout,
                use_grl=True
            )
        
        if use_mmd:
            self.mmd_loss = MMDLoss()
        
        if use_coral:
            self.coral_loss = CORALLoss()
        
        # 域分类损失
        self.domain_criterion = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor,
                domain_labels: torch.Tensor) -> dict:
        """
        计算所有域自适应损失
        
        Args:
            features: (batch_size, feature_dim) 特征
            domain_labels: (batch_size,) 域标签 (0, 1, 2, ...)
        
        Returns:
            dict: {
                'total': 总损失,
                'adversarial': 对抗损失,
                'mmd': MMD 损失,
                'coral': CORAL 损失,
                'domain_acc': 域分类准确率
            }
        """
        losses = {
            'total': torch.tensor(0.0, device=features.device),
            'adversarial': torch.tensor(0.0, device=features.device),
            'mmd': torch.tensor(0.0, device=features.device),
            'coral': torch.tensor(0.0, device=features.device),
            'domain_acc': 0.0
        }
        
        # 对抗损失
        if self.use_adversarial:
            domain_logits = self.domain_discriminator(features)
            adv_loss = self.domain_criterion(domain_logits, domain_labels)
            losses['adversarial'] = adv_loss
            losses['total'] = losses['total'] + self.adversarial_weight * adv_loss
            
            # 域分类准确率（希望越低越好，说明特征越域不变）
            domain_preds = domain_logits.argmax(dim=1)
            losses['domain_acc'] = (domain_preds == domain_labels).float().mean().item()
        
        # MMD 损失
        if self.use_mmd:
            mmd = self.mmd_loss(features, domain_labels)
            losses['mmd'] = mmd
            losses['total'] = losses['total'] + self.mmd_weight * mmd
        
        # CORAL 损失
        if self.use_coral:
            coral = self.coral_loss(features, domain_labels)
            losses['coral'] = coral
            losses['total'] = losses['total'] + self.coral_weight * coral
        
        return losses
    
    def set_lambda(self, lambda_: float):
        """设置 GRL lambda（用于训练进度调度）"""
        if self.use_adversarial:
            self.domain_discriminator.set_lambda(lambda_)


# ============== Lambda 调度器 ==============
def get_lambda_schedule(epoch: int, max_epochs: int, 
                        schedule_type: str = 'linear',
                        gamma: float = 10.0) -> float:
    """
    计算 GRL lambda 值的调度
    
    随着训练进行，逐渐增加 lambda，让模型先学习任务再学习域不变性
    
    Args:
        epoch: 当前 epoch
        max_epochs: 总 epoch 数
        schedule_type: 调度类型 ('linear', 'exp', 'step')
        gamma: 指数调度的 gamma 参数
    
    Returns:
        lambda 值 (0 到 1)
    """
    p = epoch / max_epochs
    
    if schedule_type == 'linear':
        return p
    elif schedule_type == 'exp':
        # DANN 论文中的调度：lambda = 2 / (1 + exp(-gamma * p)) - 1
        return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0
    elif schedule_type == 'step':
        # 阶梯式：前半段 0，后半段 1
        return 1.0 if p > 0.5 else 0.0
    else:
        return 1.0


# ============== 工具函数 ==============
def get_domain_id_mapping(sources: List[str]) -> dict:
    """
    获取数据集名称到域 ID 的映射
    
    Args:
        sources: 数据集名称列表
    
    Returns:
        {'FM': 0, 'OD': 1, 'MEFAR': 2, 'DROZY': 3}
    """
    unique_sources = sorted(set(sources))
    return {src: i for i, src in enumerate(unique_sources)}


if __name__ == '__main__':
    # 测试代码
    print("Testing Domain Adaptation Module...")
    
    # 创建模拟数据
    batch_size = 32
    feature_dim = 64
    num_domains = 4
    
    features = torch.randn(batch_size, feature_dim)
    domain_labels = torch.randint(0, num_domains, (batch_size,))
    
    # 测试 GRL
    grl = GradientReversalLayer(lambda_=1.0)
    out = grl(features)
    print(f"GRL output shape: {out.shape}")
    
    # 测试域判别器
    discriminator = DomainDiscriminator(feature_dim, num_domains=num_domains)
    domain_logits = discriminator(features)
    print(f"Domain logits shape: {domain_logits.shape}")
    
    # 测试 MMD
    source = features[:16]
    target = features[16:]
    mmd = mmd_loss(source, target)
    print(f"MMD loss: {mmd.item():.4f}")
    
    # 测试 CORAL
    coral = coral_loss(source, target)
    print(f"CORAL loss: {coral.item():.4f}")
    
    # 测试完整模块
    da_module = DomainAdaptationModule(
        feature_dim=feature_dim,
        num_domains=num_domains,
        use_adversarial=True,
        use_mmd=True,
        use_coral=True
    )
    
    losses = da_module(features, domain_labels)
    print(f"\nDomain Adaptation Losses:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Adversarial: {losses['adversarial'].item():.4f}")
    print(f"  MMD: {losses['mmd'].item():.4f}")
    print(f"  CORAL: {losses['coral'].item():.4f}")
    print(f"  Domain Acc: {losses['domain_acc']:.2%}")
    
    print("\nAll tests passed!")
