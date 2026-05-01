"""
Simple Multimodal Integration - 简化多模态损失集成
提供最小化修改的多模态损失函数解决方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

class SimpleMultiModalCriterion(nn.Module):
    """简化版多模态损失函数 - 可直接替换MSELoss"""
    
    def __init__(self, common_indices: List[int], common_weight: float = 1.2):
        super().__init__()
        self.common_indices = common_indices
        self.common_weight = common_weight
        self.mse_loss = nn.MSELoss()
        
        print("SimpleMultiModalCriterion初始化:")
        print(f"  common_indices: {common_indices}")
        print(f"  common_weight: {common_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                channel_idx: Optional[int] = None, is_common: Optional[bool] = None) -> torch.Tensor:
        """
        兼容现有MSELoss接口的多模态损失
        
        Args:
            pred: 预测值 [T] 或 [B, T]
            target: 目标值 [T] 或 [B, T]
            channel_idx: 当前通道索引（可选）
            is_common: 是否为common模态（可选）
        """
        
        # 基础MSE损失
        mse = self.mse_loss(pred, target)
        
        # 如果是common模态，应用权重
        if is_common or (channel_idx is not None and channel_idx in self.common_indices):
            return self.common_weight * mse
        else:
            return mse

def create_simple_multimodal_criterion(config: Dict[str, Any]) -> SimpleMultiModalCriterion:
    """创建简化版多模态损失函数"""
    
    common_modalities = config.get('common_modalities', [])
    
    # 构建所有模态列表
    all_modalities = common_modalities.copy()
    dataset_modalities = config.get('dataset_modalities', {})
    
    for dataset_name, modalities in dataset_modalities.items():
        have_mods = modalities.get('have', [])
        need_mods = modalities.get('need', [])
        for mod in have_mods + need_mods:
            if mod not in all_modalities:
                all_modalities.append(mod)
    
    # 获取common模态的索引
    common_indices = []
    for i, mod in enumerate(all_modalities):
        if mod in common_modalities:
            common_indices.append(i)
    
    # 获取权重配置
    loss_config = config.get('loss_config', {})
    multimodal_loss_config = config.get('multimodal_loss', {})
    
    # 优先使用新的multimodal_loss配置
    common_weight = multimodal_loss_config.get('common_weight', 
                                              loss_config.get('common_weight', 1.2))
    
    return SimpleMultiModalCriterion(common_indices, common_weight)

def test_simple_multimodal():
    """测试简化版多模态损失函数"""
    
    print("=" * 60)
    print("简化版多模态损失函数测试")
    print("=" * 60)
    
    # 模拟配置
    config = {
        'common_modalities': ['acc_x', 'acc_y', 'acc_z', 'ppg', 'gsr', 'hr', 'skt'],
        'dataset_modalities': {
            'FM': {'have': ['alpha_tp10', 'beta_tp10'], 'need': ['space_distance']},
            'OD': {'have': ['space_distance'], 'need': ['alpha_tp10', 'beta_tp10']}
        },
        'multimodal_loss': {
            'common_weight': 1.2,
            'have_weight': 1.0
        }
    }
    
    # 创建损失函数
    criterion = create_simple_multimodal_criterion(config)
    
    # 测试损失计算
    pred = torch.randn(100)
    target = torch.randn(100)
    
    # 测试common模态损失
    common_loss = criterion(pred, target, channel_idx=0, is_common=True)
    print(f"Common模态损失: {common_loss.item():.6f}")
    
    # 测试have模态损失
    have_loss = criterion(pred, target, channel_idx=7, is_common=False)
    print(f"Have模态损失: {have_loss.item():.6f}")
    
    # 测试权重效果
    ratio = common_loss.item() / have_loss.item()
    print(f"Common/Have损失比例: {ratio:.3f} (预期约为1.2)")
    
    return True

if __name__ == "__main__":
    # 运行测试
    test_simple_multimodal()
    
    print(f"\n" + "=" * 60)
    print("简化版多模态损失集成方案")
    print("=" * 60)
    print("\n优势：")
    print("✓ 最小代码修改，兼容现有逻辑")
    print("✓ Common模态现在参与损失计算")
    print("✓ 保持原有训练流程不变")
    print("✓ 添加详细的损失监控")
    print("\n使用步骤：")
    print("1. 将config.yaml中的multimodal_loss启用")
    print("2. 在train.py中导入并使用create_simple_multimodal_criterion")
    print("3. 观察common_loss和have_loss的变化趋势")
    print("4. 监控分类性能是否因为更好的重建而提升")
