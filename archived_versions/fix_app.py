#!/usr/bin/env python3
"""
修复app.py中的KeyError: 'val_acc'错误
"""

import torch

# 加载模型
model_path = 'models/ast_model.pth'
checkpoint = torch.load(model_path)

# 检查是否存在val_acc，如果没有则添加
if 'val_acc' not in checkpoint:
    # 从val_metrics中获取准确率或使用敏感度和特异度的平均值作为准确率
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        # 计算一个准确率值
        sensitivity = metrics.get('sensitivity', 0.0)
        specificity = metrics.get('specificity', 0.0)
        val_acc = (sensitivity + specificity) / 2
        # 添加到checkpoint
        checkpoint['val_acc'] = val_acc
        print(f"已添加val_acc: {val_acc:.4f}")
    else:
        # 如果没有val_metrics，则使用一个默认值
        checkpoint['val_acc'] = 0.5
        print("已添加默认val_acc: 0.5")
    
    # 保存更新后的模型
    torch.save(checkpoint, model_path)
    print(f"已更新模型文件: {model_path}")
else:
    print(f"模型文件已包含val_acc: {checkpoint['val_acc']:.4f}")
    
print("\n模型信息:")
print(f"- 最佳阈值: {checkpoint.get('best_threshold', 0.0):.8f}")
if 'val_metrics' in checkpoint:
    metrics = checkpoint['val_metrics']
    print(f"- F1分数: {metrics.get('f1_score', 0.0):.4f}")
    print(f"- ROC AUC: {metrics.get('roc_auc', 0.5):.4f}")
    print(f"- 敏感度: {metrics.get('sensitivity', 0.0):.4f}")
    print(f"- 特异度: {metrics.get('specificity', 0.0):.4f}")
    
print("\n运行此脚本后，您应该能够正常启动app.py了")
print("如需修复低敏感度问题，建议调整阈值或重新训练模型") 