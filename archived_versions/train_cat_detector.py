#!/usr/bin/env python3
"""
简化版猫叫声检测模型训练脚本
这个脚本专注于解决负样本得分高于正样本的问题
"""

import os
import torch
import torchaudio
import numpy as np
import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 简单的CNN模型 - 不使用复杂的Transformer架构，降低过拟合风险
class SimpleCatSoundClassifier(nn.Module):
    def __init__(self):
        super(SimpleCatSoundClassifier, self).__init__()
        
        # 特征提取层 - 使用简单的CNN
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 使用较低的dropout率
            nn.Linear(128 * 8 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 二分类问题，输出单个值
        )
        
    def forward(self, x):
        # 输入x形状为[batch_size, time_frames, freq_bins]
        # 需要调整为[batch_size, channels, freq_bins, time_frames]
        x = x.unsqueeze(1)  # 添加通道维度
        x = x.permute(0, 1, 2, 3)  # 交换维度
        
        # 通过特征提取层
        x = self.features(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 通过分类层
        x = self.classifier(x)
        
        return x

# 数据集类 - 简化版
class CatSoundDataset(Dataset):
    def __init__(self, audio_files, labels, augment=False):
        self.audio_files = audio_files
        self.labels = labels
        self.augment = augment
        
        # 梅尔频谱图转换器
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        # 数据增强转换器
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # 加载音频文件
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # 如果音频太短，进行填充
            if waveform.shape[1] < 16000 * 5:  # 小于5秒
                padding = 16000 * 5 - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # 如果音频太长，进行截断
            if waveform.shape[1] > 16000 * 5:  # 大于5秒
                waveform = waveform[:, :16000 * 5]
            
            # 应用数据增强
            if self.augment and torch.rand(1).item() > 0.5:
                # 随机音量调整
                gain_factor = 0.5 + torch.rand(1).item()
                waveform = waveform * gain_factor
                
                # 随机添加噪声
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            
            # 计算梅尔频谱图
            mel = self.mel_spec(waveform)
            mel_db = self.amplitude_to_db(mel)
            
            # 标准化到[0,1]范围
            mel_db = (mel_db + 80) / 80
            
            # 应用频谱增强
            if self.augment and torch.rand(1).item() > 0.5:
                mel_db = self.time_mask(mel_db)
                mel_db = self.freq_mask(mel_db)
            
            # 确保尺寸一致（128频率bins x 256时间帧)
            if mel_db.shape[2] < 256:
                padding = 256 - mel_db.shape[2]
                mel_db = torch.nn.functional.pad(mel_db, (0, padding))
            else:
                mel_db = mel_db[:, :, :256]
            
            # 调整格式
            mel_db = mel_db.squeeze(0)
            
            # 返回特征和标签
            return mel_db, float(label)
            
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {str(e)}")
            # 返回一个随机的梅尔频谱图和标签
            random_mel = torch.rand(128, 256)
            return random_mel, float(label)

# 加载音频文件
def load_audio_files():
    audio_dir = 'audio'
    if not os.path.exists(audio_dir):
        print("创建音频目录结构...")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(os.path.join(audio_dir, 'miao'), exist_ok=True)
        os.makedirs(os.path.join(audio_dir, 'other'), exist_ok=True)
        print(f"已创建目录结构:")
        print(f"- {audio_dir}/miao  (猫叫声)")
        print(f"- {audio_dir}/other (其他声音)")
        return [], []
    
    audio_files = []
    labels = []
    
    # 加载猫叫声（标签为1）
    miao_files = glob.glob(os.path.join(audio_dir, 'miao', '*.wav'))
    for file in miao_files:
        audio_files.append(file)
        labels.append(1)
    
    # 加载其他声音（标签为0）
    other_files = glob.glob(os.path.join(audio_dir, 'other', '*.wav'))
    for file in other_files:
        audio_files.append(file)
        labels.append(0)
    
    return audio_files, labels

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for mel_spec, labels in data_loader:
            mel_spec = mel_spec.to(device)
            outputs = model(mel_spec)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_probs.extend(probs if isinstance(probs, np.ndarray) else [probs])
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 输出分数分布信息
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    print(f"正样本数: {len(pos_probs)}, 负样本数: {len(neg_probs)}")
    
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        print(f"正样本平均分数: {np.mean(pos_probs):.6f}, 中位数: {np.median(pos_probs):.6f}")
        print(f"负样本平均分数: {np.mean(neg_probs):.6f}, 中位数: {np.median(neg_probs):.6f}")
    
    # 计算最优阈值
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 找到最佳F1分数对应的阈值
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])  # 忽略最后一个，它没有对应的阈值
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        # 如果没有计算出阈值，使用0.5
        best_threshold = 0.5
        best_f1 = 0.0
    
    # 如果计算出的最优阈值太低，可能表明模型有偏差
    if best_threshold < 0.01:
        print("警告: 计算的最优阈值非常低，可能表明模型对正样本评分过低")
        # 使用一个更合理的阈值
        alt_threshold = np.percentile(all_probs, 70)  # 使用70%分位数作为备选阈值
        print(f"提供一个备选阈值: {alt_threshold:.6f}")
        
        # 检查哪个阈值更合理
        preds_best = (all_probs >= best_threshold).astype(int)
        preds_alt = (all_probs >= alt_threshold).astype(int)
        
        f1_best = f1_score(all_labels, preds_best)
        f1_alt = f1_score(all_labels, preds_alt)
        
        if f1_alt > f1_best * 0.9:  # 如果备选阈值的F1至少是最佳的90%
            print(f"使用备选阈值: {alt_threshold:.6f} (F1: {f1_alt:.4f})")
            best_threshold = alt_threshold
            best_f1 = f1_alt
        else:
            print(f"保留计算的最优阈值: {best_threshold:.6f} (F1: {f1_best:.4f})")
    
    # 计算预测和性能指标
    predictions = (all_probs >= best_threshold).astype(int)
    
    # 计算各种指标
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    
    # 避免除零错误
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.5  # 如果计算AUC失败，使用随机猜测的值
    
    metrics = {
        'threshold': best_threshold,
        'f1_score': best_f1,
        'sensitivity': sensitivity,  # 真阳性率/召回率
        'specificity': specificity,  # 真阴性率
        'precision': precision,      # 精确率
        'roc_auc': roc_auc,          # ROC曲线下面积
        'val_acc': (sensitivity + specificity) / 2  # 平衡准确率
    }
    
    return metrics

# 主训练函数
def train_model():
    # 加载数据
    print("加载音频文件...")
    audio_files, labels = load_audio_files()
    
    if len(audio_files) == 0:
        print("没有找到音频文件。请将音频文件放入相应目录后再试。")
        return
    
    # 输出数据集信息
    num_samples = len(audio_files)
    num_positive = sum(labels)
    num_negative = num_samples - num_positive
    print(f"数据集信息:")
    print(f"- 总样本数: {num_samples}")
    print(f"- 猫叫声样本数 (标签=1): {num_positive}")
    print(f"- 其他声音样本数 (标签=0): {num_negative}")
    
    # 处理类别不平衡
    pos_weight = torch.tensor([num_negative / max(num_positive, 1) * 10.0]).to(device)
    print(f"类别权重 - 正样本权重: {pos_weight.item():.2f}倍")
    
    # 分割数据集
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据集和数据加载器
    train_dataset = CatSoundDataset(train_files, train_labels, augment=True)
    val_dataset = CatSoundDataset(val_files, val_labels, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = SimpleCatSoundClassifier().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # 学习率调度器 - 当验证损失不再下降时减小学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练设置
    num_epochs = 30
    best_val_metrics = {'f1_score': 0}
    patience = 7
    no_improve_epochs = 0
    
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for mel_spec, labels in pbar:
            mel_spec = mel_spec.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            # 前向传播
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel_spec, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                mel_spec = mel_spec.to(device)
                labels = labels.unsqueeze(1).to(device)
                
                outputs = model(mel_spec)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 评估模型
        print(f"\n评估验证集性能...")
        val_metrics = evaluate_model(model, val_loader)
        
        # 输出本轮训练结果
        print(f"\nEpoch {epoch+1}/{num_epochs} 结果:")
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        print(f"验证指标:")
        print(f"- F1分数: {val_metrics['f1_score']:.4f}")
        print(f"- 敏感度: {val_metrics['sensitivity']:.4f}")
        print(f"- 特异度: {val_metrics['specificity']:.4f}")
        print(f"- ROC AUC: {val_metrics['roc_auc']:.4f}")
        print(f"- 最佳阈值: {val_metrics['threshold']:.6f}")
        
        # 检查是否是最佳模型
        if val_metrics['f1_score'] > best_val_metrics['f1_score']:
            best_val_metrics = val_metrics
            
            # 保存模型
            os.makedirs('models', exist_ok=True)
            model_path = 'models/cat_detector.pth'
            
            # 保存模型和相关信息
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': val_metrics['threshold'],
                'val_metrics': val_metrics,
                'val_acc': val_metrics['val_acc'],
                'epoch': epoch + 1
            }, model_path)
            
            print(f"\n保存最佳模型于 {model_path}")
            print(f"F1分数提高: {best_val_metrics['f1_score']:.4f}")
            
            # 同时创建与原始app.py兼容的版本
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': val_metrics['threshold'],
                'val_metrics': val_metrics,
                'val_acc': val_metrics['val_acc'],
                'epoch': epoch + 1
            }, 'models/ast_model.pth')
            
            print("同时创建与原始app.py兼容的模型版本")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"\n{no_improve_epochs} 轮未改进")
            
            if no_improve_epochs >= patience:
                print(f"\n早停: {patience} 轮未改进")
                break
    
    print("\n训练完成!")
    print("\n最佳模型性能:")
    for metric, value in best_val_metrics.items():
        print(f"- {metric}: {value:.6f}")
    
    # 为app_fixed.py创建另一个兼容版本
    best_model_path = 'models/cat_detector.pth'
    if os.path.exists(best_model_path):
        model_data = torch.load(best_model_path)
        torch.save(model_data, 'models/ast_model_improved.pth')
        print("\n已创建与app_fixed.py兼容的模型版本")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc() 