import os 
import torch  
import torchaudio  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from torch import nn  
from tqdm import tqdm  
from src.models import ASTModel 
import glob
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, augment=False):
        self.audio_files = audio_files 
        self.labels = labels  
        self.augment = augment

        self.target_length = 512  
        self.mel_bins = 128  
        self.fmin = 50  
        self.fmax = 8000  
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024,  
            hop_length=160, 
            win_length=400, 
            n_mels=self.mel_bins,  
            f_min=self.fmin, 
            f_max=self.fmax, 
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # 增强数据增强的强度和种类
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=150)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=50)
        
    def apply_augmentation(self, waveform):
        # 随机音量调整 - 增强强度
        if torch.rand(1) > 0.2:
            volume_factor = 0.3 + torch.rand(1) * 1.4  # 0.3-1.7
            waveform = waveform * volume_factor

        # 添加随机噪声 - 增强强度
        if torch.rand(1) > 0.2:
            noise = torch.randn_like(waveform) * (0.005 + torch.rand(1) * 0.01)
            waveform = waveform + noise
        
        # 随机时间拉伸（模拟速度变化）
        if torch.rand(1) > 0.5:
            stretch_factor = 0.8 + torch.rand(1) * 0.4  # 0.8-1.2
            orig_len = waveform.shape[1]
            stretched_len = int(orig_len * stretch_factor)
            
            # 进行插值
            if stretched_len > orig_len:
                # 拉伸（变慢）
                indices = torch.linspace(0, orig_len-1, stretched_len)
                indices = indices.to(torch.int64)
                waveform_stretched = torch.zeros((1, stretched_len), device=waveform.device)
                for i in range(stretched_len):
                    src_idx = min(indices[i], orig_len-1)
                    waveform_stretched[0, i] = waveform[0, src_idx]
                # 截断到原始长度
                waveform = waveform_stretched[:, :orig_len]
            else:
                # 压缩（变快）
                indices = torch.linspace(0, stretched_len-1, orig_len)
                indices = indices.to(torch.int64)
                waveform_compressed = torch.zeros_like(waveform)
                for i in range(orig_len):
                    src_idx = min(indices[i], stretched_len-1)
                    waveform_compressed[0, i] = waveform[0, src_idx]
                waveform = waveform_compressed

        return waveform

    def __len__(self):
        return len(self.audio_files) 

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx] 
        label = self.labels[idx]  
        waveform, sr = torchaudio.load(audio_file) 

        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=True)  

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000) 
            waveform = resampler(waveform)  
        
        # 应用数据增强
        if self.augment:
            waveform = self.apply_augmentation(waveform)
        
        max_audio_length = 16000 * 5  
        if waveform.shape[1] < max_audio_length:  
            padding = max_audio_length - waveform.shape[1] 
            waveform = torch.nn.functional.pad(waveform, (0, padding))  
        else:
            waveform = waveform[:, :max_audio_length]  

        mel_spec = self.mel_spectrogram(waveform)  
        mel_spec_db = self.amplitude_to_db(mel_spec)  
        
        if mel_spec_db.shape[2] < self.target_length:  
            padding = self.target_length - mel_spec_db.shape[2]  
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding))  
        else:
            mel_spec_db = mel_spec_db[:, :, :self.target_length]  
        
        # 应用频谱增强
        if self.augment:
            mel_spec_db = self.freq_masking(mel_spec_db)
            mel_spec_db = self.time_masking(mel_spec_db)
        
        mel_spec_db = (mel_spec_db + 80) / 80  
        mel_spec_db = mel_spec_db.squeeze(0)  
        mel_spec_db = mel_spec_db.T  

        return mel_spec_db, label  

class ASTModelWrapper(nn.Module):
    def __init__(self, num_classes=1):
        super(ASTModelWrapper, self).__init__()  
        
        # 使用参数调整模型容量和过拟合风险
        self.ast = ASTModel(
            label_dim=num_classes,
            fstride=16,  # 使用标准步长
            tstride=16,  
            input_fdim=128,  
            input_tdim=512,  
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size='small224'  # 适中大小的模型，防止过拟合
        )
        
        # 降低dropout率，让模型更好地学习正样本特征
        self.dropout = nn.Dropout(0.3)  
        
    def forward(self, x):
        x = self.ast(x)
        x = self.dropout(x)
        return x

def find_best_threshold(val_loader, model, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for mel_spec, label in val_loader:
            mel_spec = mel_spec.to(device)
            label = label.float()
            outputs = model(mel_spec)
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(label.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 如果所有预测概率都很低，可能需要考虑更低的阈值
    print(f"预测概率范围: {np.min(all_probs):.8f} - {np.max(all_probs):.8f}")
    print(f"正样本平均概率: {np.mean(all_probs[all_labels==1]):.8f}")
    print(f"负样本平均概率: {np.mean(all_probs[all_labels==0]):.8f}")
    
    # 计算PR曲线和ROC曲线
    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    
    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # 查找最佳阈值 - 如果F1分数都很低，考虑使用能够平衡精确率和召回率的阈值
    if np.max(f1_scores) < 0.2:  # F1分数低于0.2，可能是正样本识别困难
        # 找到一个能给正样本更高分数的阈值
        best_idx = np.argmax(recall - fpr)  # 尝试最大化真正率和假正率的差异
        best_threshold = thresholds_roc[best_idx] if best_idx < len(thresholds_roc) else 0.5
    else:
        best_threshold = thresholds_pr[np.argmax(f1_scores)]
    
    # 确保阈值不为0
    if best_threshold <= 0:
        best_threshold = np.percentile(all_probs, 50)  # 使用中位数作为阈值
    
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    
    # 计算在最佳阈值下的指标
    predictions = (all_probs > best_threshold).astype(int)
    true_positives = np.sum((predictions == 1) & (all_labels == 1))
    false_positives = np.sum((predictions == 1) & (all_labels == 0))
    true_negatives = np.sum((predictions == 0) & (all_labels == 0))
    false_negatives = np.sum((predictions == 0) & (all_labels == 1))
    
    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    precision_score = true_positives / (true_positives + false_positives + 1e-8)
    
    metrics = {
        'threshold': best_threshold,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_score,
        'f1_score': np.max(f1_scores)
    }
    
    return metrics

def load_audio_files():
    audio_dir = 'audio'
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        os.makedirs(os.path.join(audio_dir, 'miao'))
        os.makedirs(os.path.join(audio_dir, 'other'))
        print(f"Created directory structure:")
        print(f"- {audio_dir}/miao  (for cat sounds)")
        print(f"- {audio_dir}/other (for non-cat sounds)")
        return [], []
    
    audio_files = []
    labels = []
    invalid_files = []
    
    # 从 miao 文件夹加载猫叫声（标签为1）
    miao_dir = os.path.join(audio_dir, 'miao')
    if os.path.exists(miao_dir):
        for file in glob.glob(os.path.join(miao_dir, '*.wav')):
            try:
                # 尝试加载文件以验证其格式
                waveform, sr = torchaudio.load(file)
                audio_files.append(file)
                labels.append(1)
            except Exception as e:
                invalid_files.append((file, str(e)))
    
    # 从 other 文件夹加载其他声音（标签为0）
    other_dir = os.path.join(audio_dir, 'other')
    if os.path.exists(other_dir):
        for file in glob.glob(os.path.join(other_dir, '*.wav')):
            try:
                # 尝试加载文件以验证其格式
                waveform, sr = torchaudio.load(file)
                audio_files.append(file)
                labels.append(0)
            except Exception as e:
                invalid_files.append((file, str(e)))
    
    if invalid_files:
        print("\nWarning: Some audio files could not be loaded:")
        for file, error in invalid_files:
            print(f"- {file}: {error}")
        print("\nThese files will be skipped during training.")
    
    return audio_files, labels

if __name__ == "__main__":  
    # 加载音频文件和标签
    audio_files, labels = load_audio_files()
    
    if not audio_files:
        print("No audio files found. Please add audio files to the 'audio' directory.")
        print("Name your cat sound files with '_miao' in the filename (e.g., 'cat_miao.wav')")
        print("Name other sound files without '_miao' (e.g., 'dog.wav')")
        exit()

    print(f"Found {len(audio_files)} audio files")
    print(f"Cat sounds (label 1): {sum(labels)}")
    print(f"Other sounds (label 0): {len(labels) - sum(labels)}")
    
    # 解决类别不平衡问题
    num_samples = len(labels)
    num_cat = sum(labels)
    num_other = num_samples - num_cat
    
    # 比例值修改：显著提高猫叫声样本的权重，用于解决负样本得分高于正样本的问题
    pos_weight = torch.tensor([20.0 * num_other / (num_cat + 1e-8)]).to(device)
    print(f"类别权重 - 猫叫声样本权重: {pos_weight.item():.2f}倍")
    
    # 分割训练集和验证集 - 使用分层抽样以保持类别比例
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels  
    )
    
    # 创建数据加载器，训练集使用数据增强
    train_dataset = AudioDataset(train_files, train_labels, augment=True)  
    val_dataset = AudioDataset(val_files, val_labels, augment=False)  
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 增大批次以提高稳定性
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  

    # 创建并训练模型
    model = ASTModelWrapper().to(device)  
    # 使用BCEWithLogitsLoss，它内置了sigmoid函数并且数值稳定性更好
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    
    # 使用更稳健的优化器设置 - 降低学习率以避免梯度爆炸
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  
    
    # 使用学习率调度器 - 但使用更温和的学习率变化
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,  # 降低最大学习率
        epochs=100,  # 增加训练轮数
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 增加预热阶段
        div_factor=10,  # 初始学习率 = max_lr/div_factor
        final_div_factor=100,  # 最终学习率 = max_lr/(div_factor*final_div_factor)
    )
    
    scaler = GradScaler()

    num_epochs = 100  # 增加训练轮数，让模型有更多时间学习
    best_val_metrics = {'f1_score': 0}
    patience = 10  # 增加早停耐心值，避免过早停止
    no_improve_epochs = 0
    
    print("\n开始训练...")
    print(f"使用设备: {device}")
    print(f"类别权重: 正样本={pos_weight.item():.2f}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print("模型配置:")
    print(f"- 学习率: {optimizer.param_groups[0]['lr']}")
    print(f"- 批次大小: 8")
    print(f"- 权重衰减: 0.01")
    print(f"- 使用数据增强: 是")
    print("训练进度:")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n周期 {epoch+1}/{num_epochs}")
        print("训练阶段:")
        
        train_pbar = tqdm(train_loader, desc=f'训练中', leave=True)
        for batch_idx, (mel_spec, label) in enumerate(train_pbar):
            mel_spec = mel_spec.to(device)
            label = label.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(mel_spec)
                loss = criterion(outputs, label)
            
            scaler.scale(loss).backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * mel_spec.size(0)
            
            # 使用sigmoid获取概率
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)
            
            train_pbar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"\nBatch {batch_idx+1}/{len(train_loader)}:")
                print(f"- Loss: {loss.item():.4f}")
                print(f"- Accuracy: {train_correct/train_total:.4f}")
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_metrics = None
        
        print("\n验证阶段:")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'验证中', leave=True)
            for mel_spec, label in val_pbar:
                mel_spec = mel_spec.to(device)
                label = label.float().unsqueeze(1).to(device)
                
                with autocast():
                    outputs = model(mel_spec)
                    loss = criterion(outputs, label)
                
                val_loss += loss.item() * mel_spec.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # 计算验证集上的详细指标
        val_metrics = find_best_threshold(val_loader, model, device)
        
        print(f'\n周期 {epoch+1} 总结:')
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
        print(f'验证损失: {val_loss:.4f}')
        print(f'验证指标:')
        print(f'- ROC AUC: {val_metrics["roc_auc"]:.4f}')
        print(f'- F1分数: {val_metrics["f1_score"]:.4f}')
        print(f'- 敏感度: {val_metrics["sensitivity"]:.4f}')
        print(f'- 特异度: {val_metrics["specificity"]:.4f}')
        print(f'- 精确度: {val_metrics["precision"]:.4f}')
        print(f'- 最佳阈值: {val_metrics["threshold"]:.8f}')

        # 保存最佳模型
        if val_metrics["f1_score"] > best_val_metrics["f1_score"]:
            best_val_metrics = val_metrics
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': val_metrics["threshold"],
                'val_metrics': val_metrics,
                'val_acc': (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2,  # 添加val_acc以兼容app.py
                'epoch': epoch
            }, 'models/ast_model_improved.pth')
            print(f'\n保存新的最佳模型，F1分数: {val_metrics["f1_score"]:.4f}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'\n已经 {no_improve_epochs} 个周期没有改进')
            
        # 早停
        if no_improve_epochs >= patience:
            print(f'\n{epoch+1} 个周期后提前停止训练')
            break

    print("\n训练完成!")
    print("\n最佳模型性能:")
    for metric, value in best_val_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    # 创建一个兼容app_improved.py的模型版本
    best_model_path = 'models/ast_model_improved.pth'
    model_data = torch.load(best_model_path)
    
    # 保存另一个版本供app.py使用
    torch.save(model_data, 'models/ast_model.pth')
    print("\n已创建兼容原始app.py的模型版本 models/ast_model.pth") 