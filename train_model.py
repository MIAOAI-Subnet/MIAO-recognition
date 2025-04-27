import os 
import torch  
import torchaudio  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from torch import nn  
from tqdm import tqdm  
from src.models import ASTModel 
import glob
from torch.cuda.amp import autocast, GradScaler
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
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=150)  # 增加遮蔽范围
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=50)  # 增加遮蔽范围
        
    def apply_augmentation(self, waveform):
        # 随机音量调整 - 增强强度
        if torch.rand(1) > 0.2:  # 增加应用概率
            volume_factor = 0.3 + torch.rand(1) * 1.4  # 扩大范围到0.3-1.7
            waveform = waveform * volume_factor

        # 添加随机噪声 - 增强强度
        if torch.rand(1) > 0.2:  # 增加应用概率
            noise = torch.randn_like(waveform) * (0.005 + torch.rand(1) * 0.01)  # 动态噪声强度
            waveform = waveform + noise
        
        # 随机时间拉伸（模拟速度变化）
        if torch.rand(1) > 0.5:
            stretch_factor = 0.8 + torch.rand(1) * 0.4  # 0.8-1.2的伸缩系数
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
        
        # 使用更小的模型以减少过拟合
        self.ast = ASTModel(
            label_dim=num_classes,  
            fstride=10,  # 降低步长以提高特征分辨率
            tstride=10,  
            input_fdim=128,  
            input_tdim=512,  
            imagenet_pretrain=False,
            audioset_pretrain=False,
            model_size='tiny224'  # 使用小型模型
        )
        
        # 增加Dropout以减少过拟合
        self.dropout = nn.Dropout(0.5)  # 增加dropout率
        # 不使用Sigmoid，由loss函数处理
        
    def forward(self, x):
        x = self.ast(x)
        x = self.dropout(x)
        return x  # 返回原始logits，不应用sigmoid

def find_best_threshold(val_loader, model, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for mel_spec, label in val_loader:
            mel_spec = mel_spec.to(device)
            label = label.float()
            outputs = model(mel_spec)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(label.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算PR曲线和ROC曲线
    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    
    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds_pr[np.argmax(f1_scores)]
    
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
    
    # 计算类别权重来处理数据不平衡
    num_samples = len(labels)
    num_cat = sum(labels)
    num_other = num_samples - num_cat
    # 显著增加猫叫声样本的权重，从原来的比例提高到10倍
    pos_weight = torch.tensor([10.0 * num_other / num_cat]).to(device)
    print(f"类别权重 - 猫叫声样本权重: {pos_weight.item():.2f}倍")
    
    # 分割训练集和验证集
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels  
    )
    
    # 创建数据加载器，训练集使用数据增强
    train_dataset = AudioDataset(train_files, train_labels, augment=True)  
    val_dataset = AudioDataset(val_files, val_labels, augment=False)  
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 减小批次大小
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  

    # 创建并训练模型
    model = ASTModelWrapper().to(device)  
    # 使用BCEWithLogitsLoss，它内置了sigmoid函数并且数值稳定性更好
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    
    # 使用更保守的优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)  
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=50,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # 增加预热阶段
        div_factor=25,  # 初始学习率 = max_lr/div_factor
        final_div_factor=1000,  # 最终学习率 = max_lr/(div_factor*final_div_factor)
    )
    
    scaler = GradScaler()

    num_epochs = 50  
    best_val_metrics = {'f1_score': 0}
    patience = 5
    no_improve_epochs = 0
    
    print("\nStarting training...")
    print(f"Using device: {device}")
    print(f"Class weights: positive={pos_weight.item():.2f}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("Model configuration:")
    print(f"- Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"- Batch size: 4")
    print(f"- Weight decay: 0.05")
    print(f"- Using data augmentation: Yes")
    print("Training progress:")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training phase:")
        
        train_pbar = tqdm(train_loader, desc=f'Training', leave=True)
        for batch_idx, (mel_spec, label) in enumerate(train_pbar):
            mel_spec = mel_spec.to(device)
            label = label.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(mel_spec)
                loss = criterion(outputs, label)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * mel_spec.size(0)
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
        
        print("\nValidation phase:")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validating', leave=True)
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
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Validation Metrics:')
        print(f'- ROC AUC: {val_metrics["roc_auc"]:.4f}')
        print(f'- F1 Score: {val_metrics["f1_score"]:.4f}')
        print(f'- Sensitivity: {val_metrics["sensitivity"]:.4f}')
        print(f'- Specificity: {val_metrics["specificity"]:.4f}')
        print(f'- Precision: {val_metrics["precision"]:.4f}')
        print(f'- Best Threshold: {val_metrics["threshold"]:.4f}')

        # 保存最佳模型
        if val_metrics["f1_score"] > best_val_metrics["f1_score"]:
            best_val_metrics = val_metrics
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': val_metrics["threshold"],
                'val_metrics': val_metrics,
                'epoch': epoch
            }, 'models/ast_model.pth')
            print(f'\nSaved new best model with F1 score: {val_metrics["f1_score"]:.4f}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'\nNo improvement for {no_improve_epochs} epochs')
            
        # 早停
        if no_improve_epochs >= patience:
            print(f'\nEarly stopping after {epoch+1} epochs')
            break

    print("\nTraining completed!")
    print("\nBest model performance:")
    for metric, value in best_val_metrics.items():
        print(f"{metric}: {value:.4f}")
