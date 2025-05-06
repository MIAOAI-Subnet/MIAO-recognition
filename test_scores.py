import os
import torch
import torchaudio
import glob
from train_model import ASTModelWrapper
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
model_path = 'models/ast_model.pth'
alt_model_path = 'models/cat_detector.pth'

# 尝试加载不同的模型
def load_model():
    # 首先尝试加载AST模型
    model = ASTModelWrapper().to(device)
    model.eval()
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载AST模型: {model_path}")
            return model, "AST"
        except Exception as e:
            print(f"加载AST模型失败: {str(e)}")
    
    # 如果AST模型加载失败，尝试加载备选模型
    if os.path.exists(alt_model_path):
        try:
            from torch import nn
            
            # 创建一个简单的分类器模型作为备选
            class SimpleCatClassifier(nn.Module):
                def __init__(self):
                    super(SimpleCatClassifier, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(128 * 16 * 32, 128),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(128, 1)
                    )
                    
                def forward(self, x):
                    x = x.unsqueeze(1)  # 添加通道维度
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            alt_model = SimpleCatClassifier().to(device)
            alt_checkpoint = torch.load(alt_model_path)
            alt_model.load_state_dict(alt_checkpoint)
            alt_model.eval()
            print(f"成功加载备选模型: {alt_model_path}")
            return alt_model, "Simple"
        except Exception as e:
            print(f"加载备选模型失败: {str(e)}")
    
    print("所有模型都加载失败，使用未经训练的默认模型")
    return model, "Untrained"

# 加载音频预处理函数
def preprocess_audio(waveform, sample_rate):
    # 如果采样率不是16kHz，则重采样
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 将多通道音频转换为单通道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 如果音频小于5秒，则填充；如果大于5秒，则截断
    max_audio_length = 16000 * 5  # 5秒
    if waveform.shape[1] < max_audio_length:
        padding = max_audio_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :max_audio_length]

    # 定义梅尔频谱转换器
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=128,
        f_min=50,
        f_max=8000,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    # 生成梅尔频谱
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)

    # 确保梅尔频谱的时间帧数为512
    target_length = 512
    if mel_spec_db.shape[2] < target_length:
        padding = target_length - mel_spec_db.shape[2]
        mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding))
    else:
        mel_spec_db = mel_spec_db[:, :, :target_length]

    # 归一化到[0, 1]
    mel_spec_db = (mel_spec_db + 80) / 80

    # 调整形状为[512, 128]
    mel_spec_db = mel_spec_db.squeeze(0)
    mel_spec_db = mel_spec_db.T

    return mel_spec_db

def analyze_scores():
    # 加载模型
    model, model_type = load_model()
    
    # 获取所有音频文件
    miao_files = glob.glob(os.path.join('audio', 'miao', '*.WAV')) + glob.glob(os.path.join('audio', 'miao', '*.wav'))
    other_files = glob.glob(os.path.join('audio', 'other', '*.WAV')) + glob.glob(os.path.join('audio', 'other', '*.wav'))
    
    print(f"猫叫声样本数量: {len(miao_files)}")
    print(f"非猫叫声样本数量: {len(other_files)}")
    
    # 限制每类样本的数量以加快分析速度
    max_samples = 50
    if len(miao_files) > max_samples:
        miao_files = miao_files[:max_samples]
    if len(other_files) > max_samples:
        other_files = other_files[:max_samples]
    
    miao_scores = []
    other_scores = []
    
    # 分析猫叫声样本
    print("分析猫叫声样本...")
    for file in tqdm(miao_files):
        try:
            waveform, sample_rate = torchaudio.load(file)
            mel_spec = preprocess_audio(waveform, sample_rate)
            mel_spec = mel_spec.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(mel_spec)
                probability = torch.sigmoid(outputs).item()
                miao_scores.append(probability)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 分析非猫叫声样本
    print("分析非猫叫声样本...")
    for file in tqdm(other_files):
        try:
            waveform, sample_rate = torchaudio.load(file)
            mel_spec = preprocess_audio(waveform, sample_rate)
            mel_spec = mel_spec.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(mel_spec)
                probability = torch.sigmoid(outputs).item()
                other_scores.append(probability)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
    
    # 计算统计信息
    miao_avg = np.mean(miao_scores) if miao_scores else 0
    other_avg = np.mean(other_scores) if other_scores else 0
    
    print(f"\n分析结果 (模型类型: {model_type}):")
    print(f"猫叫声样本平均分数: {miao_avg:.6f}")
    print(f"非猫叫声样本平均分数: {other_avg:.6f}")
    
    if miao_avg <= other_avg:
        print("\n警告: 非猫叫声样本的平均分数高于或等于猫叫声样本!")
        print("这表明模型可能存在问题，需要调整阈值或重新训练。")
        
        # 计算合适的阈值
        all_scores = miao_scores + other_scores
        all_labels = [1] * len(miao_scores) + [0] * len(other_scores)
        
        best_threshold = 0.5
        best_acc = 0
        
        # 简单的阈值搜索
        for threshold in np.arange(0, 1, 0.01):
            predictions = [1 if s >= threshold else 0 for s in all_scores]
            accuracy = sum(p == l for p, l in zip(predictions, all_labels)) / len(all_labels)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_threshold = threshold
        
        print(f"\n建议的阈值设置:")
        print(f"最佳阈值: {best_threshold:.6f} (准确率: {best_acc:.4f})")
        print(f"高阈值: {best_threshold * 0.8:.6f}")
        print(f"中阈值: {best_threshold * 0.5:.6f}")
        print(f"低阈值: {best_threshold * 0.3:.6f}")
    else:
        print("\n模型分数正常: 猫叫声样本的平均分数高于非猫叫声样本。")
    
    # 绘制分数分布图
    plt.figure(figsize=(10, 6))
    plt.hist(miao_scores, bins=20, alpha=0.5, label='猫叫声')
    plt.hist(other_scores, bins=20, alpha=0.5, label='非猫叫声')
    plt.xlabel('预测分数')
    plt.ylabel('样本数量')
    plt.title(f'猫叫声检测分数分布 (模型: {model_type})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    print("\n分数分布图已保存为 'score_distribution.png'")

if __name__ == "__main__":
    analyze_scores() 