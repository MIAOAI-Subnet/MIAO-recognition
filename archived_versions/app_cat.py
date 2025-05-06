#!/usr/bin/env python3
"""
简化版猫叫声检测API应用
使用SimpleCatSoundClassifier模型而不是原始AST模型
"""

import os
import torch
import torchaudio
import io
import uvicorn
from torch import nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# 导入简化的分类器模型
from train_cat_detector import SimpleCatSoundClassifier

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化模型
model = SimpleCatSoundClassifier().to(device)
model.eval()

# 加载模型
print("正在加载模型...")
model_path = 'models/cat_detector.pth'
if not os.path.exists(model_path):
    model_path = 'models/ast_model.pth'  # 尝试使用兼容版本

try:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 获取阈值和指标
    threshold = checkpoint.get('best_threshold', 0.5)
    
    # 设置分级阈值
    if threshold < 0.01:
        # 如果阈值太低，使用自定义阈值
        high_threshold = 0.01
        medium_threshold = 0.005
        low_threshold = 0.001
        print("原始阈值过低，使用备用阈值")
    else:
        # 使用相对阈值
        high_threshold = threshold * 0.8
        medium_threshold = threshold * 0.5
        low_threshold = threshold * 0.3
    
    print(f"模型加载成功: {model_path}")
    print(f"阈值设置:")
    print(f"- 原始阈值: {threshold:.6f}")
    print(f"- 高置信度阈值: {high_threshold:.6f}")
    print(f"- 中置信度阈值: {medium_threshold:.6f}")
    print(f"- 低置信度阈值: {low_threshold:.6f}")
    
    # 显示模型性能指标
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"模型性能:")
        print(f"- F1分数: {metrics.get('f1_score', 0.0):.4f}")
        print(f"- 敏感度: {metrics.get('sensitivity', 0.0):.4f}")
        print(f"- 特异度: {metrics.get('specificity', 0.0):.4f}")
        print(f"- ROC AUC: {metrics.get('roc_auc', 0.5):.4f}")
    
except Exception as e:
    print(f"加载模型出错: {str(e)}")
    # 使用默认阈值
    threshold = 0.5
    high_threshold = 0.4
    medium_threshold = 0.3
    low_threshold = 0.2
    print("使用默认阈值设置")

# 音频预处理函数
def preprocess_audio(waveform, sample_rate):
    # 重采样到16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 标准化长度到5秒
    max_length = 16000 * 5
    if waveform.shape[1] < max_length:
        padding = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :max_length]
    
    # 计算梅尔频谱图
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=20,
        f_max=8000
    )(waveform)
    
    # 转换为分贝
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    mel_db = amplitude_to_db(mel_spec)
    
    # 标准化到[0,1]范围
    mel_db = (mel_db + 80) / 80
    
    # 确保尺寸一致
    if mel_db.shape[2] < 256:
        padding = 256 - mel_db.shape[2]
        mel_db = torch.nn.functional.pad(mel_db, (0, padding))
    else:
        mel_db = mel_db[:, :, :256]
    
    # 调整格式
    mel_db = mel_db.squeeze(0)
    
    return mel_db

# 创建FastAPI应用
app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取上传的音频文件
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_stream)
        
        # 预处理音频
        mel_spec = preprocess_audio(waveform, sample_rate)
        mel_spec = mel_spec.unsqueeze(0).to(device)  # 添加批次维度
        
        # 预测
        with torch.no_grad():
            output = model(mel_spec)
            probability = torch.sigmoid(output).item()
        
        # 记录预测结果
        print(f"预测文件: {file.filename}")
        print(f"预测概率: {probability:.6f}")
        
        # 根据阈值确定结果
        if probability > high_threshold:
            result = "miao"
            confidence = "高"
            detail = "这是猫叫声 (置信度高)"
        elif probability > medium_threshold:
            result = "miao"
            confidence = "中"
            detail = "这可能是猫叫声 (置信度中)"
        elif probability > low_threshold:
            result = "可能是miao"
            confidence = "低"
            detail = "这可能是猫叫声 (置信度低)"
        else:
            result = "not miao"
            confidence = "无"
            detail = "这不是猫叫声"
        
        print(f"结果: {detail}")
        
        # 返回结果
        return JSONResponse({
            "result": result,
            "probability": probability,
            "confidence_level": confidence,
            "detail": detail,
            "thresholds": {
                "high": float(high_threshold),
                "medium": float(medium_threshold),
                "low": float(low_threshold),
                "original": float(threshold)
            }
        })
        
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("启动服务器，监听端口5000...")
    uvicorn.run(app, host="0.0.0.0", port=5000) 