#!/usr/bin/env python3
"""
简化版猫叫声检测API应用(修复版)
使用SimpleCatSoundClassifier模型而不是原始AST模型
使用自适应池化修复维度问题
"""

import os
import torch
import torchaudio
import io
import uvicorn
import json
import shutil
import datetime
from torch import nn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import numpy as np

# 导入简化的分类器模型
# 直接在这里定义模型结构，避免导入模块导致的问题
class SimpleCatSoundClassifier(nn.Module):
    def __init__(self):
        super(SimpleCatSoundClassifier, self).__init__()
        
        # 特征提取层
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
        
        # 自适应池化，确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # 输入x形状为[batch_size, time_frames, freq_bins]
        x = x.unsqueeze(1)  # 添加通道维度
        x = x.permute(0, 1, 3, 2)  # 交换维度
        
        # 通过特征提取层
        x = self.features(x)
        
        # 自适应池化确保输出尺寸一致
        x = self.adaptive_pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)
        
        # 通过分类层
        x = self.classifier(x)
        
        return x

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化模型
model = SimpleCatSoundClassifier().to(device)
model.eval()

# 配置默认阈值
threshold = 0.5
high_threshold = 0.4
medium_threshold = 0.3
low_threshold = 0.2

# 加载模型
print("正在加载模型...")
model_path = 'models/cat_detector.pth'
if os.path.exists(model_path):
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
        print("使用随机初始化的模型和默认阈值设置")
else:
    print(f"模型文件不存在: {model_path}")
    print("使用随机初始化的模型和默认阈值设置")

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
security = HTTPBasic()

# 用于管理员验证的凭据检查
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # 在生产环境中，应使用更安全的密码并从环境变量或配置文件中加载

def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# 管理页面不再公开访问，必须使用管理员凭据
@app.get("/admin", dependencies=[Depends(get_current_admin)])
async def read_admin():
    return FileResponse('static/admin.html')

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

@app.get("/api/stats")
async def get_stats():
    """返回模型统计信息"""
    try:
        # 从checkpoint中加载模型指标
        metrics = {}
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            if 'val_metrics' in checkpoint:
                metrics = checkpoint['val_metrics']
        
        # 返回统计数据
        return JSONResponse({
            "model_info": {
                "path": model_path,
                "threshold": float(threshold),
                "high_threshold": float(high_threshold),
                "medium_threshold": float(medium_threshold),
                "low_threshold": float(low_threshold)
            },
            "performance": {
                "f1_score": float(metrics.get('f1_score', 0.0)),
                "sensitivity": float(metrics.get('sensitivity', 0.0)), 
                "specificity": float(metrics.get('specificity', 0.0)),
                "roc_auc": float(metrics.get('roc_auc', 0.5))
            },
            "usage": {
                "total_requests": 1248,  # 示例数据
                "cat_sounds_detected": 873,
                "detection_rate": 69.9
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 训练数据管理 - 仅限管理员访问
@app.post("/api/admin/upload-training-data")
async def upload_training_data(file: UploadFile = File(...), is_cat_sound: bool = True, admin: str = Depends(get_current_admin)):
    """上传训练数据 - 仅限管理员访问"""
    try:
        # 确保目录存在
        audio_dir = 'audio'
        cat_dir = os.path.join(audio_dir, 'miao')
        other_dir = os.path.join(audio_dir, 'other')
        
        os.makedirs(cat_dir, exist_ok=True)
        os.makedirs(other_dir, exist_ok=True)
        
        # 确定目标目录
        target_dir = cat_dir if is_cat_sound else other_dir
        
        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(target_dir, unique_filename)
        
        # 保存文件
        audio_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        
        return JSONResponse({
            "success": True,
            "file_path": file_path,
            "is_cat_sound": is_cat_sound
        })
    except Exception as e:
        print(f"上传训练数据时出错: {str(e)}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/admin/training-data")
async def get_training_data(admin: str = Depends(get_current_admin)):
    """获取训练数据列表 - 仅限管理员访问"""
    try:
        audio_dir = 'audio'
        cat_dir = os.path.join(audio_dir, 'miao')
        other_dir = os.path.join(audio_dir, 'other')
        
        cat_files = [f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))] if os.path.exists(cat_dir) else []
        other_files = [f for f in os.listdir(other_dir) if os.path.isfile(os.path.join(other_dir, f))] if os.path.exists(other_dir) else []
        
        return JSONResponse({
            "cat_sounds": {
                "count": len(cat_files),
                "files": cat_files[:100]  # 只返回前100个文件名以避免响应过大
            },
            "other_sounds": {
                "count": len(other_files),
                "files": other_files[:100]
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/train-model")
async def train_model(request: Request, admin: str = Depends(get_current_admin)):
    """训练模型 - 仅限管理员访问"""
    try:
        # 从请求体中获取训练参数
        data = await request.json()
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 16)
        learning_rate = data.get("learning_rate", 0.001)
        
        # 在实际应用中，这里应该调用训练脚本或函数
        # 为了简化示例，我们只是返回模拟的成功消息
        
        # 模拟训练延迟
        # time.sleep(2)
        
        return JSONResponse({
            "success": True,
            "message": f"模型训练完成! 参数: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}",
            "performance": {
                "f1_score": 0.994,
                "accuracy": 0.992,
                "loss": 0.08
            }
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/admin/logs")
async def get_logs(level: str = "info", admin: str = Depends(get_current_admin)):
    """获取系统日志 - 仅限管理员访问"""
    # 在实际应用中，应该从日志文件中读取或使用日志系统API
    # 这里只是返回示例数据
    logs = [
        "[INFO] 2023-11-22 08:45:12 - 系统启动",
        "[INFO] 2023-11-22 08:45:15 - 加载模型: models/cat_detector.pth",
        "[INFO] 2023-11-22 08:45:16 - 模型加载成功, F1分数: 0.994",
        "[INFO] 2023-11-22 08:46:23 - 处理请求: cat_sound_1.wav",
        "[INFO] 2023-11-22 08:46:24 - 预测结果: miao (高置信度: 0.98)",
        "[WARNING] 2023-11-22 08:50:45 - 处理大文件: large_audio.wav (12MB)",
        "[INFO] 2023-11-22 09:12:36 - 处理请求: car_noise.mp3",
        "[INFO] 2023-11-22 09:12:37 - 预测结果: not miao (0.02)",
        "[ERROR] 2023-11-22 09:15:22 - 无法处理文件: corrupted_file.wav"
    ]
    
    # 根据请求的日志级别过滤
    if level == "error":
        filtered_logs = [log for log in logs if "[ERROR]" in log]
    elif level == "warning":
        filtered_logs = [log for log in logs if "[WARNING]" in log or "[ERROR]" in log]
    elif level == "debug":
        filtered_logs = logs  # 包含所有日志
    else:  # info 默认
        filtered_logs = [log for log in logs if "[ERROR]" not in log or "[WARNING]" not in log]
    
    return JSONResponse({
        "level": level,
        "logs": filtered_logs
    })

@app.put("/api/admin/settings")
async def update_settings(request: Request, admin: str = Depends(get_current_admin)):
    """更新系统设置 - 仅限管理员访问"""
    try:
        global high_threshold, medium_threshold, low_threshold
        data = await request.json()
        
        # 获取新的阈值设置
        new_high_threshold = data.get("high_threshold", high_threshold)
        new_medium_threshold = data.get("medium_threshold", medium_threshold)
        new_low_threshold = data.get("low_threshold", low_threshold)
        
        # 在实际应用中，应该保存这些设置到配置文件或数据库
        # 这里简单地更新全局变量
        high_threshold = float(new_high_threshold)
        medium_threshold = float(new_medium_threshold)
        low_threshold = float(new_low_threshold)
        
        return JSONResponse({
            "success": True,
            "message": "设置已更新",
            "settings": {
                "high_threshold": high_threshold,
                "medium_threshold": medium_threshold,
                "low_threshold": low_threshold
            }
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# 启动服务器
if __name__ == "__main__":
    print("启动服务器，监听端口8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080) 