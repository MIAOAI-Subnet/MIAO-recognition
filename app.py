import os  
import torch  
import torchaudio  
from torch import nn  
from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import JSONResponse, FileResponse  
from fastapi.staticfiles import StaticFiles 
from src.models import ASTModel  
import uvicorn  
import io  
from train_model import ASTModelWrapper
# Configuring Devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Detect if there is a GPU available, if so use the GPU, otherwise use the CPU

# Initialize the model and load the trained weights
model = ASTModelWrapper().to(device)
model.load_state_dict(torch.load('models/ast_model.pth', map_location=device))
model.eval()

# Define audio preprocessing functions

def preprocess_audio(waveform, sample_rate):
    # Resample if sampling rate is not 16kHz

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Convert multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # If the audio is less than 5 seconds long, it is padded; if it is longer, it is truncated
    max_audio_length = 16000 * 5  # 5 seconds of audio
    if waveform.shape[1] < max_audio_length:
        padding = max_audio_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :max_audio_length]

    # Define Mel spectrogram converter (same as training)
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

    # Generate Mel spectrum
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)

    # Make sure the number of time frames for the Mel spectrogram is 512
    target_length = 512
    if mel_spec_db.shape[2] < target_length:
        padding = target_length - mel_spec_db.shape[2]
        mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, padding))
    else:
        mel_spec_db = mel_spec_db[:, :, :target_length]

    # Normalized to [0, 1]
    mel_spec_db = (mel_spec_db + 80) / 80

    # Reshape to [512, 128]
    mel_spec_db = mel_spec_db.squeeze(0)
    mel_spec_db = mel_spec_db.T

    return mel_spec_db


app = FastAPI()

# Mount the static file directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Prediction interface
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_bytes = await file.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Reading audio files using torchaudio
        waveform, sample_rate = torchaudio.load(audio_stream)

        # Preprocessing Audio
        mel_spec = preprocess_audio(waveform, sample_rate)
        mel_spec = mel_spec.unsqueeze(0).to(device)  # Adding the batch dimension

        # Making predictions
        with torch.no_grad():
            output = model(mel_spec)
            probability = output.item()

        # Determine whether it is a dog barking
        if probability > 0.9997:
            result = "dog bark"
        else:
            result = "not dog bark"

        return JSONResponse({"result": result, "probability": probability})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
