import os 
import torch  
import torchaudio  
from torch.utils.data import Dataset, DataLoader  
from sklearn.model_selection import train_test_split  
from torch import nn  
from tqdm import tqdm  
from src.models import ASTModel 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


class AudioDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files 
        self.labels = labels  


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

        
        mel_spec_db = (mel_spec_db + 80) / 80  

        
        mel_spec_db = mel_spec_db.squeeze(0)  
        mel_spec_db = mel_spec_db.T  

        return mel_spec_db, label  


class ASTModelWrapper(nn.Module):
    def __init__(self, num_classes=1):
        super(ASTModelWrapper, self).__init__()  
        
        self.ast = ASTModel(
            label_dim=num_classes,  
            fstride=10,  
            tstride=10,  
            input_fdim=128,  
            input_tdim=512,  
            imagenet_pretrain=True,  
            audioset_pretrain=True,  
            model_size='base384'  
        )
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.ast(x)  
        x = self.sigmoid(x)  
        return x  

if __name__ == "__main__":  
    
    audio_files = [
        'audio/000000.wav',
        'audio/000001.wav',
    ]
    labels = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  

    
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42  
    )
    
    train_dataset = AudioDataset(train_files, train_labels)  
    val_dataset = AudioDataset(val_files, val_labels)  
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)  

    
    model = ASTModelWrapper().to(device)  
    criterion = nn.BCELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  

    
    num_epochs = 30  
    for epoch in range(num_epochs):  
        model.train()  
        train_loss = 0.0  
        for mel_spec, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):  
            mel_spec = mel_spec.to(device)  
            label = label.float().unsqueeze(1).to(device)  

            optimizer.zero_grad()  
            outputs = model(mel_spec)  
            loss = criterion(outputs, label)  
            loss.backward()  
            optimizer.step()  
            train_loss += loss.item() * mel_spec.size(0)  
        train_loss /= len(train_loader.dataset)  

        
        model.eval()  
        val_loss = 0.0  
        correct = 0  
        with torch.no_grad():  
            for mel_spec, label in val_loader:  
                mel_spec = mel_spec.to(device)  
                label = label.float().unsqueeze(1).to(device)  

                outputs = model(mel_spec)  
                loss = criterion(outputs, label)  
                val_loss += loss.item() * mel_spec.size(0)  

                preds = (outputs > 0.5).float()  
                correct += (preds == label).sum().item()  
        val_loss /= len(val_loader.dataset)  
        val_acc = correct / len(val_loader.dataset)  

        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    
    os.makedirs('models', exist_ok=True)  
    torch.save(model.state_dict(), 'models/ast_model.pth')  
