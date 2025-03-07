<div align="center">

# **WOOF AI** <!-- omit in toc -->

### Bridging Pet Tech and Blockchain Innovation <!-- omit in toc -->
![hero](./asset/offline.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

#  Introduction

“WOOF” is the expression of a dog’s bark. We are developing an AI project designed to analyze and mimic canine behavior. By collecting dog barks uploaded by users, we aim to build a large-scale model to analyze dogs emotions and predict their behaviors, fostering a closer emotional bond between humans and dogs. We hope to encourage people around the world to form deeper connections with dogs. In the future, we look forward to expanding Woof AI's capabilities to provide more personalized interactions, enhance pet care, and revolutionize how we communicate with our furry companions. Woof woof!


# Dog Barking Sound Recognition System
This system is designed to demonstrate how to accurately recognize dog barking sounds in audio data using advanced audio recognition techniques. The system adopts the most cutting-edge algorithms in the field of deep learning and is optimized for the characteristics of dog barking, which can efficiently and accurately recognize dog barking in complex environments.


### Core algorithm
### Audio Feature Extraction
VGGish + STFT - First, the audio data is feature extracted using the VGGish model, which is a CNN model pre-trained on large-scale audio datasets that extracts high-dimensional Mel-spectrogram features of the audio and captures the spectral information of the audio. - The raw audio signal is then converted into a spectrogram using a Short-Time Fourier Transform (STFT). These spectrograms are used as inputs to the Transformer model, which can fully display the time and frequency domain features of the audio signal and provide important input data for subsequent audio classification.
### Audio Classification
Audio Spectrogram Transformer (AST) for Audio Classification - AST is a specialized Transformer model designed for audio data, which employs Self-Attention mechanism to process spectrograms. This Self-Attention mechanism can efficiently capture long time-series dependencies and global features in audio signals, which has obvious advantages over traditional CNN + LSTM when processing complex and long time-series data. - In the task of classifying the sound of a barking dog, the AST model can accurately capture features such as rhythm, pitch, and mutation points in the audio, and thus effectively distinguish the dog's barking from other sounds (e.g., ambient noise or other animal barks).

### Classification of sound events
Barking vs. Non-Barking - After feature extraction and global modeling, the features generated by the Transformer model are fed into a binary classification model, which ultimately classifies the audio signal as “barking” or “non-barking”. Due to Transformer's powerful global modeling capabilities, the system is able to recognize barking sounds consistently and efficiently in complex audio environments.

![image](https://github.com/user-attachments/assets/6c257a07-234f-4238-9be0-174c727d285c)



### Global dependency modeling capabilities
Transformer's self-attention mechanism allows it to capture long-range dependencies across each time step of the input audio. This is important for audio signals such as the sound of a barking dog, which exhibits specific rhythmic, pitch and timing variations. Transformer is able to capture these long temporal features more efficiently and accurately than traditional LSTM models.

### Efficient training and reasoning speed
The Transformer model leverages powerful parallel computing capabilities, significantly boosting training efficiency. Unlike traditional recurrent neural networks like LSTM, the Transformer processes all time steps simultaneously rather than sequentially, resulting in faster training and more efficient inference. This parallel processing allows the Transformer to handle large-scale datasets with reduced training time, making it an ideal choice for tasks that require high computational power

### Excellent audio classification performance
Unlike CNN + LSTM, Transformer is able to capture both local and global features of audio through the self-attention mechanism, and is able to effectively handle long audio sequences. It is particularly good at classifying complex audio events (e.g., dog barking), and can accurately recognize and distinguish between different types of sound events.

### Advantages of pre-trained models
Pre-trained models based on Audio Spectrogram Transformer (AST) have achieved excellent results in several audio classification tasks. Through migration learning, the AST model can be quickly adapted to the dog barking sound recognition task to further improve the accuracy.

### Robustness
Transformer's ability to model global information and long temporal dependencies makes its recognition performance more stable in complex environments. In noisy backgrounds (e.g., traffic noise, environmental noise), the Transformer model still maintains high accuracy, while the traditional LSTM model is often prone to interference, leading to a decrease in recognition accuracy.
er recommendations.

<img width="515" alt="_20241120191951" src="https://github.com/user-attachments/assets/626e0235-f2e3-4056-8c16-480e5ff06930">



### Model Performance Comparison

![image/png](https://cdn-uploads.huggingface.co/production/uploads/673d82cc1898a8cd00977d97/tnTtxaKkR7_mH_Xbo_nS5.png)

