# Cat Sound Recognition System Deployment Guide

## System Requirements

- Python 3.8+ 
- CUDA (for GPU acceleration)
- Minimum 2GB RAM
- Sufficient disk space for model files (approximately 200MB)

## Installation Steps

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Check Model Files**

   Ensure the following model file exists in the models directory:
   
   ```
   models/cat_detector.pth
   ```

   If the file is missing, you need to run the training script first:
   
   ```bash
   python train_model.py
   ```

3. **Check Static Files**

   Ensure the static directory contains frontend files, including index.html and related CSS/JS files.

## Local Testing

1. **Start the Service**

   Method 1: Using the startup script
   ```bash
   ./run_server.sh
   ```
   
   Method 2: Running the Python file directly
   ```bash
   python app.py
   ```

2. **Test the Service**

   After the server starts, you can access it through a browser:
   
   ```
   http://localhost:5000
   ```
   
   Or test the API using curl:
   
   ```bash
   curl -F "file=@/path/to/audio.wav" http://localhost:5000/predict
   ```

## Server Deployment

### Method 1: Direct Execution (suitable for temporary use)

1. Run in the background using nohup, so it continues running even after terminal closes:

   ```bash
   nohup python app.py > cat_service.log 2>&1 &
   ```

2. Check the process ID:

   ```bash
   ps aux | grep app.py
   ```

3. To stop the service:

   ```bash
   kill <process_ID>
   ```

### Method 2: Using systemd service (suitable for long-term operation)

1. Create a systemd service file:

   ```bash
   sudo nano /etc/systemd/system/cat-detector.service
   ```

2. Add the following content (replace paths with your actual paths):

   ```
   [Unit]
   Description=Cat Sound Detection Service
   After=network.target

   [Service]
   User=<your_username>
   WorkingDirectory=/path/to/MIAO-recognition
   ExecStart=/usr/bin/python3 app.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:

   ```bash
   sudo systemctl enable cat-detector.service
   sudo systemctl start cat-detector.service
   ```

4. Check service status:

   ```bash
   sudo systemctl status cat-detector.service
   ```

5. View logs:

   ```bash
   sudo journalctl -u cat-detector.service -f
   ```

### Method 3: Using Docker (suitable for cross-platform deployment)

1. Create a Dockerfile:

   ```Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 5000
   
   CMD ["python", "app.py"]
   ```

2. Build the Docker image:

   ```bash
   docker build -t cat-detector .
   ```

3. Run the Docker container:

   ```bash
   docker run -d -p 5000:5000 cat-detector
   ```

## Security and Performance Considerations

1. **Security Considerations**:
   - Enable HTTPS in production environments
   - Consider adding API keys or authentication mechanisms
   - Limit upload file sizes

2. **Performance Optimization**:
   - For high concurrency scenarios, consider using Gunicorn with Uvicorn as WSGI server
   - Use load balancing to handle multiple requests
   - Monitor CPU and GPU usage, upgrade hardware if necessary

3. **Availability**:
   - Set up monitoring and alert systems to track service status
   - Regularly backup model files
   - Implement automatic restart mechanisms

## Training Data Management

The system now includes a training data upload interface that allows:

1. **Uploading Training Audio**:
   - Users can upload audio files through the Training Data tab
   - Files can be labeled as either "Cat Sound" or "Other Sound"
   - Uploaded files are stored in the appropriate directories:
     - `audio/miao/` for cat sounds
     - `audio/other/` for non-cat sounds

2. **Training the Model**:
   - After uploading sufficient training data, the model can be retrained
   - The training process requires running:
     ```bash
     python train_model.py
     ```
   - Training time depends on the amount of data and hardware capabilities
   - New models are saved to `models/cat_detector.pth`

3. **Best Practices for Training Data**:
   - Provide diverse samples of cat sounds (different cats, environments)
   - Include various non-cat sounds that might be confused with cat sounds
   - Maintain a balanced dataset (similar number of positive and negative samples)
   - Aim for at least 50 samples in each category for reasonable performance

## Troubleshooting

1. **Service Startup Failure**:
   - Check CUDA and PyTorch version compatibility
   - Verify model file integrity
   - Check if port 5000 is already in use

2. **Abnormal Prediction Results**:
   - Check if audio format is supported (WAV format is recommended)
   - Verify threshold settings are reasonable
   - Try reloading the model

3. **Performance Issues**:
   - If inference is slow, consider using GPU or a faster CPU
   - Check for memory leaks
   - Optimize audio preprocessing workflow

## Updating the Model

When you need to update the model:

1. Ensure the new model file is named `cat_detector.pth`
2. Backup the old model
3. Replace with the new model
4. Restart the service 