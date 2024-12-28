# Hand-Numeric-Gesture-Recognition-System  

This project is a real-time hand gesture recognition system that uses computer vision and deep learning technologies to classify hand gestures from webcam input. The system leverages MediaPipe for hand landmark detection and a custom Convolutional Neural Network (CNN) for gesture classification.  

---

## Key Features  
- Real-time hand gesture recognition  
- Uses MediaPipe for hand landmark detection  
- Custom CNN model for gesture classification  
- Developed and trained on Google Colab  
- Supports multiple gesture categories  

---

## Project Structure  

### 1. Data Generation (`generate.py`)  
- Captures hand landmark images using webcam.  
- Processes and saves landmark images for training.  
- Supports different hand configurations (left/right, normal/flipped).  

### 2. Model Training (`Project_HGR.ipynb`)  
- Prepares and preprocesses the image dataset.  
- Builds a Convolutional Neural Network (CNN).  
- Trains and validates the gesture recognition model.  
- Saves the best-performing model.  

**CNN Architecture**:  
- Input  
- 4 layers of Conv2D and MaxPooling  
- Global Average Pooling  
- Dense  
- Dropout  
- Dense  
- Output  

### 3. Live Classification (`live_cam_test.py`)  
- Loads the pre-trained model.  
- Processes real-time webcam input.  
- Performs hand gesture recognition.  
- Displays prediction results.  

---

## Potential Improvements  
- Increase training dataset diversity.  
- Implement data augmentation.  
- Experiment with model architectures.  
- Add more gesture categories.  

---

## Limitations  
- Requires good lighting conditions.  
- Performance depends on training data quality.  
- Currently supports a limited number of gesture categories.  
