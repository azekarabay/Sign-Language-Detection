Sign Language Detection
This project performs American Sign Language (ASL) detection using OpenCV and MediaPipe. The dataset contains 26 different letters, each with 100 samples. Running the camera_demo.py file is sufficient for direct detection.

Table of Contents
Requirements
Installation
Usage
Data Collection
Dataset Creation
Model Training
Running the Demo
Important Notes
Requirements
You will need the following libraries to run this project:

Python 3.x
OpenCV
MediaPipe
NumPy
TensorFlow/Keras
Install the requirements using:

bash
Kodu kopyala
pip install opencv-python mediapipe numpy tensorflow
Installation
Clone the project files to your computer:

bash
Kodu kopyala
git clone https://github.com/username/sign-language-detection.git
cd sign-language-detection
Usage
1. Data Collection
If you want to work with a new alphabet or different data, you can use collect_imgs.py to collect data:

bash
Kodu kopyala
python collect_imgs.py
2. Dataset Creation
Create a dataset from the collected data using create;_dataset.py:

bash
Kodu kopyala
python create_dataset.py
3. Model Training
Train the model with the dataset using train_model.py:

bash
Kodu kopyala
python train_model.py
4. Running the Demo
Run real-time detection using the trained model with camera_demo.py:

bash
Kodu kopyala
python camerademo.py
Important Notes
Before running camera_demo.py, ensure that the trained and saved model's path is correctly set in the camerademo.py file.