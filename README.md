
# Sign Language Detection

Unlock the power of communication with our Sign Language Detection project! This innovative project uses OpenCV and the MediaPipe Hand module to recognize American Sign Language (ASL) letters. With a dataset containing 100 samples for each of the 26 letters, the project is ready to go for ASL recognition. Whether you're directly detecting ASL or want to customize it for another alphabet, we've got you covered!

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Dataset Creation](#dataset-creation)
  - [Model Training](#model-training)
  - [Running the Demo](#running-the-demo)
- [Important Notes](#important-notes)

## Requirements

- opencv-python==4.7.0.68
- mediapipe==0.9.0.1
- scikit-learn==1.2.0

## Installation

Install the requirements using:
```bash 
pip install opencv-python mediapipe numpy tensorflow 
```

## Usage

To start recognizing ASL letters, simply run the `camera_demo.py` file: 
```bash 
python camera_demo.py
```

If you want to train the model for a different alphabet, follow these steps:

- Collect data by running `collect_imgs.py`: 
    ```bash 
    python collect_imgs.py
    ```

- Process the collected images into a dataset using `create_dataset.py`:
    ```bash 
    python create_dataset.py
    ``` 

- Train your model with the new dataset using `train_model.py`: 
    ```bash
    python train_model.py
    ```

- Run the demo with the trained model using `camera_demo.py`. Ensure the model path in `camera_demo.py` is correctly set.

## Important Notes

Before running `camera_demo.py`, make sure the model path is correctly set to the trained model in the script.

## Contributing

We welcome contributions! If you want to contribute, please send a pull request or open an issue. Your feedback and contributions are highly valued.
