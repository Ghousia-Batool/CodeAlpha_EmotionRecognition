# Emotion Recognition from Speech
This project recognizes human emotions from speech audio using Machine Learning.

## Project Description
The project uses the RAVDESS speech dataset to classify speech into different emotions. MFCC (Mel-Frequency Cepstral Coefficients) are extracted from the audio files and used as features for a Convolutional Neural Network (CNN).

## Emotions
The model recognizes 8 emotions:
* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

## Technologies Used
* Python
* TensorFlow
* Librosa
* NumPy
* CNN
* MFCC

## Dataset
The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.
The dataset contains 1,440 speech audio files from 24 actors.

## How It Works
1. Speech audio files are loaded using Librosa.
2. MFCC features are extracted from the audio.
3. The MFCC features are prepared for the CNN model.
4. The dataset is shuffled and divided into training and testing sets.
5. A CNN model is trained on the training data.
6. The trained model is evaluated on the test data.

## Model
The CNN model contains:
* Two convolutional layers
* Two max-pooling layers
* A flatten layer
* A dense layer
* An output layer for 8 emotions

## Result
The model achieved approximately **46.53% test accuracy**.

## How to Run
Install the required libraries and run:
```bash
python main.py
```

## Project Structure
```text
CodeAlpha_EmotionRecognition/
│
├── main.py
├── README.md
└── dataset/
```

## Internship Task
This project was completed as part of the CodeAlpha Machine Learning Internship.