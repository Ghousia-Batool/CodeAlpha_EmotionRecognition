import os
import librosa
import numpy as np
from tensorflow import keras

Sequential = keras.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Flatten = keras.layers.Flatten
Dense = keras.layers.Dense
Input = keras.layers.Input
to_categorical = keras.utils.to_categorical


DATASET_PATH = "dataset/Audio_Speech_Actors_01-24_16k"


emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


X = []
y = []


for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            parts = file.split("-")
            emotion_code = parts[2]

            if emotion_code in emotions:

                audio, sample_rate = librosa.load(
                    file_path,
                    duration=3
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=40
                )

                # Make every MFCC the same size
                if mfcc.shape[1] < 130:
                    mfcc = np.pad(
                        mfcc,
                        ((0, 0), (0, 130 - mfcc.shape[1]))
                    )
                else:
                    mfcc = mfcc[:, :130]

                X.append(mfcc)
                y.append(emotions[emotion_code])


X = np.array(X)
X = X[..., np.newaxis]

print("Total audio files:", len(X))
print("MFCC shape:", X.shape)


emotion_names = sorted(set(y))

emotion_to_number = {
    emotion: i
    for i, emotion in enumerate(emotion_names)
}


y_numbers = np.array([
    emotion_to_number[emotion]
    for emotion in y
])


y_categorical = to_categorical(
    y_numbers,
    num_classes=len(emotion_names)
)


# Shuffle dataset

np.random.seed(42)

indices = np.random.permutation(len(X))

X = X[indices]
y_categorical = y_categorical[indices]


# Split dataset

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y_categorical[:split]
y_test = y_categorical[split:]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# CNN model

model = Sequential([
    Input(shape=(40, 130, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(len(emotion_names), activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# Evaluate model

loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print("Test Accuracy:", accuracy)