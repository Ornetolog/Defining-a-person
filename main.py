import librosa
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
from tensorflow.keras import Sequential
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


# Выделение параметров звукового сигнала
def audioDataProcessing(y, sr):
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return rmse, chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc


# Создание CSV файла
def creatAudioCSV(headerFile, path, typeData):
    file = open('dataset.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(headerFile)
    for vt in typeData:
        for filename in os.listdir(f'{path}/{vt}'):
            songname = f'{path}/{vt}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            rmse, chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc = audioDataProcessing(y, sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {vt}'
            file = open('dataset.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())


# fileTraningPath = f'D:/Zub/Programming/Python/TestCase/idrnd/Defining a person/Training_Data/'
# voice_type = 'human spoof'.split()
# Создание заголовка для файла CSV
# header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
# for i in range(1, 21):
#     header += f' mfcc{i}'
# header += ' label'
# header = header.split()

# Выполняем предварительную обработку данных
#   загрузку данных CSV
#   создание меток
#   масштабирование признаков
#   разбивку данных на наборы для обучения и тестирования
data = pd.read_csv('dataset.csv')
data.head()
print(data)
# Удаление ненужных столбцов
data = data.drop(['filename'], axis=1)
print(data)
# Создание меток
genre_list = data.iloc[:, -1]
encoder = LabelEncoder().fit(genre_list)
y = encoder.transform(genre_list)
print(y)
# Масштабирование столбцов признаков
X = np.array(data.iloc[:, :-1], dtype=float)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
print(X)
# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(np.array(X_test[0]))
print(X_train.shape[1])
# Создаем модель ANN
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

classifier = model.fit(X_train,
                       y_train,
                       epochs=100,
                       batch_size=128)
print(model.summary())
# так как модель не принимает 1 объект нужно передовать объект через np.expand_dims
vrem = np.expand_dims(X_test[0], axis=0)
model.evaluate(X_test, y_test)
print(model.predict(vrem), y_test[0])
vrem = np.expand_dims(X_train[0], axis=0)
print(model.predict(vrem), y_train[0])
