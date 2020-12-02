# Программа для определения человеческой речи 
## Описание
Программа для создания и обучения нейросети, с целью отличить четкую человеческую речь, от искусственно сгенерированной или фоновой.
## Основные функции
* audioDataProcessing - выделение параметров звукового сигнала
* creatAudioCSV - создание и заполнение CSV файла данными звуковых сигналов
## Библиотеки 
* Python 3.7
* pip install librosa
* pip install pandas
* pip install numpy
* pip install csv
* pip install tanserfow==2.3.1
## Установка/сборка проекта 
1. Установите библиотеку
   * pip install pyinstaller
2. Соберите проект с помощью команды
   * pyinstaller main.py
## Примечания
Данная программа работает с **wav** расширением аудио файлов


