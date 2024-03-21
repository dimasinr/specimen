import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd

# data = [4023,3558,5732,6623,5144,1200,4851, 4020, 4829,4834,5717,4182,1344,5723,2442,4672,3603]
data = []

file = 'Book1.xlsx'

df = pd.read_excel(file)
for index, row in df.iterrows():
    if row is not None and row[5] != '22:00':
        try:
            if int(row[5]):
                data.append(row[1])
                data.append(row[2])
                data.append(row[3])
                data.append(row[4])
                data.append(row[5])
                data.append(row[6])
        except:
           continue

data_max = max(data)
data_min = min(data)
normalized_data = [(value - data_min) / (data_max - data_min) for value in data]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

look_back = 3
X, Y = create_dataset(normalized_data, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=130, batch_size=1, verbose=2)

last_data = normalized_data[-look_back:]
last_data = np.reshape(last_data, (1, look_back, 1))
prediction = model.predict(last_data)

predicted_value = prediction[0, 0] * (data_max - data_min) + data_min

def format_rupiah(angka):
    angka = str(angka)
    if angka.isdigit():
        angka = int(angka)
        rupiah = "{:,}".format(angka).replace(',', '.')
        return f'Rp {rupiah}'
    else:
        return 'Format input tidak valid'


print("Prediksi angka berikutnya: ", (int(round(predicted_value, 0))))
