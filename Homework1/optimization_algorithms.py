import csv
import numpy as np
import tensorflow as tf

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

x = []
y = []
with open('Video_Game_Sales_Revised.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if isfloat(row[0]) and isfloat(row[1]):
            x.append(float(row[0]))
            y.append(float(row[1]))

# adam = tf.keras.optimizers.Adam()
# var1 = tf.Variable(x)
# var2 = tf.Variable(y)
# loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# adam.minimize(loss, var_list=[var1, var2])
# print("Adam Training :",var1[:])

adam = tf.keras.optimizers.Adam()
m = tf.keras.models.Sequential([tf.keras.layers.Dense(100)])
m.compile(adam, loss='mse')
data = np.arange(len(x)*2).reshape(2, len(x))
for i in range(len(x)):
    data[0][i] = x[i]
    data[1][i] = y[i]

labels = np.zeros(2)
print('Adam Training'); results = m.fit(data, labels)
print(len(adam.get_weights()))

rms = tf.keras.optimizers.Nadam()
m = tf.keras.models.Sequential([tf.keras.layers.Dense(100)])
m.compile(rms, loss='mse')

print('RMS Training'); results = m.fit(data, labels)
print(len(rms.get_weights()))
