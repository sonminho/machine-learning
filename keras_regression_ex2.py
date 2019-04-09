from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 모델 구성
x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise
y_data = y_data.reshape(2000, 1)

model = Sequential()
model.add(Dense(1, input_shape=(3,)))
model.compile(loss='mse', optimizer='sgd')
model.summary()

# 훈련
history= model.fit(x_data, y_data, epochs=1000, verbose=0)

# 가중치, bias 확인
weights, bias = model.layers[0].get_weights()
print(weights, bias)

# history 시각화
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
