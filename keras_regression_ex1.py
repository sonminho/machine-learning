from keras.models import Sequential
from keras.layers import Dense
import numpy as np

num_points = 1000

x_data = np.random.normal(0.0, 0.55, (num_points))
y_data = x_data * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (num_points))

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

# 훈련
history= model.fit(x_data, y_data, epochs=10000, verbose=0)

weights, bias = model.layers[0].get_weights()
print(weights, bias)

# 테스트
test_indexs = np.random.choice(num_points, 10) # 10개 무작위 추출
test_x = x_data[test_indexs]
test_y = y_data[test_indexs]
print('Targets :', test_y)
print('Predictions:', model.predict(test_x).flatten())
