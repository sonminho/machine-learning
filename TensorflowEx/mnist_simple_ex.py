import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './mnist/data/'
data = input_data.read_data_sets(DATA_DIR, one_hot = True)

print(tf.convert_to_tensor(data.train.images).get_shape())
print(tf.convert_to_tensor(data.train.labels).get_shape())

# 입력 데이터를 위한 플레이스홀더
x = tf.placeholder(tf.float32, [None, 784])

# 가중치
W = tf.Variable(tf.zeros([784, 10]))

# 편향
b = tf.Variable(tf.zeros([10]))

y_true = tf.placeholder(tf.float32, [None, 10])

# 모델 정의
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 손실 함수
cross_entropy = -tf.reduce_mean(y_true * tf.log(y_pred))

# 최적화 함수
train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cross_entropy)

# 정답 판별 연산 노드 - 정답 예측 인덱스와 정답 레이블 인덱스 일치 여부
correct_mask = tf.equal(tf.argmax(y_pred, axis = 1), tf.argmax(y_true, axis = 1))

# 정확도 연산 노드
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # 학습을 위한 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(100):
        sess.run(train_step, feed_dict = {
            x : data.train.images, # 훈련 이미지
            y_true : data.train.labels # 훈련 이미지 정답 레이블
        })
    
    ans = sess.run(accuracy, feed_dict = {
        x : data.test.images,
        y_true : data.test.labels
    })

    print("정확도 = {:.4}".format(ans * 100))

# 배치 트레이닝 적용
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

with tf.Session() as sess:
    # 학습을 위한 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(train_step, feed_dict = {
            x : batch_xs,
            y_true : batch_ys
        })
    
    # 테스트
    ans = sess.run(accuracy, feed_dict = {
        x : data.test.images,
        y_true : data.test.labels
    })

    print("배치 트레이닝 정확도 = {:.4}".format(ans * 100))