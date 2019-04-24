import tensorflow as tf 
import numpy as np 

x_data = [[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]]
y_data = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1]]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random.uniform([2,10], -1, 1))
W2 = tf.Variable(tf.random.uniform([10,3], -1, 1))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.add(tf.matmul(x, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

# 손실 함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model))

# 최적화 함수
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# 경사하강법
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

with tf.Session() as sess:
    # 학습을 위해 변수를 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(100):
        sess.run(optimizer, feed_dict = {
            x : x_data, # 입력 데이터
            y : y_data  # 정답 레이블
        })

        # 중간 과정 cost 출력하기
        if(step + 1) % 10 == 0:
            print(step + 1, sess.run(cost, feed_dict = {
                x : x_data,
                y : y_data
            }))
    
    # 학습 결과 확인
    prediction = tf.argmax(model, axis = 1)
    target = tf.argmax(y, axis = 1)

    print("예측 값 : ", sess.run(prediction, feed_dict = {
        x : x_data,
    }))

    print("실제 값 : ", sess.run(target, feed_dict = {
        y : y_data
    }))

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    print("정확도 : {:.2f}".format(sess.run(accuracy * 100, feed_dict = {
        x : x_data,
        y : y_data
    })))