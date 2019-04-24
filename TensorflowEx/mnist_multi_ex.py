import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './mnist/data/'
data = input_data.read_data_sets(DATA_DIR, one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 은닉 1계층 입출력
W1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.01))
L1 = tf.nn.relu(tf.matmul(x, W1))

# 은닉 2계층 입출력
W2 = tf.Variable(tf.random_normal([256, 256], stddev = 0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 최종 출력
W3 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01))
model = tf.matmul(L2, W3)

# 손실함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))

# 최적화함수
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    # 학습을 위한 변수 초기화
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(data.train.num_examples/batch_size)

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            _, cost_val =  sess.run([optimizer, cost], feed_dict = {
                x : batch_xs,
                y : batch_ys
            })
            total_cost += cost_val
        print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    print('최적화 완료!')

    # 학습 결과 확인
    prediction = tf.argmax(model, axis = 1)
    target = tf.argmax(y, axis = 1)

    print("예측 값 : ", sess.run(prediction, feed_dict = {
        x : data.test.images
    }))

    print("실제 값 : ", sess.run(target, feed_dict = {
        y : data.test.labels
    }))

    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) 
    print('정확도:', sess.run(accuracy, feed_dict = {
        x : data.test.images, 
        y : data.test.labels}))