import tensorflow as tf
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import wave
import scipy.io.wavfile as sci
import numpy as np
from pydub import AudioSegment as aseg

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파이미터를 튜닝하기 위한 용도 (흔들리면 무엇때문에 좋아졌는지 알기 어려움)
# tf.set_random_seed(777)
tf.compat.v1.set_random_seed(777)
sec = 1  # 1sec sampling size
data = []  # 총 데이터 리스트

sample_rate, ary = sci.read(r'C:\Users\KD\Desktop\noise.wav')  # mono로 변환된 wav 열기

# ary = np.array(pwm_d, dtype = 'int32') #array로 저장

ary = ary[:, 0]  # stereo to mono
ary = ary[:sec * 180*44100]

print('ary : ', ary)
print('len : ', len(ary))

for i in range(int(len(ary) / sec)):  # 1초 단위로 슬라이싱하여 분할
    data.append(ary[i * sec:(i + 1) * sec])


# standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


# 너무 작거나 너무 큰값이 학습을 방해하는 것을 막기위해 정규화 함
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1 사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 zero division error 방지차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


##하이퍼파라미터
input_data_column_cnt = sec  # 입력데이터의 컬럼갯수 (Variable 갯수)
output_data_column_cnt = sec  # 결과데이터의 컬럼 개수

seq_length = int(180*44100*0.7) # 1개 시퀸스의 길이 (시계열데이터 입력 개수)

rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향 (default 1.0)
num_stacked_layers = 1  # stacked LSTM 갯수
keep_prob = 1.0  # dropout 할 때 keep비율    test떈 1.0 !!! 까먹지 마셈

epoch_num = 1000  # 에폭횟수 (학습데이터 반복학습 횟수)
learning_rate = 0.01  # 학습률

'''
# 데이터 로딩...
#w = wave.open('noise.wav', 'r') #wav 파일 열기
#s = aseg.from_wav('noise.wav')  stereo to mono#
#s = s.set_channels(1)
#s.export('noise2.wav', format = 'wav') #mono 출력
'''

# 데이터 정규화

norm_data = min_max_scaling(data)

x = norm_data[0:-1]  # axis =1, 세로로 합친다

y = norm_data[1:]  # 타겟데이터

dataX = x  # 입력으로 사용될 Sequence Data
dataY = y  # 출력(타겟)으로 사용

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # 정답 데이터
    if i is 0:
        print(_x, '->', _y)  # 첫번째 행만 출력
    # dataX.append(_x)# dataX 리스트에 추가
    # dataY.append(_y)# dataY 리스트에 추가
    dataX = np.vstack((dataX, _x))
    dataY = np.vstack((dataY, _y))

# 학습용 / 테스트용 데이터 분리
# 70%를 학습용으로 사용
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

# print('len(dataX) : ',len(dataX))
# print('shape(dataX) : ',np.shape(dataX))

# placeholder 생성 (입력 X, 출력 Y 생성)

X= tf.compat.v1.placeholder(tf.float32,[None, seq_length, input_data_column_cnt])
# X = tf.compat.v1.placeholder(tf.float32, [None, input_data_column_cnt])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
# Y = tf.compat.v1.placeholder(tf.float32, [None, output_data_column_cnt])

# 검증용 측정지표를 산출하기 위한 targets, prediction을 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
# targets = tf.compat.v1.placeholder(tf.float32,[None,output_data_column_cnt])
predictions = tf.placeholder(tf.float32, [None, 1])


# predictions = tf.compat.v1.placeholder(tf.float32,[None,output_data_column_cnt])


# LSTM네트워크 생성

def lstm_cell():
    # lstm셀 생성

    # num_units : 각 Cell 출력 크기
    # forget_bias : forget gate biases (default : 1.0)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim, forget_bias=forget_bias, state_is_tuple=True,
                                        activation=tf.nn.softsign)

    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# num_stacked_layers개의 층으로 쌀인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]

multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# RNN Cells (LSTM Cells)들 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

# LSTM RNN의 마지막 (hidden)출력만을 사용했다
# Many to one 형태

hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

# cost function = 평균제곱오차
loss = tf.reduce_sum(tf.square(hypothesis - Y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate)  #LSTM과 궁합 별로

train = optimizer.minimize(loss)

# rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))
rmse = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(targets, predictions)))

# RMSE (Root Mean Square Error)
# 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균오차가 나옴
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))  # 윗줄과 같다

train_error_summary = []  # 학습용 데이터의 오류를 중간중간 기록
test_error_summary = []  # 테스트 데이터의 오류를 중간중간 기록
test_predict = ''

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# sess.run(tf.compat.v1.global_variables_initializer())


# 학습!
start_time = datetime.datetime.now()  # 시작시간 기록!
print('학습 시작!')

for epoch in range(epoch_num):

    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})

    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다, 마지막 epoch에서

        # 학습용 데이터로 rmse오차 구하기
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 데스트용 데이터로 rmse오차 구하기
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        # 현재 오류 출력
        print('epoch : {}, train_error(A) : {}, test_error(B): {}, B-A : {}'.format(epoch + 1, train_error, test_error,
                                                                                    test_error - train_error))

end_time = datetime.datetime.now()  # 종료시각 기록
elapsed_time = end_time - start_time  # 경과시간
print('elapsed_time : ', elapsed_time)
print('elapsed_time per epoch : ', elapsed_time / epoch_num)

### 하이퍼파라미터 출력

print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))
