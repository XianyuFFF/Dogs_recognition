from itertools import groupby
from tensorflow.python.ops import random_ops
from collections import defaultdict
import tensorflow as tf
import glob
from PIL import Image
import numpy as np

BATCH_SIZE = 1
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 151
IMAGE_CHANNEL = 1
BREEDS = 3  # 狗的种类
# 对不同文件夹中的图片进行处理
image_filenames = glob.glob('imagenet-dogs/Images/n02*/*.jpg')
# print(image_filenames[0:2])
train_dataset = defaultdict(list)  # defaultdict可使用未定义的Key
test_dataset = defaultdict(list)

image_filename_with_breed = map(lambda filename: (filename.split("\\")[1], filename), image_filenames)
# python map(fun,[arg]+) return iterators,turn to list method:list(iterators)
# print(list(image_filename_with_breed)[0:2])
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # dog_breed表示狗的类型
    for i, breed_image in enumerate(breed_images):
        # enumerate同时列出迭代对象的下标和值
        # print(i,breed_image)
        # 将20%的数据划入测试集
        if i % 5 == 0:
            test_dataset[dog_breed].append(breed_image[1])
        else:
            train_dataset[dog_breed].append(breed_image[1])
    breed_train_count = len(train_dataset[dog_breed])
    breed_test_count = len(test_dataset[dog_breed])
    assert round(breed_test_count / (breed_train_count + breed_test_count), 2) > 0.18, 'Not enough testing images'


def write_records_file(sess, dataset, record_location):
    current_index = 0
    writer = 0
    for breed, image_filenames in dataset.items():
        for image_filename in image_filenames:
            # 将每100个图片划入一TFRecord文件
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location,
                                                                                       current_index=current_index)
                print('record_filename:', record_filename)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)  # 转化为灰度图
            resized_image = tf.image.resize_images(grayscale_image, [IMAGE_WIDTH, IMAGE_HEIGHT])  # 更改图片尺寸
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = breed.encode('utf-8')
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))  # 以比特流的形式进行存储，固定格式
            writer.write(example.SerializeToString())
    writer.close()


'''
#存储TFRecord session
with tf.Session() as sess:
	coord=tf.train.Coordinator()
	threads= tf.train.start_queue_runners(coord=coord)
	write_records_file(sess,test_dataset,'./imagenet-dogs/testing-images/testing-image')
	write_records_file(sess,train_dataset,'./imagenet-dogs/training-images/training-image')
	coord.request_stop()
	coord.join(threads)
'''


def read_records_file(record_path):
    # filename_queue = tf.train.string_input_producer(['./imagenet-dogs/training-images/training-image-100.tfrecords','./imagenet-dogs/training-images/training-image-100.tfrecords'])
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(record_path),
                                                    shuffle=True)  # shuffle参数True表示不按顺序执行，False表示按顺序执行
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized, features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        })
    recordimage = tf.decode_raw(features['image'], tf.uint8)
    recordimage = tf.reshape(recordimage, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    recordlabel = tf.cast(features['label'], tf.string)
    return recordlabel, recordimage


def Batch_dataset(image, label):#将图片进行打包
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    return tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=capacity,
                                  min_after_dequeue=min_after_dequeue)


def Float_image_batch(image_batch):
    return tf.image.convert_image_dtype(image_batch, tf.float32)


def LabelsToNum(label_batch):
    # 将狗品种标签字符串变为数值型标签 label_batch
    # Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    breeds = list(map(lambda c: c.split("\\")[-1], glob.glob("./imagenet-dogs/Images/*")))
    # Match every label from label_batch and return the index where they exist in the list of classes
    # Numlabels = tf.map_fn(lambda l: tf.where(tf.equal(breeds, l))[0, 0:1][0], label_batch, dtype=tf.int64)
    Numlabels = np.zeros([BATCH_SIZE, len(breeds)], int)
    for i in range(BATCH_SIZE):
        for j in range(len(breeds)):
            # print(label_batch[i],breeds[j].encode(encoding='utf-8'))
            if label_batch[i] == breeds[j].encode(encoding='utf-8'):
                Numlabels[i][j] = 1
    return Numlabels


# 需要提前定义graph
train_recordlabel, train_recordimage = read_records_file('./imagenet-dogs/training-images/*.tfrecords')
train_image_batch, train_label_batch = Batch_dataset(train_recordimage, train_recordlabel)
test_recordlabel, test_recordimage = read_records_file('./imagenet-dogs/testing-images/*.tfrecords')
test_image_batch, test_label_batch = Batch_dataset(test_recordimage, test_recordlabel)
# tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity)  产生乱序的batch
# capacity是队列中的容量
float_trainimage_batch = Float_image_batch(train_image_batch)
float_testimage_batch = Float_image_batch(test_image_batch)

image_holder = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
label_holder = tf.placeholder(tf.int64, [BATCH_SIZE, BREEDS])
#第一层卷积池化操作
conv2d_layer_one = tf.contrib.layers.convolution2d(image_holder, num_outputs=32, kernel_size=(5, 5),
                                                   activation_fn=tf.nn.relu,
                                                   stride=(2, 2), trainable=True)
pool_layer_one = tf.nn.max_pool(conv2d_layer_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#第二层卷积池化操作
conv2d_layer_two = tf.contrib.layers.convolution2d(pool_layer_one, num_outputs=64, kernel_size=(5, 5),
                                                   activation_fn=tf.nn.relu, stride=(1, 1), trainable=True)
pool_layer_two = tf.nn.max_pool(conv2d_layer_two, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
#扁平化处理
flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        BATCH_SIZE,  # Each image in the image_batch
        -1  # Every other dimension of the input
    ])  # 用于连接全连接层
#全连接层
hidden_layer_three = tf.contrib.layers.fully_connected(

    flattened_layer_two,
    512,
    # weights_initializer=tf.Variable(tf.truncated_normal([38912, 200], stddev=0.1)),
    activation_fn=tf.nn.relu
)
# hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    BREEDS,  # Number of dog breeds in the ImageNet Dogs dataset
)
train_prediction = tf.nn.softmax(final_fully_connected)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_holder, logits=train_prediction))
optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

# correctlist = tf.equal(tf.argmax(label_holder, 1), tf.argmax(train_prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correctlist, tf.float32))

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(final_fully_connected)
    # print(sess.run(train_labels))
    # print(sess.run(train_labels))
    for h in range(10):#训练
        print('第' + str(h) + '轮训练')
        for _ in range(3000):
            train_i = sess.run(float_trainimage_batch)
            train_l = sess.run(train_label_batch)
            train_labels = LabelsToNum(train_l)
            # iii = train_i.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
            # img = Image.fromarray(iii, 'L')  # 测试TFRecord转化生成的图片,灰度图“L”,彩色图“RGB”
            # img.save('./output/' + str(train_labels) + str(_) + '.jpg')  # 存下图片
            sess.run(optimizer, feed_dict={image_holder: train_i, label_holder: train_labels})
        print('loss函数：', sess.run(loss, feed_dict={image_holder: train_i, label_holder: train_labels}))
    for t in range(20):#测试
        test_i = sess.run(float_testimage_batch)
        test_l = sess.run(test_label_batch)
        test_labels = LabelsToNum(test_l)
        result = sess.run(train_prediction, feed_dict={image_holder: test_i})
        print("测试：", test_labels, result)
    coord.request_stop()
    coord.join(threads)
