# -*- coding: utf-8 -*-
# @Time : 2020/6/15 上午9:28
# @Author : LiBin
# @File : model.py
# @Software: PyCharm

import os
import numpy as np
import tensorflow as tf

def get_files(file_path):
	"""
	:param file_path: 训练数据所在的目录
	:return:
	"""
	class_train = []
	label_train = []
	for train_class in os.listdir(file_path):
		for pic_name in os.listdir(file_path + train_class):
			class_train.append(file_path + train_class + '/' + pic_name)
			#train_class is 0,1
			label_train.append(train_class)
	#merge trainimage and trainlabel to 2D array (2,n)
	CombineLine = np.array([class_train, label_train])

	temp = CombineLine.transpose()
	#随机打乱数据
	np.random.shuffle(temp)

	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	# class is 0:Negative samples 1:Postive samples
	label_list = [int(i) for i in label_list]
	return image_list, label_list

def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
	"""
	Get batch size data
	:param image: train dataset
	:param label: label
	:param resize_w: image width
	:param resize_h: image height
	:param batch_size:number of data in a batch
	:param capacity:
	:return:
	"""
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int64)
	queue = tf.train.slice_input_producer([image, label])
	label = queue[1]
	image_temp = tf.read_file(queue[0])
	image = tf.image.decode_jpeg(image_temp, channels = 3)
	#resize image
	image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

	image = tf.image.per_image_standardization(image)

	image_batch, label_batch = tf.train.batch([image, label], batch_size = batch_size,
		num_threads = 64,
		capacity = capacity)
	images_batch = tf.cast(image_batch, tf.float32)
	labels_batch = tf.reshape(label_batch, [batch_size])
	return images_batch, labels_batch
