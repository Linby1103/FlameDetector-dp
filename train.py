# -*- coding: utf-8 -*-
# @Time : 2020/6/15 上午9:28
# @Author : LiBin
# @File : model.py
# @Software: PyCharm

import tensorflow as tf
import generator
import model
from tqdm import tqdm
import cfg

def train():
	"""
	Train model
	:return:
	"""
	X = tf.placeholder(tf.float32, [cfg.CFG['batch_size'],cfg.CFG['image_size'],cfg.CFG['image_size'],cfg.CFG['image_channel']], name='train-input')
	Y = tf.placeholder(tf.float32, [None], name='label-input')
	train, train_label = generator.get_files('./dataset/')

	train_batch, train_label_batch = generator.get_batches(train, train_label, cfg.CFG["image_size"], cfg.CFG["image_size"], cfg.CFG["batch_size"], 20)

	train_logits = model.model(train_batch, cfg.CFG["batch_size"], cfg.CFG["classes"])

	train_loss = model.losses(train_logits, train_label_batch)
	train_op = model.trainning(train_loss, cfg.CFG["learning_rate"])
	train_acc = model.evaluation(train_logits, train_label_batch)

	sess = tf.Session()
	saver = tf.train.Saver(max_to_keep=5)
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)


	try:
		ptar =tqdm(range(50))
		for iter in ptar:
			if coord.should_stop():
				break

			image_data, label = sess.run([train_batch, train_label_batch])
			# _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

			_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc], feed_dict={X: image_data, Y: label})

			ptar.set_description("iterative  %d times,train loss=%.4f     train accuracy = %.4f" % (iter,tra_loss, tra_acc))


			checkpoint_path=cfg.CFG['model_path']+'flame_dector_%.4f.ckpt'%tra_acc
			# print(checkpoint_path)
			saver.save(sess, checkpoint_path, global_step=iter)


	except tf.errors.OutOfRangeError:
		print('Done training')
	finally:
		coord.request_stop()
	coord.join(threads)


if __name__=="__main__":
	train()