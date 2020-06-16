from PIL import Image
import numpy as np
import tensorflow as tf
import model
import glob
import cfg
import os
acc=[0,0]
CHECK_POINT_DIR = './checkpoint/'

def Preprocess(image_array):
	image = tf.cast(image_array, tf.float32)
	image = tf.image.per_image_standardization(image)
	image = tf.reshape(image, [1, 64, 64, 3])
	return image
def evaluate_one_image():

	with tf.Graph().as_default():
		X = tf.placeholder(tf.float32, [1, cfg.CFG['image_size'], cfg.CFG['image_size'],
										cfg.CFG['image_channel']],
						   name='train-input')


		logit = model.model(X, 1, 2)
		logit = tf.nn.softmax(logit)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			print ('Reading checkpoints...')
			ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Loading success, global_step is %s' %global_step)
			else:
				print ('No checkpoint file found')

			img_list = glob.glob(os.path.join('./dataset/1/', '*.jpg'))
			for img_path in img_list:
				image = Image.open(img_path)
				image = image.resize([64, 64])
				image = np.array(image)

				img=sess.run(Preprocess(image))

				prediction = sess.run(logit,feed_dict={X:img})

				max_index = np.argmax(prediction)
				print(prediction)
				acc[max_index] += 1
				if max_index == 0:
					print('this is Negative rate: %.6f, result prediction is [%s]' % (
					prediction[:, 0], ','.join(str(i) for i in prediction[0])))
				else:
					print ('this is Positive rate: %.6f, result prediction is [%s]' % (
					prediction[:, 1], ','.join(str(i) for i in prediction[0])))
			print('Negative=%d Positive=%d' % (acc[0], acc[1]))





if __name__ == '__main__':
	evaluate_one_image()









