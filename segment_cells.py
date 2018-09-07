import tensorflow as tf
import os, sys
slim = tf.contrib.slim
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import model_cmc
import os
from utils.data_operations_lite_pipeline import read_data_from_folder
from sklearn import metrics
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

tf.app.flags.DEFINE_boolean(
    'post_processing', False, 'post-processing')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'Logs_cmc/', 'path to checkpoint')
tf.app.flags.DEFINE_string(
    'model', 'unet', 'Model to eval')
FLAGS = tf.app.flags.FLAGS

import numpy as np

def normalize_streams(data):
    mu_0, std_0 = np.mean(data[:, :, :,:, 0]), np.std(data[:, :, :,:, 0])
    mu_1, std_1 = np.mean(data[:, :, :,:, 1]), np.std(data[:, :, :,:, 1])
    mu_2, std_2 = np.mean(data[:, :, :,:, 2]), np.std(data[:, :, :,:, 2])
    mu_3, std_3 = np.mean(data[:, :, :,:, 3]), np.std(data[:, :, :,:, 3])
    data[:, :, :,:, 0] = (data[:, :, :,:, 0] - mu_0) / std_0
    data[:, :, :,:, 1] = (data[:, :, :,:, 1] - mu_1) / std_1
    data[:, :, :,:, 2] = (data[:, :, :,:, 2] - mu_2) / std_2
    data[:, :, :,:, 3] = (data[:, :, :,:, 3] - mu_3) / std_3

    return data

output_window_size = 240

#File path for test images
#dataset_path = input('>>> Enter Folder Path from Detecion Network (the parent folder of \'image/\' and \'boxes/\') \n => ')
#if dataset_path[-1] != '/':
#    dataset_path += '/'

dataset_path = 'result/'

trX, experiment_sizes, experiment_names = read_data_from_folder(dataset_path)

trX = normalize_streams(trX)

batchsize = 1

#print('experiment_sizes:', experiment_sizes)
#print('experiment_names:', experiment_names)

T = 1  
target = 5
num_steps = 30
stride = (T - 1)//2

img_input = tf.placeholder(tf.float32, shape=(batchsize, T, output_window_size, output_window_size, 4))
la_input = tf.placeholder(tf.int32, shape=(batchsize, T, output_window_size, output_window_size, 1))
is_training = tf.placeholder(tf.bool)
la_input_onehot = la_input #tf.one_hot(la_input,2)

net = model_cmc.Model()
logits, _ = net.net(img_input, is_training)
logits_packed = tf.stack(logits)
logits_softmaxed = tf.nn.softmax(logits)
logits_amax = tf.squeeze(tf.argmax(logits_packed, axis=4))
loss_op, mean_iou_op, growth = net.weighted_losses_growth_term(logits, la_input_onehot)
mean_iou_scalar = mean_iou_op[0]
optimizer = tf.train.AdamOptimizer(1e-03) # 1e-03 was good

train_step = slim.learning.create_train_op(loss_op,optimizer)

print('Please wait...')

saver = tf.train.Saver()
#config = tf.ConfigProto(device_count = {'GPU': 3}, allow_soft_placement=True)
config = tf.ConfigProto(allow_soft_placement=True)
#config = tf.ConfigProto(log_device_placement=True)

with tf.Session(config=config) as sess:
    saver.restore(sess, 'model/model_1525375412.8794093.ckpt')
    sess.run(tf.local_variables_initializer())
    print("Model Loaded !!!")

    # Build folder to store masks for each image with each cell 
    res_dir = dataset_path + 'masks/'
    try:
        os.stat(res_dir)
    except:
        os.mkdir(res_dir)

    trY = np.random.random((batchsize, T, 240, 240, 1))
    countTrX = 0

    for num in range(len(experiment_sizes)):

        name = experiment_names[num]
        res_path = res_dir + name + '.mat'

        image_log = np.zeros((experiment_sizes[num],1,output_window_size,output_window_size,4))      # (cell_nums in one image, 1, 240, 240, 4)
        preds_log = np.zeros((experiment_sizes[num],1,output_window_size, output_window_size, 1))

        for j in range(experiment_sizes[num]):
            #np.random.random((1, T, 240, 240, X:4 or Y:1))
            image_batch = np.reshape( trX[(countTrX+j):(countTrX+j+batchsize),0,:,:,:] , [batchsize,T,output_window_size,output_window_size,4])
            label_batch = np.reshape( trY , [batchsize,T,output_window_size,output_window_size,1])

            lss, sample_logits, current_logits_amax = sess.run([loss_op, logits_packed, logits_amax], feed_dict={
                img_input: image_batch, la_input: label_batch,
                is_training: False})

            image_log[j] = image_batch[0,stride]
            preds_log[j] = np.reshape(current_logits_amax,[output_window_size,output_window_size,1])

        # Save all cells of each image in a mat file
        print('Saving Segmentation Results of total ', experiment_sizes[num],  ' Cells in ', name + '.png')
        sio.savemat(res_path,{'images':image_log,'preds':preds_log})

        # Offset for the next experiment size
        countTrX += experiment_sizes[num]
 
