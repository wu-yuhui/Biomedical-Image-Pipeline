from scipy.misc import imread, imresize
import glob
import os
import numpy as np
import tensorflow as tf
import scipy
import scipy.io as sio

large_frame_size = 512

output_window_size = 240

T = 1


def read_data_from_folder(path):
    list_of_input = []
    list_of_name = []

    img_files = sorted(glob.glob(path + "image/*.png"))
    print('Reading Images: ', img_files)

    for img in img_files:
        slash_pos = img.rindex('/')
        dot_pos = img.rindex('.')
        img_name = img[slash_pos+1:dot_pos]    # '9_1'

        list_of_name.append(img_name)

        crop_path = path + "boxes/" + img_name + ".txt"
        current_coords = read_boxes_in_list(crop_path)

	# large_frame_size = 512
	# output_window_size = 240
	# T = 1
        current_large_labels = np.zeros((T,large_frame_size,large_frame_size,1))

        current_im = imread(img)
        current_sample = np.zeros((T,large_frame_size,large_frame_size,3))

        current_sample[0,...] = current_im

        X= extract_tubes(current_sample, current_coords)
        list_of_input.append(X)

    #print('list_of_names:', list_of_name)
    # cell num array
    experiment_sizes = [x.shape[0] for x in list_of_input]

    trX= np.zeros((sum(experiment_sizes),T,output_window_size,output_window_size,4))
    
    for i in range(len(list_of_input)):
        trX[sum(experiment_sizes[:i]):sum(experiment_sizes[:i+1])] = list_of_input[i]

    return trX, experiment_sizes, list_of_name



def extract_tubes(full_video, coordinates):
    # reshape (20, 512, 512, 3) -> (  1  , 20, 512, 512, 3)
    full_video = np.reshape(full_video, (1,T,large_frame_size,large_frame_size,3))

    window_size = 128
    half_size = window_size / 2

    # process center of each cell
    tube_centers = coordinates
    tube_centers = np.array([[x[1], x[0]] for x in tube_centers])

    num_tubes = tube_centers.shape[0]

    all_tubes = np.zeros((num_tubes,T,output_window_size,output_window_size,4))

    for i in range(num_tubes):
	## Convert all elements to int  2018.03.07
        current_tube = full_video[:,:,int(max(tube_centers[i,0]-half_size,0)):int(min(tube_centers[i,0]+half_size,large_frame_size-1)),int(max(tube_centers[i,1]-half_size,0)):int(min(tube_centers[i,1]+half_size,large_frame_size-1)),:]

        current_tube_large = np.zeros((1,T,output_window_size,output_window_size,4))
        for j in range(T):
            current_tube_large[0,j,:,:,:3] = scipy.misc.imresize(np.squeeze(current_tube[0,j,...]),(output_window_size,output_window_size,3),interp='nearest')
            current_tube_large[0,j,:,:,3] = current_tube_large[0,j,:,:,0]

        all_tubes[i,...] = current_tube_large		# size: (cells_num, 20 frames, 240, 240, 4)

    return all_tubes#, all_labels



def read_boxes_in_list(crop_path):
    with open(crop_path, 'r') as f:
        p = f.readline()
        L = []
        while p:
            num = [float(x) for x in p.split()]
            center = [int((num[0]+num[2])/2), int((num[1]+num[3])/2)]
            L.append(center)
            p = f.readline()
        return np.array(L)

