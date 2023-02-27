# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import cv2
import csv
import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from pickle import dump, load
from sklearn import preprocessing
from datetime import datetime
from scipy.spatial.transform import Rotation as R


dataset_path = "/home/willow/Robotics/datasets/PRI/MarkedHeavyBox/"

train_data_dir = dataset_path + 'train/'
test_data_dir = dataset_path + 'test/'
test_data_dir_2 = dataset_path + "test_examples/" # leave as = dataset_path if no unseen data

train_out_dir  = dataset_path + 'Dataset_2c_15p/train_formatted/'
test_out_dir   = dataset_path + 'Dataset_2c_15p/test_formatted/'
test_out_dir_2 = dataset_path + 'Dataset_2c_15p/test_examples_formatted/' # dataset_path + 'Dataset_2c_5p/test_unseen_formatted/' just set as '' for nothing
scaler_out_dir = dataset_path + 'Dataset_2c_5p/scalers/'


# dataset_path = "/home/willow/Robotics/datasets/PRI/household_object_dataset/"

# train_data_dir = dataset_path + 'train/'
# test_data_dir = dataset_path + 'test_seen/'
# test_data_dir_2 = dataset_path + "test_unseen/" # leave as = dataset_path if no unseen data

# train_out_dir  = dataset_path + 'Dataset_2c_20p/train_formatted/'
# test_out_dir   = dataset_path + 'Dataset_2c_20p/test_seen_formatted/'
# test_out_dir_2 = dataset_path + 'Dataset_2c_20p/test_unseen_formatted/' # dataset_path + 'Dataset_2c_5p/test_unseen_formatted/' just set as '' for nothing
# scaler_out_dir = dataset_path + 'Dataset_2c_20p/scalers/'


image = False
context_length = 2
horrizon_length = 15
one_sequence_per_test = False
image_height, image_width = 64, 64

lines = ["image: " + str(image), "image_height: " + str(image_height), "image_width: " + str(image_width), "context_length: " + str(context_length),
         "horrizon_length: " + str(horrizon_length), "one_sequence_per_test: " + str(one_sequence_per_test), "dataset_path: " + str(dataset_path),
         "train_data_dir: " + str(train_data_dir), "test_data_dir: " + str(test_data_dir), "train_out_dir: " + str(train_out_dir), "test_data_dir_2: " + str(test_data_dir_2),
         "test_out_dir: " + str(test_out_dir), "test_out_dir_2: " + str(test_out_dir_2), "scaler_out_dir: " + str(scaler_out_dir)]
with open(scaler_out_dir + "dataset_info.txt", 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

class data_formatter:
    def __init__(self):
        self.image = image
        self.files_test = []
        self.files_test_2 = []
        self.files_train = []
        self.full_data_robot = []
        self.full_data_image = []
        self.full_data_tactile = []
        self.image_width = image_width
        self.image_height = image_height
        self.context_length = context_length
        self.horrizon_length = horrizon_length
        self.one_sequence_per_test = one_sequence_per_test
        self.sequence_length = self.context_length + self.horrizon_length

    def create_map(self):
        if test_out_dir_2 == '':
            stages = [train_out_dir, test_out_dir]
        else:
            stages = [train_out_dir, test_out_dir, test_out_dir_2]
        for stage in stages:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            elif stage == test_out_dir:
                files_to_run = self.files_test
            elif stage == test_out_dir_2:
                files_to_run = self.files_test_2

            print(files_to_run)
            path_save = stage
            fail_case = 0
            progress_bar = tqdm(enumerate(files_to_run), total=(len(files_to_run)))

            for experiment_number, file in progress_bar:
                # try:
                tactile, robot, image = self.load_file_data(file)
                # except:
                #     fail_case += 1
                #     print("fail-case: ", fail_case)
                #     continue

                # scale the data
                for index, (standard_scaler, min_max_scalar) in enumerate(zip(self.tactile_standard_scaler, self.tactile_min_max_scalar)):
                    tactile[:, index] = standard_scaler.transform(tactile[:, index])
                    tactile[:, index] = min_max_scalar.transform(tactile[:, index])

                # create tactile_images:
                tactile_images = []
                for tactile_instances in tactile:
                    tactile_images.append(self.create_image(tactile_instances))
                np.array(tactile_images)

                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))

                # save images and save space:
                image_names = []
                for time_step in range(len(image)):
                    image_name = "image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
                    image_names.append(image_name)
                    np.save(path_save + image_name, image[time_step])

                for time_step in range(len(tactile) - self.sequence_length):
                    experiment_data_sequence    = experiment_number
                    robot_data_euler_sequence, tactile_data_sequence, tactile_images_sequence, image_name_sequence, time_step_data_sequence = tuple(zip(*[[robot[time_step + t], tactile[time_step + t], tactile_images[time_step + t], image_names[time_step + t], time_step + t] for t in range(self.sequence_length)]))
                    ref = []
                    for name, data_to_save in zip(['robot_data_euler_', 'tactile_data_sequence_', 'tactile_images_sequence', 'image_name_sequence_', 'experiment_number_', 'time_step_data_'], [robot_data_euler_sequence, tactile_data_sequence, tactile_images_sequence, image_name_sequence, experiment_data_sequence, time_step_data_sequence]):
                        np.save(path_save + name + str(index_to_save), data_to_save)
                        ref.append(name + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1

                progress_bar.set_description("file_number: {}, saved_pushes: {}".format(experiment_number, len(self.path_file)))
                progress_bar.update()
            self.save_map(path_save)

    def save_map(self, path, test=False):
        with open(path + '/map.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'tactile_images_sequence', 'image_name_sequence', 'experiment_number', 'time_steps'])
            for row in self.path_file:
                writer.writerow(row)

    def scale_data(self):
        files = self.files_train + self.files_test + self.files_test_2
        fail_case = 0
        for file in tqdm(files):
            # try:
            tactile, robot, image = self.load_file_data(file)
            # except:
            #     fail_case += 1
            #     print("fail-case: ", fail_case)
            #     continue
            self.full_data_tactile += list(tactile)
            self.full_data_robot += list(robot)

        self.full_data_robot = np.array(self.full_data_robot)
        self.full_data_tactile = np.array(self.full_data_tactile)

        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot[:, feature].reshape(-1, 1)) for feature in range(6)]
        self.tactile_standard_scaler = [preprocessing.StandardScaler().fit(self.full_data_tactile[:, feature]) for feature in range(3)]
        self.tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.tactile_standard_scaler[feature].transform(self.full_data_tactile[:, feature])) for feature in range(3)]

        self.save_scalars()

    def load_file_data(self, file):
        robot_state = np.array(pd.read_csv(file + '/robot_states.csv', header=None))
        xela_sensor = np.array(np.load(file + '/tactile_states.npy'))
        image_data = np.array(np.load(file + '/color_images.npy'))
        depth_data = np.array(np.load(file + '/depth_images.npy'))
        # image_data = np.array(np.load(file + '/color_image_top.npy'))
        # depth_data = np.array(np.load(file + '/depth_image_top.npy'))

        # convert orientation to euler, and remove column labels:
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state[1:]]).astype(float)

        # split tactile sensor into the three forces | find start value average for each force | find offsets for each taxel | take away start value average from the tactile data:
        tactile_data_split = [np.array(xela_sensor[:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
        tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
        tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]

        # Resize the image using PIL antialiasing method (Copied from CDNA data formatting)
        raw = []
        for k in range(len(image_data)):
            tmp = Image.fromarray(image_data[k])
            tmp = tmp.resize((image_height, image_width), Image.ANTIALIAS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((image_height, image_width, 3))
            tmp = tmp.astype(np.float32) / 255.0
            raw.append(tmp)
        image_data = np.array(raw)

        raw = []
        for k in range(len(depth_data)):
            image = np.expand_dims(depth_data[k], 2).repeat(3, axis=2)
            tmp = Image.fromarray(np.uint8(image))
            tmp = tmp.resize((image_height, image_width), Image.ANTIALIAS)
            tmp = np.frombuffer(tmp.tobytes(), dtype=np.uint8)
            tmp = tmp.reshape((image_height, image_width, 3))
            tmp = tmp.astype(np.float32) / 255.0
            raw.append(tmp[:, :, 0])
        depth_data = np.array(raw)
        # add depth channel to the back end of the image data:
        image_data = np.concatenate((image_data, np.expand_dims(depth_data, 3)), axis=3)

        return np.array(tactile_data), robot_task_space, image_data

    def create_image(self, tactile):
        # convert tactile data into an image:
        image = np.zeros((4, 4, 3), np.float32)
        index = 0
        for x in range(4):
            for y in range(4):
                image[x][y] = [tactile[0][index],
                               tactile[1][index],
                               tactile[2][index]]
                index += 1
        reshaped_image = cv2.resize(image.astype(np.float32), dsize=(self.image_height, self.image_width), interpolation=cv2.INTER_CUBIC)
        return reshaped_image


    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')
        if test_data_dir_2 != dataset_path:
            self.files_test_2 = glob.glob(test_data_dir_2 + '/*')
        else:
            self.files_test_2 = []

    def smooth_the_trial(self, tactile_data):
        for force in range(tactile_data.shape[1]):
            for taxel in range(tactile_data.shape[2]):
                tactile_data[:, force, taxel] = [None for i in range(3)] + list(self.smooth_func(tactile_data[:, force, taxel], 6)[3:-3]) + [None for i in range(3)]

        return tactile_data

    def smooth_func(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def save_scalars(self):
        # save the scalars
        dump(self.tactile_standard_scaler[0], open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', 'wb'))
        dump(self.tactile_standard_scaler[1], open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', 'wb'))
        dump(self.tactile_standard_scaler[2], open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[0], open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[1], open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', 'wb'))
        dump(self.tactile_min_max_scalar[2], open(scaler_out_dir + 'tactile_min_max_scalar.pkl', 'wb'))

        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'))
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'))
        dump(self.robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'))
        dump(self.robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'))
        dump(self.robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'))
        dump(self.robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'))

    def load_scalars(self):
        self.tactile_standard_scaler = []
        self.tactile_min_max_scalar = []
        self.robot_min_max_scalar = []

        self.tactile_standard_scaler.append(load(open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', "rb")))
        self.tactile_standard_scaler.append(load(open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', "rb")))
        self.tactile_standard_scaler.append(load(open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', "rb")))
        self.tactile_min_max_scalar.append(load(open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', "rb")))
        self.tactile_min_max_scalar.append(load(open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', "rb")))
        self.tactile_min_max_scalar.append(load(open(scaler_out_dir + 'tactile_min_max_scalar.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', "rb")))
        self.robot_min_max_scalar.append(load(open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', "rb")))

def main():
    df = data_formatter()
    df.load_file_names()
    # df.scale_data()
    df.load_scalars()
    df.create_map()


if __name__ == "__main__":
    main()