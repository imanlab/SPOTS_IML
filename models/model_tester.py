# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import sys
import csv
import cv2
import click
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

# standard video and tactile prediction models:
from universal_networks.SVG import Model as SVG
from universal_networks.SVTG_SE import Model as SVTG_SE
from universal_networks.SVG_MMFM import Model as SVG_MMFM
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP
from universal_networks.SPOTS_SVG_ACTP_STP import Model as SPOTS_SVG_ACTP_STP
from universal_networks.SPOTS_SVG_ACTP_STP_SAMESIZE import Model as SPOTS_SVG_ACTP_STP_SAMESIZE

# Tactile enhanced models:
from universal_networks.SVG_TE import Model as SVG_TE

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)

class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)

class BatchGenerator:
    def __init__(self, batch_size, image_width, features):
        self.batch_size = batch_size
        self.image_size = image_width
        self.features = features

    def load_full_data(self):
        dataset_test = FullDataSet(self.features)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.data_map = []
        return test_loader

class FullDataSet(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features
        self.samples = data_map[1:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot = np.load(tdd + value[0]).astype(np.float32)

        if uti == True:
            tactile = np.load(tdd + value[2]).astype(np.float32)
        else:
            tactile = np.load(tdd + value[1]).astype(np.float32)
        if tr:
            tactile = np.random.normal(0, 1, tactile.shape).astype(np.float32)
            tactile = (tactile - tactile.min()) / (tactile.max() - tactile.min())
            # tactile[:self.features["n_past"]] = 0.0
        if tz:
            tactile[:] = 0.5

        image = []
        for image_name in np.load(tdd + value[3]):
            if udd:
                image.append(np.load(tdd + image_name).astype(np.float32))
            else:
                image.append(np.load(tdd + image_name).astype(np.float32)[:, :, 0:3])
        image= np.array(image)
        if ir:
            image[:self.features["n_past"]] = 0.0

        return [torch.tensor(robot), torch.tensor(image), torch.tensor(tactile)]


class UniversalTester:
    def __init__(self, features):
        self.list_of_p_measures = ["MAE", "MSE", "PSNR", "SSIM", "MAE_last", "MSE_last", "PSNR_last", "SSIM_last"]

        saved_model = torch.load(features["model_save_path"] + features["model_save_name"] + features["model_name_save_appendix"])
        self.features = saved_model['features']

        if features["model_name"] == "SVG": self.model = SVG(features = self.features)
        elif features["model_name"] == "SVTG_SE": self.model = SVTG_SE(features = self.features)
        elif features["model_name"] == "SVG_MMFM": self.model = SVG_MMFM(features = self.features)
        elif features["model_name"] == "SPOTS_SVG_ACTP": self.model = SPOTS_SVG_ACTP(features = self.features)
        elif features["model_name"] == "SVG_TE": self.model = SVG_TE(features = self.features)
        elif features["model_name"] == "SPOTS_SVG_ACTP_STP": self.model = SPOTS_SVG_ACTP_STP(features = self.features)
        elif features["model_name"] == "SPOTS_SVG_ACTP_STP_SAMESIZE": self.model = SPOTS_SVG_ACTP_STP_SAMESIZE(features = self.features)

        self.test_features = features
        self.model.load_model(full_model=saved_model)
        saved_model = None

        BG = BatchGenerator(self.features["batch_size"], self.features["image_width"], self.features)
        self.test_full_loader = BG.load_full_data()

        self.test_model()

    def test_model(self):
        batch_losses = []
        batch_tactile_losses = []
        self.model.set_test()
        for index, batch_features in enumerate(self.test_full_loader):
            print(str(index) + "\r")

            groundtruth_scene, predictions_scene, groundtruth_tactile, prediction_tactile = self.format_and_run_batch(batch_features, test=True)              # run the model
            if self.test_features["quant_analysis"] == True: #  and prediction_tactile == 100:
                batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[self.features["n_past"]:], prediction_tactile))
            if self.test_features["quant_tactile_analysis"] == True:
                print(groundtruth_tactile)
                batch_tactile_losses.append(self.calculate_tactile_scores(prediction_tactile, groundtruth_tactile[self.features["n_past"]]))

            batches_to_save = [0,1,2,3]
            if self.test_features["qual_analysis"] == True and index in batches_to_save:
                print("here, index: " + str(index))
                self.save_images(predictions_scene, groundtruth_scene[self.features["n_past"]:], index)
            
            if self.test_features["qual_tactile_analysis"] == True and index in batches_to_save:
                self.save_tactile(prediction_tactile, groundtruth_tactile[self.features["n_past"]:], index)

        if self.test_features["quant_analysis"] == True:
            print(batch_losses)
            batch_losses = np.array(batch_losses)
            
            if self.test_features["seen"]: data_save_path_append = "seen_"
            else:                          data_save_path_append = "unseen_"

            np.save(self.test_features["data_save_path"] + data_save_path_append + "test_loss_scores_alltimesteps.npy", batch_losses)
            lines = [[float(i) for i in j] for j in batch_losses[0][2]]
            with open (self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores_alltimesteps.txt", 'w') as f:
                for index, line in enumerate(lines):
                    f.write(self.list_of_p_measures[index] + ": " + str(line))
                    f.write('\n')

            full_losses = [sum(batch_losses[:,0,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]
            last_ts_losses = [sum(batch_losses[:,1,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]

            full_losses = [float(i) for i in full_losses]
            last_ts_losses = [float(i) for i in last_ts_losses]

            np.save(self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.npy", batch_losses)
            lines = full_losses + last_ts_losses
            with open (self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.txt", 'w') as f:
                for index, line in enumerate(lines):
                    f.write(self.list_of_p_measures[index] + ": " + str(line))
                    f.write('\n')

        if self.test_features["quant_tactile_analysis"] == True:
            print(batch_tactile_losses)
            batch_tactile_losses = np.array(batch_tactile_losses)

            if self.test_features["seen"]: data_save_path_append = "seen_"
            else:                          data_save_path_append = "unseen_"

            np.save(self.test_features["data_save_path"] + data_save_path_append + "test_tactile_loss_scores_alltimesteps.npy", batch_tactile_losses)
            lines = [[float(i) for i in j] for j in batch_tactile_losses[0][2]]
            with open (self.test_features["data_save_path"] + data_save_path_append  + "test_tactile_loss_scores_alltimesteps.txt", 'w') as f:
                for index, line in enumerate(lines):
                    f.write(self.list_of_p_measures[index] + ": " + str(line))
                    f.write('\n')

            full_losses = [sum(batch_tactile_losses[:,0,i]) / batch_tactile_losses.shape[0] for i in range(batch_tactile_losses.shape[2])]
            last_ts_losses = [sum(batch_tactile_losses[:,1,i]) / batch_tactile_losses.shape[0] for i in range(batch_tactile_losses.shape[2])]

            full_losses = [float(i) for i in full_losses]
            last_ts_losses = [float(i) for i in last_ts_losses]

            np.save(self.test_features["data_save_path"] + data_save_path_append  + "test_tactile_loss_scores.npy", batch_tactile_losses)
            lines = full_losses + last_ts_losses
            with open (self.test_features["data_save_path"] + data_save_path_append  + "test_tactile_loss_scores.txt", 'w') as f:
                for index, line in enumerate(lines):
                    f.write(self.list_of_p_measures[index] + ": " + str(line))
                    f.write('\n')


    def save_tactile(self, prediction_tactile, groundtruth_tactile, index):
        try:
            os.mkdir(self.test_features["data_save_path"] + "push_" + str(index))
        except FileExistsError or FileNotFoundError:
            pass

        predictions_tactile = prediction_tactile.cpu().detach().numpy()
        groundtruth_tactile = groundtruth_tactile.cpu().detach().numpy()

        for j in range(0, predictions_tactile.shape[1]):
            for i in range(0, predictions_tactile.shape[0]):
                np.save(self.test_features["data_save_path"] + "push_" + str(index) + "/PR_tactile_batch_" + str(j) + "_timestep_" + str(i) + ".npy", predictions_tactile[i,j])
                np.save(self.test_features["data_save_path"] + "push_" + str(index) + "/GT_tactile_batch_" + str(j) + "_timestep_" + str(i) + ".npy", groundtruth_tactile[i,j])

    def save_images(self, predictions_scene, groundtruth_scene, index):
        try:
            os.mkdir(self.test_features["data_save_path"] + "push_" + str(index))
        except FileExistsError or FileNotFoundError:
            pass

        predictions_scene = predictions_scene.cpu().detach().numpy()
        groundtruth_scene = groundtruth_scene.cpu().detach().numpy()
        for j in range(0, predictions_scene.shape[1]):
            for i in range(0, predictions_scene.shape[0]):
                im = predictions_scene[i][j].T * 255
                im = Image.fromarray(im[:,:,::-1].astype(np.uint8))
                im.save(self.test_features["data_save_path"] + "push_" + str(index) + "/PR_batch_" + str(j) + "_timestep_" + str(i) + ".png")

                im = groundtruth_scene[i][j].T * 255
                im = Image.fromarray(im[:,:,::-1].astype(np.uint8))
                im.save(self.test_features["data_save_path"] + "push_" + str(index) + "/GT_batch_" + str(j) + "_timestep_" + str(i) + ".png")

    def calculate_tactile_scores(self, prediction_tactile, groundtruth_tactile):
        tactile_losses_full, tactile_losses_last, tactile_losses_alltimesteps = [],[],[]
        for criterion in [nn.L1Loss(), nn.MSELoss()]: #, PSNR(), SSIM(window_size=self.features["image_width"])]:
            tactile_batch_loss_full = []
            for i in range(prediction_tactile.shape[0]):
                tactile_batch_loss_full.append(criterion(prediction_tactile[i], groundtruth_tactile[i]).cpu().detach().data)

            tactile_losses_alltimesteps.append(tactile_batch_loss_full)
            tactile_losses_full.append(sum(tactile_batch_loss_full) / len(tactile_batch_loss_full))
            tactile_losses_last.append(tactile_batch_loss_full[-1])

        return tactile_losses_full, tactile_losses_last, tactile_losses_alltimesteps

    def calculate_scores(self, prediction_scene, groundtruth_scene, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last, scene_losses_alltimesteps = [],[],[]
        for criterion in [nn.L1Loss(), nn.MSELoss(), PSNR(), SSIM(window_size=self.features["image_width"])]:  #, SSIM(window_size=self.image_width)]:
            scene_batch_loss_full = []
            for i in range(prediction_scene.shape[0]):
                scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

            scene_losses_alltimesteps.append(scene_batch_loss_full)
            scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
            scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_losses_full, scene_losses_last, scene_losses_alltimesteps]

    def format_and_run_batch(self, batch_features, test):
        # # """"""""""""""""""""""""" REMOVE THIS !!!!!!!!!
        self.features["n_eval"] = 17
        # # """"""""""""""""""""""""" REMOVE THIS !!!!!!!!!

        cut_required = False
        if batch_features[1].shape[0] != self.features["batch_size"]:
            cut_required = batch_features[1].shape[0]
            # add zeros of the data to reach the batch_size:
            if self.features["using_tactile_images"]:
                batch_features[2] = torch.cat((batch_features[2], torch.zeros(self.features["batch_size"] - batch_features[2].shape[0], self.features["n_eval"], self.features["image_width"], self.features["image_width"], int(self.features["channels"] / 2))), dim=0)
                batch_features[1] = torch.cat((batch_features[1], torch.zeros(self.features["batch_size"] - batch_features[1].shape[0], self.features["n_eval"], self.features["image_width"], self.features["image_width"], int(self.features["channels"] / 2))), dim=0)
            else:
                batch_features[2] = torch.cat((batch_features[2], torch.zeros(self.features["batch_size"] - batch_features[2].shape[0], self.features["n_eval"], batch_features[2].shape[2], batch_features[2].shape[3])), dim=0)
                batch_features[1] = torch.cat((batch_features[1], torch.zeros(self.features["batch_size"] - batch_features[1].shape[0], self.features["n_eval"], self.features["image_width"], self.features["image_width"], self.features["channels"])), dim=0)
            batch_features[0] = torch.cat((batch_features[0], torch.zeros(self.features["batch_size"] - batch_features[0].shape[0], self.features["n_eval"], int(self.features["state_action_size"] / 2))), dim=0)

        
        mae, kld, mae_tactile, predictions = 100, 100, 100, 100
        tactile_predictions, tactile = 100, 100
        if self.features["model_name"] == "SVG" or self.features["model_name"] == "SVG_MMFM":
            images = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            mae, kld, predictions = self.model.run(scene=images, actions=action, test=test)

        elif self.features["model_name"] == "SVTG_SE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            tactile = batch_features[2].permute(1, 0, 4, 3, 2).to(self.features["device"])
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            scene_and_touch = torch.cat((tactile, images), 2)
            mae, kld, predictions = self.model.run(scene_and_touch=scene_and_touch, actions=action, test=test)
            predictions = predictions[:,:,3:6,:,:]
            tactile_predictions = predictions[:,:,0:3,:,:]

        elif self.features["model_name"] == "SPOTS_SVG_ACTP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP_SAMESIZE":
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            tactile = torch.flatten(batch_features[2].permute(1, 0, 2, 3), start_dim=2).to(self.features["device"])
            mae, kld, mae_tactile, predictions, tactile_predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=0, test=test, stage="")

        elif self.features["model_name"] == "SVG_TE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            tactile = torch.flatten(batch_features[2].permute(1, 0, 2, 3), start_dim=2).to(self.features["device"])
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            mae, kld, predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=0, test=test)

        if cut_required:
            predictions = predictions[:, :cut_required]
            images = images[:,:cut_required]
            if self.features["model_name"] == "SPOTS_SVG_ACTP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP_SAMESIZE":
                tactile = tactile[:cut_required]
            if self.features["model_name"] == "SVTG_SE":
                tactile = tactile[:cut_required]

        return images, predictions, tactile, tactile_predictions


@click.command()
@click.option('--model_name',                   type=click.Path(),  default = "SVG",        help='Set name for prediction model, SVG, SVTG_SE, SVG_TC, SVG_TC_TE, SPOTS_SVG_ACTP')
@click.option('--model_stage',                  type=click.Path(),  default = "",           help='what stage of model should you test? BEST, stage1 etc.')
@click.option('--tactile_random',               type=click.BOOL,    default = False,        help='if you want to provide random tactile data to the model instead of real tactile data')
@click.option('--tactile_zero',                 type=click.BOOL,    default = False,        help='if you want to provide random tactile data to the model instead of real tactile data')
@click.option('--image_random',                 type=click.BOOL,    default = False,        help='if you want to provide random scene data to the model instead of real scene data')
@click.option('--model_folder_name',            type=click.Path(),  default = "/home/willow/Robotics/SPOTS/models/saved_models/SVG/marked_object_dataset/model_12_11_2022_17_18/", help='Folder name where the model is stored')
@click.option('--quant_analysis',               type=click.BOOL,    default = True,         help='Perform quantitative analysis on the test data')
@click.option('--qual_analysis',                type=click.BOOL,    default = False,        help='Perform qualitative analysis on the test data')
@click.option('--qual_tactile_analysis',        type=click.BOOL,    default = False,        help='Perform qualitative analysis on the test tactile data')
@click.option('--quant_tactile_analysis',       type=click.BOOL,    default = False,        help='Perform quantitative analysis on the test tactile data')
@click.option('--test_sample_time_step',        type=click.Path(),  default = "[1, 2, 10]", help='which time steps in prediciton sequence to calculate performance metrics for.')
@click.option('--model_name_save_appendix',     type=click.Path(),  default = "",           help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--test_data_dir',                type=click.Path(),  default = "/home/willow/Robotics/datasets/PRI/MarkedHeavyBox/Dataset_2c_15p/", help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--scaler_dir',                   type=click.Path(),  default = "/home/willow/Robotics/datasets/PRI/MarkedHeavyBox/Dataset_2c_5p/scalar/", help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--using_tactile_images',         type=click.BOOL,    default = False,        help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--using_depth_data',             type=click.BOOL,    default = False,        help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--seen',                         type=click.BOOL,    default = False,        help = "What to add to the save file to identify the model as a specific subset, _1c")
@click.option('--device',                       type=click.Path(),  default = "cuda:1",     help = "What to add to the save file to identify the model as a specific subset, _1c")
def main(model_name, model_stage, tactile_random, tactile_zero, image_random, model_folder_name, quant_analysis, qual_analysis, qual_tactile_analysis, quant_tactile_analysis, test_sample_time_step, model_name_save_appendix, test_data_dir, scaler_dir, using_tactile_images, using_depth_data, seen, device):
    model_save_path = model_folder_name
    test_data_dir   = test_data_dir
    scaler_dir      = scaler_dir
    if tactile_random:
        data_save_path  = model_save_path + "performance_data_tactile_random/"
    elif image_random:
        data_save_path  = model_save_path + "performance_data_image_random/"
    elif tactile_zero:
        data_save_path  = model_save_path + "performance_data_tactile_zero/"
    else:
        data_save_path  = model_save_path + "performance_data_17_tactile/"

    model_save_name = model_name + "_model"

    if "household_object_dataset" in test_data_dir:
        if seen:
            test_data_dir += "test_seen_formatted/"
        else:
            test_data_dir += "test_unseen_formatted/"
    elif "MarkedHeavyBox" in test_data_dir:
        if seen:
            test_data_dir += "test_formatted/"
        else:
            test_data_dir += "test_examples_formatted/"

    try:
        os.mkdir(data_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    print(model_save_name)

    global tz
    global tr
    global data_map
    global tdd
    global uti
    global udd
    global ir
    data_map = []
    tdd = test_data_dir
    uti = using_tactile_images
    udd = using_depth_data
    tr = tactile_random
    tz = tactile_zero
    ir = image_random

    with open(test_data_dir + 'map.csv', 'r') as f:  # rb
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            data_map.append(row)

    features = {"model_name":model_name, "model_stage":model_stage, "model_folder_name":model_folder_name,
                "quant_analysis":quant_analysis, "qual_analysis":qual_analysis, "model_save_name":model_save_name,
                "qual_tactile_analysis":qual_tactile_analysis, "quant_tactile_analysis":quant_tactile_analysis, "test_sample_time_step":test_sample_time_step,
                "model_name_save_appendix":model_name_save_appendix, "test_data_dir":test_data_dir, "scaler_dir":scaler_dir,
                "using_tactile_images":using_tactile_images, "using_depth_data":using_depth_data, "model_save_path":model_save_path,
                "data_save_path": data_save_path, "seen": seen, "device": device, "tactile_random": tactile_random, 
                "image_random": image_random}

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # use gpu if available

    MT = UniversalTester(features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main()
