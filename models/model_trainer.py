# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import sys
import csv
import cv2
import numpy as np
import click
import random

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

# standard video and tactile prediction models:
from universal_networks.SVG import Model as SVG
from universal_networks.SVG_MMFM import Model as SVG_MMFM
from universal_networks.SVTG_SE import Model as SVTG_SE
from universal_networks.SPOTS_SVG_ACTP import Model as SPOTS_SVG_ACTP
from universal_networks.SPOTS_SVG_ACTP_STP import Model as SPOTS_SVG_ACTP_STP
from universal_networks.SPOTS_SVG_ACTP_STP_SAMESIZE import Model as SPOTS_SVG_ACTP_STP_SAMESIZE

# Tactile enhanced models:
from universal_networks.SVG_TE import Model as SVG_TE


class BatchGenerator:
    def __init__(self, train_percentage, batch_size, image_width, num_workers, device, features):
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_width
        self.num_workers = num_workers
        self.train_percentage = train_percentage
        self.features = features

    def load_full_data(self):
        dataset_train = FullDataSet(self.features, train=True, train_percentage=self.train_percentage)
        dataset_validate = FullDataSet(self.features, validation=True, train_percentage=self.train_percentage)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=24, pin_memory=True, pin_memory_device=self.device)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True, num_workers=24, pin_memory=True, pin_memory_device=self.device)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet(torch.utils.data.Dataset):
    def __init__(self, features, train=False, validation=False, train_percentage=1.0):
        self.features = features
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]

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
            tactile[:self.features["n_past"]] = 0.0

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


class UniversalModelTrainer:
    def __init__(self, features):
        self.features = features
        self.gain = 0.0
        self.stage = self.features["training_stages"][0]

        if self.features["model_name"] == "SVG": self.model = SVG(features)
        if self.features["model_name"] == "SVG_MMFM": self.model = SVG_MMFM(features)
        elif self.features["model_name"] == "SVTG_SE": self.model = SVTG_SE(features)
        elif self.features["model_name"] == "SVG_TE": self.model = SVG_TE(features)
        elif self.features["model_name"] == "SPOTS_SVG_ACTP": self.model = SPOTS_SVG_ACTP(features)
        elif self.features["model_name"] == "SPOTS_SVG_ACTP_STP": self.model = SPOTS_SVG_ACTP_STP(features)
        elif self.features["model_name"] == "SPOTS_SVG_ACTP_STP_SAMESIZE": self.model = SPOTS_SVG_ACTP_STP_SAMESIZE(features)

        self.model.initialise_model()

        BG = BatchGenerator(self.features["train_percentage"], self.features["batch_size"], self.features["image_width"], self.features["num_workers"], self.features["device"], self.features)
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()

        if self.features["criterion"] == "L1":
            self.criterion = nn.L1Loss()
        if self.features["criterion"] == "L2":
            self.criterion = nn.MSELoss()

    def train_full_model(self):
        plot_training_loss = []
        plot_validation_loss = []
        plot_training_save_points = []

        best_val_loss = 100.0
        early_stop_clock = 0
        previous_val_mean_loss = 100.0

        progress_bar = tqdm(range(0, self.features["epochs"]), total=(self.features["epochs"]*len(self.train_full_loader)))
        for epoch in progress_bar:
            # set the stages:
            if epoch <= self.features["training_stages_epochs"][0]: self.stage = self.features["training_stages"][0]
            elif epoch <= self.features["training_stages_epochs"][1]: self.stage = self.features["training_stages"][1]
            elif epoch <= self.features["training_stages_epochs"][2]: self.stage = self.features["training_stages"][2]

            self.model.set_train()

            # TRAINING:
            epoch_mae_losses, epoch_kld_losses, epoch_mae_tactile_losses= 0.0, 0.0, 0.0
            mean_mae, mean_kld, mean_mae_tactile = 100.0, 100.0, 100.0
            for index, batch_features in enumerate(self.train_full_loader):
                if batch_features[1].shape[0] == self.features["batch_size"]:                                               # messes up the models when running with a batch that is not complete
                    mae, kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=False)              # run the model
                    epoch_mae_losses += mae.item()
                    mean_mae = epoch_mae_losses / (index + 1)
                    if kld != 100:
                        epoch_kld_losses += float(kld.item())
                        mean_kld = epoch_kld_losses / (index + 1)
                    if mae_tactile != 100:
                        epoch_mae_tactile_losses += mae_tactile.item()
                        mean_mae_tactile = epoch_kld_losses / (index + 1)

                    progress_bar.set_description("epoch: {}, ".format(epoch) + "MAE: {:.4f}, ".format(float(mae.item())) + "kld: {:.4f}, ".format(float(kld)) + "mean MAE: {:.4f}, ".format(float(mean_mae)) + "mean kld: {:.4f}, ".format(float(mean_kld))  + "mean tac mae: {:.4f}, ".format(float(mean_mae_tactile)))
                    progress_bar.update()

            plot_training_loss.append([mean_mae, mean_kld, mean_mae_tactile])

            # VALIDATION:
            self.model.set_test()
            val_mae_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    if batch_features[1].shape[0] == self.features["batch_size"]:
                        val_mae, val_kld, mae_tactile, predictions = self.format_and_run_batch(batch_features, test=True)
                        val_mae_losses += val_mae.item()

            val_mae_loss = val_mae_losses / index__
            plot_validation_loss.append(val_mae_loss)
            print("Validation mae: {:.4f}, ".format(val_mae_loss))

            # save the train/validation performance data
            np.save(self.features["model_save_path"] + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(self.features["model_save_path"] + "plot_training_loss", np.array(plot_training_loss))

            # Early stopping:
            if previous_val_mean_loss < val_mae_loss:
                early_stop_clock += 1
            else:
                if best_val_loss > val_mae_loss:
                    print("saving model")
                    plot_training_save_points.append(epoch+1)
                    self.model.save_model()
                    best_val_loss = val_mae_loss
                early_stop_clock = 0
            previous_val_mean_loss = val_mae_loss

            np.save(self.features["model_save_path"] + "plot_training_loss", np.array(plot_training_save_points))
            lines = list(plot_training_save_points)
            with open (self.features["model_save_path"] + "plot_training_loss.txt", 'w') as f:
                for line in lines:
                    f.write("saved after epoch: " + str(line))
                    f.write('\n')

            if early_stop_clock > self.features["early_stop_clock"]:
                print("early stop clock triggered at {} consecutive regressive validation scores".format(early_stop_clock))
                break

    def format_and_run_batch(self, batch_features, test):
        mae, kld, mae_tactile, predictions = 100, 100, 100, 100
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

        elif self.features["model_name"] == "SPOTS_SVG_ACTP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP" or self.features["model_name"] == "SPOTS_SVG_ACTP_STP_SAMESIZE":
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            tactile = torch.flatten(batch_features[2].permute(1, 0, 2, 3), start_dim=2).to(self.features["device"])
            mae, kld, mae_tactile, predictions, tactile = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)

        elif self.features["model_name"] == "SVG_TE":
            images  = batch_features[1].permute(1, 0, 4, 3, 2).to(self.features["device"])
            tactile = torch.flatten(batch_features[2].permute(1, 0, 2, 3), start_dim=2).to(self.features["device"])
            action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(self.features["device"])
            mae, kld, predictions = self.model.run(scene=images, tactile=tactile, actions=action, gain=self.gain, test=test, stage=self.stage)

        return mae, kld, mae_tactile, predictions


@click.command()
@click.option('--model_name',                  type=click.Path(),       default = "ACTVP",                help='Set name for prediction model, SVG, SVTG_SE, SPOTS_SVG_ACTP, SVG_TC, SVG_MMFM')
@click.option('--batch_size',                  type=click.INT,          default = 248,                  help='Batch size for training.')
@click.option('--lr',                          type=click.FLOAT,        default = 0.0001,               help = "learning rate")
@click.option('--beta1',                       type=click.FLOAT,        default = 0.9,                  help = "Beta gain")
@click.option('--optimizer',                   type=click.Path(),       default = 'adam',               help = "what optimiser to use - only adam available currently")
@click.option('--seed',                        type=click.INT,          default = 1,                    help = "")
@click.option('--image_width',                 type=click.INT,          default = 64,                   help = "Size of scene image data")
@click.option('--dataset',                     type=click.Path(),       default = '---',                help = "name of the dataset")
@click.option('--n_past',                      type=click.INT,          default = 2,                    help = "context sequence length")
@click.option('--n_future',                    type=click.INT,          default = 5,                    help = "time horizon sequence length")
@click.option('--n_eval',                      type=click.INT,          default = 7,                    help = "sum of context and time horizon")
@click.option('--prior_rnn_layers',            type=click.INT,          default = 3,                    help = "number of LSTMs in the prior model")
@click.option('--posterior_rnn_layers',        type=click.INT,          default = 3,                    help = "number of LSTMs in the posterior model")
@click.option('--predictor_rnn_layers',        type=click.INT,          default = 4,                    help = "number of LSTMs in the frame predictor model")
@click.option('--state_action_size',           type=click.INT,          default = 12,                   help = "size of action conditioning data")
@click.option('--z_dim',                       type=click.INT,          default = 10,                   help = "number of latent variables to estimate")
@click.option('--beta',                        type=click.FLOAT,        default = 0.0001,               help = "beta gain")
@click.option('--epochs',                      type=click.INT,          default = 100,                  help = "number of epochs to run for ")
@click.option('--train_percentage',            type=click.FLOAT,        default = 0.9,                  help = "")
@click.option('--validation_percentage',       type=click.FLOAT,        default = 0.1,                  help = "")
@click.option('--criterion',                   type=click.Path(),       default = "L1",                 help = "")
@click.option('--tactile_size',                type=click.INT,          default = 0,                    help = "size of tacitle frame - 48, if no tacitle data set to 0")
@click.option('--g_dim',                       type=click.INT,          default = 256,                  help = "size of encoded data for input to prior")
@click.option('--rnn_size',                    type=click.INT,          default = 256,                  help = "size of encoded data for input to frame predictor (g_dim = rnn-size)")
@click.option('--channels',                    type=click.INT,          default = 3,                    help = "input channels")
@click.option('--out_channels',                type=click.INT,          default = 3,                    help = "output channels")
@click.option('--training_stages',             type=click.Path(),       default = "",                   help = "define the training stages - if none leave blank - available: 3part")
@click.option('--training_stages_epochs',      type=click.Path(),       default = "50,75,125",          help = "define the end point of each training stage")
@click.option('--num_workers',                 type=click.INT,          default = 24,                   help = "number of workers used by the data loader")
@click.option('--model_save_path',             type=click.Path(),       default = "/home/willow/Robotics/SPOTS/models/saved_models/",                help = "")
@click.option('--train_data_dir',              type=click.Path(),       default = "/home/willow/Robotics/datasets/PRI/household_object_dataset/Dataset_2c_5p/train_formatted/",              help = "")
@click.option('--scaler_dir',                  type=click.Path(),       default = "/home/willow/Robotics/datasets/PRI/household_object_dataset/Dataset_2c_5p/scalers/",              help = "")
@click.option('--model_name_save_appendix',    type=click.Path(),       default = "",                   help = "What to add to the save file to identify the model as a specific subset (_1c= 1 conditional frame, GTT=groundtruth tactile data)")
@click.option('--tactile_encoder_hidden_size', type=click.INT,          default = 0,                    help = "Size of hidden layer in tactile encoder, 200")
@click.option('--tactile_encoder_output_size', type=click.INT,          default = 0,                    help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_test',              type=click.BOOL,         default = False,                help = "if you would like to train for occlusion")
@click.option('--occlusion_gain_per_epoch',    type=click.FLOAT,        default = 0.05,                 help = "increasing size of the occlusion block per epoch 0.1=(0.1 x MAX) each epoch")
@click.option('--occlusion_start_epoch',       type=click.INT,          default = 35,                   help = "size of output layer from tactile encoder, 100")
@click.option('--occlusion_max_size',          type=click.FLOAT,        default = 0.4,                  help = "max size of the window as a % of total size (0.5 = 50% of frame (32x32 squares in ))")
@click.option('--using_depth_data',            type=click.BOOL,         default = False,                help = "if the image has depth included, set to True")
@click.option('--using_tactile_images',        type=click.BOOL,         default = False,                help = "if the image has depth included, set to True")
@click.option('--early_stop_clock',            type=click.INT,          default = 5,                    help = "if the image has depth included, set to True")
@click.option('--device',                      type=click.Path(),       default = "cuda:0",             help = "if the image has depth included, set to True")
@click.option('--save_intervals',              type=click.Path(),       default = "0.1",                help = "how often to save a model of the data ")
@click.option('--tactile_random',              type=click.BOOL,         default = False,                help = 'if you want to provide random tactile data to the model instead of real tactile data')
@click.option('--image_random',                type=click.BOOL,         default = False,                help = 'if you want to provide random scene data to the model instead of real scene data')
def main(model_name, batch_size, lr, beta1, optimizer, seed, image_width, dataset,
         n_past, n_future, n_eval, prior_rnn_layers, posterior_rnn_layers, predictor_rnn_layers, state_action_size,
         z_dim, beta, epochs, train_percentage, validation_percentage,
         criterion, tactile_size, g_dim, rnn_size, channels, out_channels, training_stages, training_stages_epochs,
         num_workers, model_save_path, train_data_dir, scaler_dir, model_name_save_appendix, tactile_encoder_hidden_size,
         tactile_encoder_output_size, occlusion_test, occlusion_gain_per_epoch, occlusion_start_epoch, occlusion_max_size,
         using_depth_data, using_tactile_images, early_stop_clock, device, save_intervals, tactile_random, image_random):

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global data_map
    global tdd
    global uti
    global udd
    global ir
    global tr
    data_map = []
    tdd = train_data_dir
    uti = using_tactile_images
    udd = using_depth_data
    tr = tactile_random
    ir = image_random

    with open(train_data_dir + 'map.csv', 'r') as f:  # rb
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            data_map.append(row)

    # unique save title:
    model_save_path = model_save_path + model_name
    try:
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass
    try:
        model_save_path = model_save_path + "/model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
        os.mkdir(model_save_path)
    except FileExistsError or FileNotFoundError:
        pass

    model_dir = model_save_path
    data_root = train_data_dir

    if model_name == "SVG" or model_name == "SVG_TC" or model_name == "SVG_MMFM" or model_name == "VG" or model_name == "VG_MMMM" or model_name == "SVG_occ" or model_name == "SVG_TC_occ":
        g_dim = 256  # 128
        rnn_size = 256
        if using_depth_data:
            channels = 4
            out_channels = 4
        else:
            channels = 3
            out_channels = 3

        training_stages = [""]
        training_stages_epochs = [epochs]
    elif model_name == "SVTG_SE" or model_name == "SVTG_SE_occ" or model_name == "VTG_SE":
        if model_name_save_appendix== "large":
            g_dim = 256 * 4
            rnn_size = 256 * 4
        else:
            g_dim = 256 * 2
            rnn_size = 256 * 2
        if using_depth_data:
            channels = 7
            out_channels = 7
        else:
            channels = 6
            out_channels = 6
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = [image_width, image_width]
    elif model_name == "SPOTS_SVG_ACTP" or model_name == "SPOTS_VG_ACTP" or model_name == "SPOTS_SVG_ACTP_occ" or model_name == "SPOTS_SVG_PTI_ACTP"  or model_name == "SPOTS_SVG_ACTP_STP":
        g_dim = 256
        rnn_size = 256
        if using_depth_data:
            channels = 4
            out_channels = 4
        else:
            channels = 3
            out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = 48
        if training_stages == "3part":
            g_dim = 256
            rnn_size = 256
            if using_depth_data:
                channels = 4
                out_channels = 4
            else:
                channels = 3
                out_channels = 3
            training_stages = ["scene_only", "tactile_loss_plus_scene_fixed", "scene_loss_plus_tactile_gradual_increase"]
            training_stages_epochs = [35, 65, 150]
            tactile_size = 48
            epochs = training_stages_epochs[-1] + 1

    elif model_name == "SVG_TC_TE" or model_name == "SVG_TC_TE_occ" or model_name == "SVG_TE":
        g_dim = 256
        rnn_size = 256
        if using_depth_data:
            channels = 4
            out_channels = 4
        else:
            channels = 3
            out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = 48
        tactile_encoder_hidden_size = 200
        tactile_encoder_output_size = 100

    elif model_name == "SPOTS_SVG_ACTP_STP_SAMESIZE":
        g_dim = 209
        rnn_size = 209
        channels = 3
        out_channels = 3
        training_stages = [""]
        training_stages_epochs = [epochs]
        tactile_size = 48

    features = {"lr": lr, "beta1": beta1, "batch_size": batch_size, "model_dir": model_dir, "data_root": data_root, "optimizer": optimizer, "seed": seed,
                "image_width": image_width, "channels": channels, "out_channels": out_channels, "dataset": dataset, "n_past": n_past, "n_future": n_future, "n_eval": n_eval, "rnn_size": rnn_size, "prior_rnn_layers": prior_rnn_layers,
                "posterior_rnn_layers": posterior_rnn_layers, "predictor_rnn_layers": predictor_rnn_layers, "state_action_size": state_action_size, "z_dim": z_dim, "g_dim": g_dim, "beta": beta,
                "epochs": epochs, "train_percentage": train_percentage, "validation_percentage": validation_percentage, "criterion": criterion, "model_name": model_name,
                "train_data_dir": train_data_dir, "scaler_dir": scaler_dir, "device": device, "training_stages":training_stages, "training_stages_epochs": training_stages_epochs, "tactile_size":tactile_size, "num_workers":num_workers,
                "model_save_path":model_save_path, "model_name_save_appendix":model_name_save_appendix, "tactile_encoder_hidden_size":tactile_encoder_hidden_size, "tactile_encoder_output_size":tactile_encoder_output_size,
                "occlusion_test": occlusion_test, "occlusion_gain_per_epoch":occlusion_gain_per_epoch, "occlusion_start_epoch":occlusion_start_epoch, "occlusion_max_size":occlusion_max_size,
                "using_depth_data": using_depth_data, "using_tactile_images": using_tactile_images, "early_stop_clock": early_stop_clock, "tactile_random": tactile_random, "image_random":image_random}

    # save features
    w = csv.writer(open(model_save_path + "/features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

    print(features)

    UMT = UniversalModelTrainer(features)
    UMT.train_full_model()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    main()










