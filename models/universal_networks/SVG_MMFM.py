# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
# from utils import init_weights
import universal_networks.utils as utility_prog
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.features = features

        if self.features["optimizer"] == "adam" or self.features["optimizer"] == "Adam":
            self.optimizer = optim.Adam

        if self.features["criterion"] == "L1":
            self.criterion = nn.L1Loss().to(self.features["device"])
        if self.features["criterion"] == "L2":
            self.criterion = nn.MSELoss().to(self.features["device"])


    def load_model(self, full_model):
        self.frame_predictor = full_model["frame_predictor"].to(self.features["device"])
        self.posterior = full_model["posterior"].to(self.features["device"])
        self.prior = full_model["prior"].to(self.features["device"])
        self.encoder = full_model["encoder"].to(self.features["device"])
        self.decoder = full_model["decoder"].to(self.features["device"])
        self.MMFM = full_model["MMFM"].to(self.features["device"])

    def initialise_model(self):
        from universal_networks.lstm import lstm
        from universal_networks.lstm import gaussian_lstm
        self.frame_predictor = lstm(self.features["g_dim"] + self.features["z_dim"] + self.features["state_action_size"], self.features["g_dim"], self.features["rnn_size"], self.features["predictor_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.posterior = gaussian_lstm(self.features["g_dim"], self.features["z_dim"], self.features["rnn_size"], self.features["posterior_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.prior = gaussian_lstm(self.features["g_dim"], self.features["z_dim"], self.features["rnn_size"], self.features["prior_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.frame_predictor.apply(utility_prog.init_weights).to(self.features["device"])
        self.posterior.apply(utility_prog.init_weights).to(self.features["device"])
        self.prior.apply(utility_prog.init_weights).to(self.features["device"])

        import universal_networks.dcgan_64 as model
        self.encoder = model.encoder(self.features["g_dim"], self.features["channels"])
        self.decoder = model.decoder(self.features["g_dim"], self.features["channels"])
        self.encoder.apply(utility_prog.init_weights).to(self.features["device"])
        self.decoder.apply(utility_prog.init_weights).to(self.features["device"])

        # Added the MMFM for fair comparison of network parameters:
        self.MMFM = model.MMFM_scene(self.g_dim, self.g_dim, self.channels)
        self.MMFM.apply(utility_prog.init_weights).to(self.features["device"])

        self.frame_predictor_optimizer = self.optimizer(self.frame_predictor.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.MMFM_optimizer = self.optimizer(self.MMFM.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))

    def save_model(self):
        torch.save({'encoder': self.encoder, 'decoder': self.decoder, 'frame_predictor': self.frame_predictor, 'MMFM': self.MMFM,
                    'posterior': self.posterior, 'prior': self.prior, 'features': self.features}, 
                    self.features["model_dir"] + self.features["model_name"] + "_model" + self.features["model_name_save_appendix"])

    def set_train(self):
        self.frame_predictor.train()
        self.posterior.train()
        self.prior.train()
        self.encoder.train()
        self.decoder.train()
        self.MMFM.train()

    def set_test(self):
        self.frame_predictor.eval()
        self.posterior.eval()
        self.prior.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.MMFM.eval()

    def run(self, scene, actions, test=False):
        mae, kld = 0, 0
        outputs = []

        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.MMFM.zero_grad()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        state = actions[0].to(self.features["device"])
        for index, (sample_sscene, sample_action) in enumerate(zip(scene[:-1], actions[1:])):
            state_action = torch.cat((state, sample_action), 1)

            if index > self.features["n_past"] - 1:  # horizon
                h, skip = self.encoder(x_pred)
                h_target = self.encoder(scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                # Multi-modal feature model:
                MM_rep = self.MMFM(h)

                h_pred = self.frame_predictor(torch.cat([MM_rep, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append(x_pred)
            else:  # context
                h, skip = self.encoder(scene[index])
                h_target = self.encoder(scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior(h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior(h)  # learned prior

                # Multi-modal feature model:
                MM_rep = self.MMFM(h)

                h_pred = self.frame_predictor(torch.cat([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder([h_pred, skip])  # prediction model

                mae += self.mae_criterion(x_pred, scene[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = x_pred

        outputs = [last_output] + outputs

        if test is False:
            loss = mae + (kld * self.features["beta"])
            loss.backward()

            self.frame_predictor_optimizer.step()
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.MMFM_optimizer.step()

        return mae.data.cpu().numpy() / (self.features["n_past"] + self.features["n_future"]), kld.data.cpu().numpy() / (self.features["n_future"] + self.features["n_past"]), torch.stack(outputs)

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features["batch_size"]
