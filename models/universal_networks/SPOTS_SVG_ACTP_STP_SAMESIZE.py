# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import universal_networks.utils as utility_prog


class Model:
    def __init__(self, features):
        self.features = features

        if self.features["optimizer"] == "adam" or self.features["optimizer"] == "Adam":
            self.optimizer = optim.Adam

        if self.features["criterion"] == "L1":
            self.criterion = nn.L1Loss().to(self.features["device"])
            self.criterion_scene = nn.L1Loss().to(self.features["device"])
            self.criterion_tactile = nn.L1Loss().to(self.features["device"])
        if self.features["criterion"] == "L2":
            self.criterion = nn.MSELoss().to(self.features["device"])
            self.criterion_scene = nn.MSELoss().to(self.features["device"])
            self.criterion_tactile = nn.MSELoss().to(self.features["device"])


    def load_model(self, full_model):
        self.frame_predictor_tactile = full_model["frame_predictor_tactile"].to(self.features["device"])
        self.frame_predictor_scene = full_model["frame_predictor_scene"].to(self.features["device"])
        self.posterior = full_model["posterior"].to(self.features["device"])
        self.prior = full_model["prior"].to(self.features["device"])
        self.encoder_scene = full_model["encoder_scene"].to(self.features["device"])
        self.decoder_scene = full_model["decoder_scene"].to(self.features["device"])
        self.MMFM_scene = full_model["MMFM_scene"].to(self.features["device"])
        self.MMFM_tactile = full_model["MMFM_tactile"].to(self.features["device"])

    def calc_model_size(self):
        total_params = 0
        for model in [self.frame_predictor_tactile, self.frame_predictor_scene, self.posterior, self.prior, self.encoder_scene, self.decoder_scene, self.MMFM_scene, self.MMFM_tactile]:
            params = sum(p.numel() for p in model.parameters())
            total_params += params
        return total_params

    def initialise_model(self):
        import universal_networks.dcgan_64 as model
        import universal_networks.ACTP as ACTP_model
        import universal_networks.lstm as lstm_models

        # SCENE:
        self.frame_predictor_scene = lstm_models.lstm(self.features["g_dim"] + self.features["tactile_size"] + self.features["z_dim"] + self.features["state_action_size"], self.features["g_dim"], self.features["rnn_size"], self.features["predictor_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.frame_predictor_scene.apply(utility_prog.init_weights).to(self.features["device"])

        self.MMFM_scene = model.MMFM_scene(self.features["g_dim"] + self.features["tactile_size"], self.features["g_dim"] + self.features["tactile_size"], self.features["channels"])
        self.MMFM_scene.apply(utility_prog.init_weights).to(self.features["device"])
        self.MMFM_scene_optimizer = self.optimizer(self.MMFM_scene.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))

        self.encoder_scene = model.encoder(self.features["g_dim"], self.features["channels"])
        self.decoder_scene = model.decoder(self.features["g_dim"], self.features["channels"])
        self.encoder_scene.apply(utility_prog.init_weights).to(self.features["device"])
        self.decoder_scene.apply(utility_prog.init_weights).to(self.features["device"])

        self.frame_predictor_optimizer_scene = self.optimizer(self.frame_predictor_scene.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.encoder_optimizer_scene = self.optimizer(self.encoder_scene.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.decoder_optimizer_scene = self.optimizer(self.decoder_scene.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))

        # TACTILE:
        self.frame_predictor_tactile = ACTP_model.ACTP(device=self.features["device"], input_size=(self.features["g_dim"] + self.features["tactile_size"]), tactile_size=self.features["tactile_size"])
        self.frame_predictor_tactile.apply(utility_prog.init_weights).to(self.features["device"])

        self.MMFM_tactile = model.MMFM_tactile(self.features["g_dim"] + self.features["tactile_size"], self.features["g_dim"] + self.features["tactile_size"], self.features["channels"])
        self.MMFM_tactile.apply(utility_prog.init_weights).to(self.features["device"])
        self.MMFM_tactile_optimizer = self.optimizer(self.MMFM_tactile.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))

        self.frame_predictor_optimizer_tactile = self.optimizer(self.frame_predictor_tactile.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))

        # PRIOR:
        self.posterior = lstm_models.gaussian_lstm(self.features["g_dim"] + self.features["tactile_size"], self.features["z_dim"], self.features["rnn_size"], self.features["posterior_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.prior = lstm_models.gaussian_lstm(self.features["g_dim"] + self.features["tactile_size"], self.features["z_dim"], self.features["rnn_size"], self.features["prior_rnn_layers"], self.features["batch_size"], self.features["device"])
        self.posterior.apply(utility_prog.init_weights).to(self.features["device"])
        self.prior.apply(utility_prog.init_weights).to(self.features["device"])
        self.posterior_optimizer = self.optimizer(self.posterior.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))
        self.prior_optimizer = self.optimizer(self.prior.parameters(), lr=self.features["lr"], betas=(self.features["beta1"], 0.999))


    def run(self, scene, tactile, actions, gain, test=False, stage=False):
        mae_tactile = 0
        kld_tactile = 0
        mae_scene = 0
        kld_scene = 0
        outputs_scene = []
        outputs_tactile = []

        # scene
        self.frame_predictor_scene.zero_grad()
        self.encoder_scene.zero_grad()
        self.decoder_scene.zero_grad()
        self.frame_predictor_scene.hidden = self.frame_predictor_scene.init_hidden()

        # tactile
        self.frame_predictor_tactile.zero_grad()
        self.frame_predictor_tactile.init_hidden(scene.shape[1])

        self.MMFM_scene.zero_grad()
        self.MMFM_tactile.zero_grad()

        # prior
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

        state = actions[0].to(self.features["device"])
        for index, (sample_scene, sample_tactile, sample_action) in enumerate(zip(scene[:-1], tactile[:-1], actions[1:])):
            state_action = torch.cat((state, sample_action), 1)

            if index > self.features["n_past"] - 1:  # horizon
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(x_pred_scene)
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                h_target_scene_and_tactile = torch.cat([tactile[index + 1], h_scene], 1)

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([x_pred_tactile, h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile= self.frame_predictor_tactile(MM_rep_tactile, state_action, x_pred_tactile)  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs_tactile.append(x_pred_tactile)
                outputs_scene.append(x_pred_scene)

            else:  # context
                # Scene Encoding
                h_scene, skip_scene = self.encoder_scene(scene[index])
                h_target_scene      = self.encoder_scene(scene[index + 1])[0]

                h_target_scene_and_tactile = torch.cat([tactile[index + 1], h_scene], 1)

                # cat scene and tactile together for crossover input to pipelines
                h_scene_and_tactile = torch.cat([tactile[index], h_scene], 1)

                # Learned Prior - Z_t calculation
                if test:
                    _, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    z_t, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior(h_target_scene_and_tactile)  # learned prior
                    _, mu_p, logvar_p = self.prior(h_scene_and_tactile)  # learned prior

                # Multi-modal feature model:
                MM_rep_scene = self.MMFM_scene(h_scene_and_tactile)
                MM_rep_tactile = self.MMFM_tactile(h_scene_and_tactile)

                # Tactile Prediction
                x_pred_tactile = self.frame_predictor_tactile(MM_rep_tactile, state_action, tactile[index])  # prediction model

                # Scene Prediction
                h_pred_scene = self.frame_predictor_scene(torch.cat([MM_rep_scene, z_t, state_action], 1))  # prediction model
                x_pred_scene = self.decoder_scene([h_pred_scene, skip_scene])  # prediction model

                # loss calulations for tactile and scene:
                mae_tactile += self.criterion_tactile(x_pred_tactile, tactile[index + 1])  # prediction model

                mae_scene += self.criterion_scene(x_pred_scene, scene[index + 1])  # prediction model
                kld_scene += self.kl_criterion_scene(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output_scene = x_pred_scene
                last_output_tactile = x_pred_tactile

        outputs_scene = [last_output_scene] + outputs_scene
        outputs_tactile = [last_output_tactile] + outputs_tactile

        if test is False:
            if stage == "":
                loss_scene = mae_scene + (kld_scene * self.features["beta"])
                loss_tactile = mae_tactile + (kld_tactile * self.features["beta"])
                combined_loss = loss_scene + loss_tactile
                combined_loss.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()

                self.MMFM_scene_optimizer.step()
                self.MMFM_tactile_optimizer.step()

                self.posterior_optimizer.step()
                self.prior_optimizer.step()

            elif stage == "scene_only":
                loss_scene = mae_scene + (kld_scene * self.features["beta"])
                loss_scene.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()

                self.MMFM_scene_optimizer.step()

                self.posterior_optimizer.step()
                self.prior_optimizer.step()

            elif stage == "tactile_loss_plus_scene_fixed":
                loss_tactile = mae_tactile
                loss_tactile.backward()

                self.frame_predictor_optimizer_tactile.step()
                self.MMFM_tactile_optimizer.step()

            elif stage == "scene_loss_plus_tactile_gradual_increase":
                loss_scene = mae_scene + (kld_scene * self.features["beta"])
                loss_tactile = mae_tactile + (kld_tactile * self.features["beta"])
                combined_loss = loss_scene + (loss_tactile * gain)
                combined_loss.backward()

                self.frame_predictor_optimizer_scene.step()
                self.encoder_optimizer_scene.step()
                self.decoder_optimizer_scene.step()

                self.frame_predictor_optimizer_tactile.step()

                self.MMFM_scene_optimizer.step()
                self.MMFM_tactile_optimizer.step()

                self.posterior_optimizer.step()
                self.prior_optimizer.step()

        return mae_scene.data.cpu().numpy() / (self.features["n_past"] + self.features["n_future"]), kld_scene.data.cpu().numpy() / (self.features["n_future"] + self.features["n_past"]), \
               mae_tactile.cpu().data.numpy() / (self.features["n_past"] + self.features["n_future"]), torch.stack(outputs_scene), torch.stack(outputs_tactile)

    def kl_criterion_scene(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features["batch_size"]

    def kl_criterion_tactile(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.features["batch_size"]

    def set_train(self):
        self.frame_predictor_tactile.train()
        self.frame_predictor_scene.train()
        self.encoder_scene.train()
        self.decoder_scene.train()

        self.MMFM_scene.train()
        self.MMFM_tactile.train()

        self.posterior.train()
        self.prior.train()

    def set_test(self):
        self.frame_predictor_tactile.eval()
        self.frame_predictor_scene.eval()
        self.encoder_scene.eval()
        self.decoder_scene.eval()

        self.MMFM_scene.eval()
        self.MMFM_tactile.eval()

        self.posterior.eval()
        self.prior.eval()

    def save_model(self, stage="best"):
        if stage == "best":
            save_name = ""
        elif stage == "scene_only":
            save_name = "stage1"
        elif stage == "tactile_loss_plus_scene_fixed":
            save_name = "stage2"
        elif stage == "scene_loss_plus_tactile_gradual_increase":
            save_name = "stage3"

        torch.save({"frame_predictor_tactile": self.frame_predictor_tactile,
                    "frame_predictor_scene": self.frame_predictor_scene, "encoder_scene": self.encoder_scene,
                    "decoder_scene": self.decoder_scene,
                    "posterior": self.posterior, "prior": self.prior, 'features': self.features,
                    "MMFM_scene": self.MMFM_scene, "MMFM_tactile": self.MMFM_tactile}, 
                    self.features["model_dir"] + self.features["model_name"] + "_model"  + save_name + self.features["model_name_save_appendix"])
