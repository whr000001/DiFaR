import math
import random
from re import X
# from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import numpy as np
from SENet import Network


class EncodingPart(nn.Module):
    def __init__(
        self,
        shared_image_dim=128,
        shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_linear = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        # text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text)
        image_shared = self.shared_image(image) 
        return text_shared, image_shared


class SimilarityModule(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding(text, image)
        text_aligned = self.text_aligner(text_encoding) 
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        skl = torch.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime


class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2) 
        image_in = image.unsqueeze(1) 
        corre_dim = text.shape[1] 
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze() 
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out


class DetectionModule(nn.Module):
    def __init__(self, feature_dim=64+16+16, h_dim=64):
        super(DetectionModule, self).__init__()
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.senet = Network(64, 128, 19, 3)
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_raw, image_raw, text, image):
       
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_se, image_se = self.uni_se(text_prime, image_prime)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
       
        correlation = self.cross_module(text, image) 

        text_se, image_se, corre_se = text_se.unsqueeze(-1), image_se.unsqueeze(-1), correlation.unsqueeze(-1)
        attention_score = self.senet(torch.cat([text_se, image_se, corre_se], -1))

        text_final = text_prime * attention_score[:,0].unsqueeze(1)
        img_final = image_prime * attention_score[:,1].unsqueeze(1) 
        corre_final = correlation * attention_score[:,2].unsqueeze(1)
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier_corre(final_corre)

        skl = self.ambiguity_module(text, image) 
        weight_uni = (1-skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
       
        return pre_label,attention_score, skl_score

