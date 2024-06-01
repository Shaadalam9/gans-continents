import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_shape[1] // 4

        self.embedding = nn.Linear(latent_dim + num_classes + 3, 128)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=128, nhead=8), num_layers=6)

        self.fc = nn.Sequential(
            nn.Linear(128, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels, counts):
        gen_input = torch.cat((self.label_emb(labels), noise, counts), -1)
        embedding = self.embedding(gen_input)
        transformer_output = self.transformer_encoder(embedding.unsqueeze(0)).squeeze(0)
        out = self.fc(transformer_output)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.embedding = nn.Linear(512 * (img_shape[1] // 16) ** 2 + num_classes + 3, 128)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=128, nhead=8), num_layers=6)
        self.fc = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, img, labels, counts):
        conv_output = self.conv_blocks(img)
        conv_output = conv_output.view(conv_output.size(0), -1)
        d_in = torch.cat((conv_output, self.label_embedding(labels), counts), -1)
        embedding = self.embedding(d_in)
        transformer_output = self.transformer_encoder(embedding.unsqueeze(0)).squeeze(0)
        validity = self.fc(transformer_output)
        return validity
