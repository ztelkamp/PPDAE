import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2):
        super(Linear_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height

        # Encoder specification
        self.enc = nn.Sequential(
            nn.Linear(self.img_size, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, self.latent_dim)
        )
        # Decoder specification
        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, self.img_size)
        )

    def encode(self, images):
        img = images.view([images.size(0), -1])
        code = self.enc(img)
        return code

    def decode(self, code):
        out = self.dec(code)
        out = out.view([code.size(0), 1, self.img_width, self.img_height])
        return out

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code


class TranConv_AutoEncoder(nn.Module):
    def __init__(self, latent_dim, img_dim=28, dropout=.2):
        super(TranConv_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height

        # Encoder specification
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),  # (N, 1, 28, 28)->(N,  3, 24, 24)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),       # (N, 3, 24, 24)->(N,  3, 12, 12)
            nn.Conv2d(3, 6, kernel_size=3),  # (N, 3, 24, 24)->(N,  3, 10, 10)
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),       # (N, 6, 10, 10)->(N,  6, 5, 5)
            nn.Conv2d(6, 6, kernel_size=3),  # (N, 6, 5, 5)  ->(N,  6, 3, 3)
            nn.ReLU(),
            #nn.AvgPool2d(2, stride=2),       # (N, 6, 4, 4)  ->(N,  6, 2, 2)
        )

        # Decoder specification
        self.dec_convt = nn.Sequential(
            nn.ConvTranspose2d(6, 3, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2)
        )

    def encode(self, images):
        feats = self.enc_conv(images)
        return feats

    def decode(self, code):
        out = self.dec_convt(code)
        return out

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code.flatten(1)


class ConvLin_AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, img_dim=28, dropout=.2,
                 kernel=3, n_conv_blocks=5):
        super(ConvLin_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height

        # Encoder specification
        def conv_out(l0, k, st):
            return int((l0 - k)/st + 1)

        def avgpool_out(l0, k, st):
            return int((l0 - k)/st + 1)

        self.enc_conv_blocks = nn.Sequential()
        h_ch = 1
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module('conv2d_%i' % (i+1),
                                            nn.Conv2d(h_ch, h_ch*2,
                                                      kernel_size=kernel))
            self.enc_conv_blocks.add_module('relu_%i' % (i+1), nn.ReLU())
            self.enc_conv_blocks.add_module('avgpool_%i' % (i+1),
                                            nn.AvgPool2d(2, stride=2))
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            print(img_dim)
            img_dim = avgpool_out(img_dim, 2, 2)
            print(h_ch, img_dim)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim**2, 50),
            nn.ReLU(),
            nn.Linear(50, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(320, self.img_size)
        )

    def encode(self, images):
        feats = self.enc_conv_blocks(images)
        feats = feats.flatten(1)
        code = self.enc_linear(feats)
        return code

    def decode(self, code):
        out = self.dec_linear(code)
        out = out.view([code.size(0), 1, self.img_width, self.img_height])
        return out

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code