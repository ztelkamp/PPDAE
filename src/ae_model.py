import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
            img_dim = avgpool_out(img_dim, 2, 2)

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


class ResNet_AE(nn.Module):
    def __init__(self, fc_hidden1=128, fc_hidden2=64, dropout=0.3,
                 latent_dim=32, in_ch=1, img_dim=28):
        super(ResNet_AE, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        # 2d kernal size
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)
        # 2d strides
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)
        # 2d padding
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)

        # encoding components
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        # add 1 cnv layer if input channels are different than 3
        if in_ch != 3:
            first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=3,
                                          stride=1, padding=1, dilation=1,
                                          groups=1, bias=True)]
            first_conv_layer.extend(modules)
            modules = first_conv_layer
            del first_conv_layer
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors
        self.fc3_z = nn.Linear(self.fc_hidden2, self.latent_dim)

        # Sampling vector
        self.fc4 = nn.Linear(self.latent_dim, self.fc_hidden1)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden1)
        self.fc5 = nn.Linear(self.fc_hidden1, 128 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(128 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8,
                               kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=in_ch,
                               kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(in_ch, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3_z(x)
        return x

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        print(x.shape)
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 128, 4, 4)
        print(x.shape)
        x = self.convTrans6(x)
        print(x.shape)
        x = self.convTrans7(x)
        print(x.shape)
        x = self.convTrans8(x)
        print(x.shape)
        x = self.convTrans9(x)
        print(x.shape)
        x = self.convTrans10(x)
        print(x.shape)
        x = F.interpolate(x, size=(self.img_width, self.img_height),
                          mode='bilinear')
        print(x.shape)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst, z