import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class VisualSemanticEmbedding(nn.Module):
    def __init__(self, embed_ndim):
        super(VisualSemanticEmbedding, self).__init__()
        self.embed_ndim = embed_ndim

        # image feature
        self.img_encoder = models.vgg16(pretrained=True)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.feat_extractor = nn.Sequential(*(self.img_encoder.classifier[i] for i in range(6)))
        self.W = nn.Linear(4096, embed_ndim, False)

        # text feature
        self.txt_encoder = nn.GRU(embed_ndim, embed_ndim, 1)

    def forward(self, img, txt):
        # image feature
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.feat_extractor(img_feat)
        img_feat = self.W(img_feat)

        # text feature
        h0 = torch.zeros(1, img.size(0), self.embed_ndim)
        h0 = Variable(h0.cuda() if txt.data.is_cuda else h0)
        _, txt_feat = self.txt_encoder(txt, h0)
        txt_feat = txt_feat.squeeze()

        return img_feat, txt_feat


class ResidualBlock(nn.Module):
    def __init__(self, fusing_method = '', conditioning = True):
        super(ResidualBlock, self).__init__()
        self.fusing_method = fusing_method
        self.conditioning = conditioning
        self.encoder = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512)
            )
        if self.conditioning:
            if self.fusing_method == 'lowrank_BP':
                self.U = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1,bias=False),
                     nn.ReLU(inplace=True)
                                   )
                self.V = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1,bias=False),
                                   nn.ReLU(inplace=True)
                                   )
                self.P = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1),
                                   nn.ReLU(inplace=True)
                                   )
            elif self.fusing_method == 'FiLM':
                self.s_w = nn.Sequential(nn.Conv2d(128, 512 , kernel_size=1),
                                   nn.Sigmoid()
                                   )
                self.a_w = nn.Conv2d(128, 512 , kernel_size=1)
            else:
                self.concat_conv = nn.Sequential(
                nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
                nn.ReLU(inplace=True))

    def forward(self, x, t):
        if self.conditioning:
            if self.fusing_method == 'lowrank_BP':
                internal_emb1 = self.U(x)
                internal_emb2 = self.V(t)
                fusion = self.P(internal_emb1 * internal_emb2)
            elif self.fusing_method == 'FiLM':
                fusion = x * self.s_w(t) + self.a_w(t)
            else:
                fusion = self.concat_conv(torch.cat((x, t), dim=1))

        else:
            fusion = x


        return F.relu(x + self.encoder(fusion))


class Residual_blocks(nn.Module):
    def __init__(self, fusing_method = ''):
        super(Residual_blocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.fusing_method = fusing_method
        self.residual_block1 = ResidualBlock(fusing_method = self.fusing_method)
        self.residual_block2 = ResidualBlock(fusing_method = self.fusing_method)
        self.residual_block3 = ResidualBlock(fusing_method = self.fusing_method)
        self.residual_block4 = ResidualBlock(fusing_method = self.fusing_method)

    def forward(self, x, t):
        x = self.conv(x)
        x = self.residual_block1(x, t)
        x = self.residual_block2(x, t)
        x = self.residual_block3(x, t)
        x = self.residual_block4(x, t)
        return x


class Generator(nn.Module):
    def __init__(self, use_vgg=True, fusing = ''):
        super(Generator, self).__init__()
        self.fusing_method = fusing
        # encoder
        if use_vgg:
            self.encoder = models.vgg16_bn(pretrained=True)
            self.encoder = \
                nn.Sequential(*(self.encoder.features[i] for i in range(23) + range(24, 33)))
            self.encoder[24].dilation = (2, 2)
            self.encoder[24].padding = (2, 2)
            self.encoder[27].dilation = (2, 2)
            self.encoder[27].padding = (2, 2)
            self.encoder[30].dilation = (2, 2)
            self.encoder[30].padding = (2, 2)
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

        # residual blocks
        self.residual_blocks = Residual_blocks(fusing_method=self.fusing_method)

        # decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)

    def forward(self, img, txt_feat, z=None):
        # encoder
        img_feat = self.encoder(img)
        z_mean = self.mu(txt_feat)
        z_log_stddev = self.log_sigma(txt_feat)
        z = torch.randn(txt_feat.size(0), 128)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        txt_feat = z_mean + z_log_stddev.exp() * Variable(z)

        # residual blocks
        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        fusion = self.residual_blocks(img_feat, txt_feat)

        # decoder
        output = self.decoder(fusion)
        output = torch.tanh(output)
        return output, (z_mean, z_log_stddev)


class Discriminator(nn.Module):
    def __init__(self, fusing_method = ''):
        super(Discriminator, self).__init__()
        self.fusing_method = fusing_method
        self.encoder = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.residual_branch = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4)
        )

        self.compression = nn.Sequential(
            nn.Linear(300, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)

    def forward(self, img, txt_feat):
        img_feat = self.encoder(img)
        img_feat = F.leaky_relu(img_feat + self.residual_branch(img_feat), 0.2)
        txt_feat = self.compression(txt_feat)

        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        fusion = torch.cat((img_feat, txt_feat), dim=1)
        output = self.classifier(fusion)
        return output.squeeze()

