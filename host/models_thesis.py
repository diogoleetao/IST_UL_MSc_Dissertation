import torch
import torch.nn as nn
import torch.nn.functional as F

def cbr(ic, oc, k=3):
    return nn.Sequential(nn.Conv2d(ic, oc, k, padding=k // 2), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

class Encoder(nn.Module):

    def __init__(self, in_channels=4, width=1.0, widths=None):
        super().__init__()
        c = list(widths) if widths is not None else [max(8, int(round(w * width))) for w in (32, 64, 128, 256, 512)]

        def blk(ic, oc, pool=True):
            L = [nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
            if pool:
                L.append(nn.MaxPool2d(2))
            return nn.Sequential(*L)
        self.enc1 = blk(in_channels, c[0], pool=False)
        self.enc2 = blk(c[0], c[1])
        self.enc3 = blk(c[1], c[2])
        self.enc4 = blk(c[2], c[3])
        self.enc5 = blk(c[3], c[4])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c[4], 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return self.fc(self.gap(x).flatten(1))

class FCN4Band(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()

        def blk(ic, oc, pool=True):
            L = [nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
            if pool:
                L.append(nn.MaxPool2d(2))
            return nn.Sequential(*L)
        self.enc1 = blk(in_channels, 32, pool=False)
        self.enc2 = blk(32, 64)
        self.enc3 = blk(64, 128)
        self.enc4 = blk(128, 256)
        self.enc5 = blk(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = blk(512, 256, pool=False)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = blk(256, 128, pool=False)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = blk(128, 64, pool=False)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = blk(64, 32, pool=False)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        x = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), e3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), e2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), e1], dim=1))
        return self.out(x)

class ContrArm(nn.Module):

    def __init__(self, in_ch, filters):
        super().__init__()
        self.c1 = cbr(in_ch, filters, 3)
        self.c2 = cbr(filters, filters, 3)
        self.cb = cbr(in_ch, filters // 2, 1)

    def forward(self, inp):
        x = self.c2(self.c1(inp))
        x1 = torch.cat([inp, self.cb(inp)], 1)
        return F.relu(x + x1)

class ImprvContrArm(nn.Module):

    def __init__(self, in_ch, filters):
        super().__init__()
        self.c1 = cbr(in_ch, filters, 3)
        self.c0 = cbr(filters, filters, 3)
        self.c2 = cbr(filters, filters, 3)
        self.cb = cbr(in_ch, filters // 2, 1)
        self.cx2 = cbr(filters, filters, 1)

    def forward(self, inp):
        x = self.c1(inp)
        x0 = self.c0(x)
        x = self.c2(x0)
        x1 = torch.cat([inp, self.cb(inp)], 1)
        x2 = self.cx2(x0)
        return F.relu(x + x1 + x2)

class Bridge(nn.Module):

    def __init__(self, in_ch, filters):
        super().__init__()
        self.c1 = cbr(in_ch, filters, 3)
        self.c2conv = nn.Conv2d(filters, filters, 3, padding=1)
        self.drop = nn.Dropout(0.15)
        self.bn = nn.BatchNorm2d(filters)
        self.cb = cbr(in_ch, filters // 2, 1)

    def forward(self, inp):
        x = self.c1(inp)
        x = F.relu(self.bn(self.drop(self.c2conv(x))))
        x1 = torch.cat([inp, self.cb(inp)], 1)
        return F.relu(x + x1)

def _ff(t, copies, pool):
    return F.max_pool2d(torch.cat([t] * copies, 1), pool)

class _FaithfulBackbone(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.init = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.a1 = ContrArm(16, 32)
        self.a2 = ContrArm(32, 64)
        self.a3 = ContrArm(64, 128)
        self.a4 = ContrArm(128, 256)
        self.a5 = ImprvContrArm(256, 512)
        self.bridge = Bridge(512, 1024)

    def forward(self, x):
        c1 = self.a1(self.init(x))
        c2 = self.a2(F.max_pool2d(c1, 2))
        c3 = self.a3(F.max_pool2d(c2, 2))
        c4 = self.a4(F.max_pool2d(c3, 2))
        c5 = self.a5(F.max_pool2d(c4, 2))
        c6 = self.bridge(F.max_pool2d(c5, 2))
        return (c1, c2, c3, c4, c5, c6)

class CloudNetFaithful(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.bb = _FaithfulBackbone(in_channels)
        self.T7 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv7 = nn.Sequential(cbr(1024, 512), cbr(512, 512), cbr(512, 512))
        self.T8 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv8 = nn.Sequential(cbr(512, 256), cbr(256, 256))
        self.T9 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv9 = nn.Sequential(cbr(256, 128), cbr(128, 128))
        self.T10 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv10 = nn.Sequential(cbr(128, 64), cbr(64, 64))
        self.T11 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv11 = nn.Sequential(cbr(64, 32), cbr(32, 32))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1, c2, c3, c4, c5, c6 = self.bb(x)
        t7 = self.T7(c6)
        p7 = F.relu(_ff(c4, 2, 2) + _ff(c3, 4, 4) + _ff(c2, 8, 8) + _ff(c1, 16, 16) + c5)
        x = self.conv7(torch.cat([t7, p7], 1))
        x = F.relu(x + c5 + t7)
        t8 = self.T8(x)
        p8 = F.relu(_ff(c3, 2, 2) + _ff(c2, 4, 4) + _ff(c1, 8, 8) + c4)
        x = self.conv8(torch.cat([t8, p8], 1))
        x = F.relu(x + c4 + t8)
        t9 = self.T9(x)
        p9 = F.relu(_ff(c2, 2, 2) + _ff(c1, 4, 4) + c3)
        x = self.conv9(torch.cat([t9, p9], 1))
        x = F.relu(x + c3 + t9)
        t10 = self.T10(x)
        p10 = F.relu(_ff(c1, 2, 2) + c2)
        x = self.conv10(torch.cat([t10, p10], 1))
        x = F.relu(x + c2 + t10)
        t11 = self.T11(x)
        x = self.conv11(torch.cat([t11, c1], 1))
        x = F.relu(x + c1 + t11)
        return self.out(x)

class EncoderFaithful(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.bb = _FaithfulBackbone(in_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        *_, c6 = self.bb(x)
        return self.fc(self.gap(c6).flatten(1))

def build_model(name, bands, width=1.0):
    if name == 'encoder':
        return Encoder(bands, width)
    if name == 'fcn':
        return FCN4Band(bands)
    if name == 'faithful':
        return CloudNetFaithful(bands)
    if name == 'encoder_faithful':
        return EncoderFaithful(bands)
    raise ValueError(f"unknown model '{name}'")
