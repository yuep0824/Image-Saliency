import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------- FCN 基础版 ----------------------
class FCN8s_Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN8s_Baseline, self).__init__()
        # 加载预训练VGG16，仅用前几层做特征提取
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        # 编码器：下采样（提取特征）
        self.features1 = nn.Sequential(*features[:17])  # 下采样2倍
        self.features2 = nn.Sequential(*features[17:24]) # 下采样4倍
        self.features3 = nn.Sequential(*features[24:])   # 下采样8倍
        
        # 解码器：上采样（恢复尺寸）
        self.upsample8 = nn.ConvTranspose2d(512, num_classes, kernel_size=8, stride=8, padding=0)
        self.upsample4 = nn.ConvTranspose2d(512, num_classes, kernel_size=4, stride=4, padding=0)
        self.upsample2 = nn.ConvTranspose2d(256, num_classes, kernel_size=2, stride=2, padding=0)
        
        # 1x1卷积调整通道数
        self.conv1x1_3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器前向
        x1 = self.features1(x)  # [B,256,H/2,W/2]
        x2 = self.features2(x1) # [B,512,H/4,W/4]
        x3 = self.features3(x2) # [B,512,H/8,W/8]
        
        # 解码器前向（融合多尺度特征）
        out3 = self.conv1x1_3(x3)
        out3_up = self.upsample8(out3)  # 上采样8倍
        
        out2 = self.conv1x1_2(x2)
        out2_up = self.upsample4(out2)  # 上采样4倍
        
        out1_up = self.upsample2(x1)    # 上采样2倍
        
        # 特征融合 + 输出（回归任务无激活）
        out = out3_up + out2_up + out1_up
        return out


# ---------------------- U-Net 基础版 ----------------------
class DoubleConv(nn.Module):
    """U-Net基本单元：两次卷积+BN+ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样：MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样：转置卷积 + 拼接 + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸（防止整除问题）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出层：1x1卷积调整通道数"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_Baseline(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet_Baseline, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits