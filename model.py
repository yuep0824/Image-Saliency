import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V3_Large_Weights


class DoubleConv(nn.Module):
    """双卷积层：(Conv2d → BN → ReLU) × 2"""
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


class Up(nn.Module):
    """可学习上采样模块：转置卷积 + 双卷积（带跳连）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 转置卷积实现上采样（stride=2 → 尺寸×2）
        self.up_conv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # 初始化转置卷积权重为双线性插值（缓解棋盘效应）
        nn.init.kaiming_normal_(self.up_conv.weight, mode='fan_out', nonlinearity='relu')
        self.conv = DoubleConv(in_channels, out_channels)  # 拼接后通道数：in_channels//2 + skip_channels = in_channels

    def forward(self, x1, x2):
        """
        x1: 解码器输入（小尺寸）
        x2: 编码器跳连输入（大尺寸）
        """
        x1 = self.up_conv(x1)  # 上采样到x2的尺寸
        # 对齐尺寸（防止padding导致的微小偏差）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)  # 跳连拼接
        return self.conv(x)


class FCN8s_Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN8s_Baseline, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        self.features1 = nn.Sequential(*features[:5])   # 下2倍 → 128×128, 64通道
        self.features2 = nn.Sequential(*features[5:10]) # 下4倍 → 64×64, 128通道
        self.features3 = nn.Sequential(*features[10:17])# 下8倍 → 32×32, 256通道
        
        self.conv1x1_1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 转置卷积（严格匹配256×256输出）
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)
        
        # 初始化转置卷积权重
        for m in [self.upsample2, self.upsample4, self.upsample8]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 输入尺寸：256×256
        x1 = self.features1(x)  # [B,64,128,128]
        x2 = self.features2(x1) # [B,128,64,64]
        x3 = self.features3(x2) # [B,256,32,32]
        
        out1 = self.conv1x1_1(x1)  # [B,1,128,128]
        out2 = self.conv1x1_2(x2)  # [B,1,64,64]
        out3 = self.conv1x1_3(x3)  # [B,1,32,32]
        
        # 上采样到256×256
        out1_up = self.upsample2(out1)  # 128→256
        out2_up = self.upsample4(out2)  # 64→256
        out3_up = self.upsample8(out3)  # 32→256
        
        return out3_up + out2_up + out1_up


class EnhancedFCN(nn.Module):
    def __init__(self, num_classes=1):
        super(EnhancedFCN, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        # 编码器（下采样2/4/8/16/32倍）
        self.encoder1 = nn.Sequential(*features[:5])    # 128×128, 64
        self.encoder2 = nn.Sequential(*features[5:10])  # 64×64, 128
        self.encoder3 = nn.Sequential(*features[10:17]) # 32×32, 256
        self.encoder4 = nn.Sequential(*features[17:24]) # 16×16, 512
        self.encoder5 = nn.Sequential(*features[24:])   # 8×8, 512
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 1×1降维
        self.conv1 = nn.Conv2d(64, num_classes, 1)
        self.conv2 = nn.Conv2d(128, num_classes, 1)
        self.conv3 = nn.Conv2d(256, num_classes, 1)
        self.conv4 = nn.Conv2d(512, num_classes, 1)
        self.conv5 = nn.Conv2d(512, num_classes, 1)
        
        # 可学习上采样（替换插值）
        self.up_c5 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2)  # 8→16
        self.up_c4 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2)  # 16→32
        self.up_c3 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2)  # 32→64
        self.up_c2 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2)  # 64→128
        self.up_final = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2) # 128→256
        
        # 特征融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(num_classes * 5, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)  # [B,64,128,128]
        e2 = self.encoder2(e1) # [B,128,64,64]
        e3 = self.encoder3(e2) # [B,256,32,32]
        e4 = self.encoder4(e3) # [B,512,16,16]
        e5 = self.encoder5(e4) # [B,512,8,8]
        e5 = self.mid_conv(e5)
        
        # 降维
        c1 = self.conv1(e1)  # [B,1,128,128]
        c2 = self.conv2(e2)  # [B,1,64,64]
        c3 = self.conv3(e3)  # [B,1,32,32]
        c4 = self.conv4(e4)  # [B,1,16,16]
        c5 = self.conv5(e5)  # [B,1,8,8]
        
        # 可学习上采样到128×128
        c5_up = self.up_c5(c5)  # 8→16
        c5_up = self.up_c4(c5_up) # 16→32
        c5_up = self.up_c3(c5_up) # 32→64
        c5_up = self.up_c2(c5_up) # 64→128
        
        c4_up = self.up_c4(c4)  # 16→32
        c4_up = self.up_c3(c4_up) # 32→64
        c4_up = self.up_c2(c4_up) # 64→128
        
        c3_up = self.up_c3(c3)  # 32→64
        c3_up = self.up_c2(c3_up) # 64→128
        
        c2_up = self.up_c2(c2)  # 64→128
        
        # 特征融合
        fused = torch.cat([c1, c2_up, c3_up, c4_up, c5_up], dim=1)  # [B,5,128,128]
        fused = self.fuse_conv(fused)  # [B,1,128,128]
        
        # 最终上采样到256×256
        return self.up_final(fused)


class OutConv(nn.Module):
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
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 输入：256×256
        x1 = self.inc(x)       # 256×256, 64
        x2 = self.down1(x1)    # 128×128, 128
        x3 = self.down2(x2)    # 64×64, 256
        x4 = self.down3(x3)    # 32×32, 512
        x5 = self.down4(x4)    # 16×16, 1024
        
        x = self.up1(x5, x4)   # 32×32, 512
        x = self.up2(x, x3)    # 64×64, 256
        x = self.up3(x, x2)    # 128×128, 128
        x = self.up4(x, x1)    # 256×256, 64
        return self.outc(x)
    

class UNet_ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18并拆分编码器
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64通道, 1/2
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64通道, 1/4
        self.encoder3 = resnet.layer2  # 128通道, 1/8
        self.encoder4 = resnet.layer3  # 256通道, 1/16
        self.encoder5 = resnet.layer4  # 512通道, 1/32

        # 解码器：上采样+特征融合
        self.decoder5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose2d(64 + 64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose2d(64 + 64, 1, kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器提取多尺度特征
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(feat1)
        feat3 = self.encoder3(feat2)
        feat4 = self.encoder4(feat3)
        feat5 = self.encoder5(feat4)

        # 解码器融合与上采样
        dec5 = self.decoder5(feat5)
        fuse4 = torch.cat([dec5, feat4], dim=1)
        dec4 = self.decoder4(fuse4)

        fuse3 = torch.cat([dec4, feat3], dim=1)
        dec3 = self.decoder3(fuse3)

        fuse2 = torch.cat([dec3, feat2], dim=1)
        dec2 = self.decoder2(fuse2)

        fuse1 = torch.cat([dec2, feat1], dim=1)
        out = self.decoder1(fuse1)

        return self.sigmoid(out)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_block(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2, padding=0, output_padding=0
        )
        nn.init.kaiming_normal_(self.up_conv.weight, mode='fan_out', nonlinearity='relu')
        self.fuse_conv = ConvBlock(out_channels + skip_channels, out_channels)
    
    def forward(self, x, skip_x):
        x = self.up_conv(x)
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])

        skip_x = F.interpolate(skip_x, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip_x], dim=1)
        x = self.fuse_conv(x)
        return x
    
class MobileNetV3_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(MobileNetV3_UNet, self).__init__()
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        features = mobilenet.features
        
        self.encoder1 = nn.Sequential(*features[:4])   # [B,24,64,64]
        self.encoder2 = nn.Sequential(*features[4:7])  # [B,40,32,32]
        self.encoder3 = nn.Sequential(*features[7:11]) # [B,80,16,16]
        self.encoder4 = nn.Sequential(*features[11:16])# [B,160,8,8]
        self.encoder5 = nn.Sequential(*features[16:])  # [B,960,8,8]
        
        # 3. 瓶颈层（处理encoder5的高通道数）
        self.bottleneck = ConvBlock(960, 160)  # 960→160，匹配encoder4通道数
        
        # 4. 解码器（完全适配MobileNetV3的通道/尺寸规律，无插值）
        self.up4 = UpSampleBlock(in_channels=160, skip_channels=160, out_channels=80)
        self.up3 = UpSampleBlock(in_channels=80, skip_channels=80, out_channels=40)
        self.up2 = UpSampleBlock(in_channels=40, skip_channels=40, out_channels=24)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(24, 24, kernel_size=4, stride=4, padding=0),  # 64→256，可训练
            ConvBlock(24 + 24, 24)  # 拼接encoder1的24通道后融合
        )
        
        self.out_conv = nn.Conv2d(24, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # [B,24,64,64]
        e2 = self.encoder2(e1) # [B,40,32,32]
        e3 = self.encoder3(e2) # [B,80,16,16]
        e4 = self.encoder4(e3) # [B,160,8,8]
        e5 = self.encoder5(e4) # [B,960,8,8]

        bottleneck = self.bottleneck(e5)  # [B,160,8,8]
        
        d4 = self.up4(bottleneck, e4)     # [B,80,16,16]（8×8→16×16，转置卷积）
        d3 = self.up3(d4, e3)             # [B,40,32,32]（16×16→32×32，转置卷积）
        d2 = self.up2(d3, e2)             # [B,24,64,64]（32×32→64×64，转置卷积）
        # 最后一步上采样到256×256（64×64→256×256，转置卷积stride=4）
        d1 = self.up1[0](d2)              # [B,24,256,256]
        # 拼接encoder1的特征（先上采样encoder1到256×256）
        e1_up = nn.ConvTranspose2d(24, 24, kernel_size=4, stride=4, padding=0).to(x.device)(e1)
        d1 = torch.cat([d1, e1_up], dim=1)# [B,48,256,256]
        d1 = self.up1[1](d1)              # [B,24,256,256]
        
        out = self.out_conv(d1)           # [B,1,256,256]
        return out

def window_partition(x, window_size):
    """将特征图划分为窗口（确保H/W是window_size的整数倍）
    x: [B, H, W, C]
    return: [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, f"H/W {H}*{W} 不能被window_size {window_size} 整除"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """将窗口合并为特征图
    windows: [B*num_windows, window_size, window_size, C]
    return: [B, H, W, C]
    """
    assert H % window_size == 0 and W % window_size == 0, f"H/W {H}*{W} 不能被window_size {window_size} 整除"
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# 新增：padding特征图到能被window_size整除
def pad_to_window_size(x, window_size):
    """
    x: [B, H, W, C]
    return: padded_x, (pad_h, pad_w)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad (left,right,top,bottom) for H/W
    return x, (pad_h, pad_w)

# 新增：去除padding
def unpad_from_window_size(x, pad_h, pad_w):
    """
    x: [B, H+pad_h, W+pad_w, C]
    return: [B, H, W, C]
    """
    if pad_h > 0 or pad_w > 0:
        x = x[:, :-pad_h, :-pad_w, :] if pad_h > 0 and pad_w > 0 else x
        x = x[:, :-pad_h, :, :] if pad_h > 0 and pad_w == 0 else x
        x = x[:, :, :-pad_w, :] if pad_w > 0 and pad_h == 0 else x
    return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "特征图尺寸不匹配"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H//2, W//2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, num_patches//4, 4*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, num_patches//4, 2*C]
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 改为8，适配64×64
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置：适配8×8窗口（(2*8-1)^2=225）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), num_heads))
        coords = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords, coords], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [num_windows*B, window_size*window_size, C]
        """
        B_, N, C = x.shape
        assert N == self.window_size * self.window_size, f"输入窗口尺寸error"
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]
        
        # 相对位置偏置：适配8×8窗口（225个偏置）
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # 核心修改：7→8
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W):
        B, L, C = x.shape

        shortcut = x
        
        # 1. LayerNorm
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 2. Padding到能被window_size整除（8）
        x_padded, (pad_h, pad_w) = pad_to_window_size(x, self.window_size)
        H_padded, W_padded = x_padded.shape[1], x_padded.shape[2]
        
        # 3. 划分窗口（8×8）
        x_windows = window_partition(x_padded, self.window_size)  # [B*num_windows, 8,8,C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows,64,C]
        
        # 4. 窗口注意力
        attn_windows = self.attn(x_windows)  # [B*num_windows,64,C]
        
        # 5. 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_padded, W_padded)  # [B,H_padded,W_padded,C]
        
        # 6. 去除padding
        x = unpad_from_window_size(x, pad_h, pad_w)  # [B,H,W,C]
        x = x.view(B, H * W, C)  # [B,L,C]
        
        # 7. 残差
        x = shortcut + x
        
        # 8. MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, num_heads, window_size) for _ in range(depth)])
        self.patch_merge = PatchMerging(dim)

    def forward(self, x, H, W):
        """
        x: [B, L, C] where L=H×W
        return: x (当前层输出), x_down (下采样输出), H_new, W_new
        """
        for block in self.blocks:
            x = block(x, H, W)
        # 下采样
        x_down = self.patch_merge(x, H, W)
        H_new, W_new = H // 2, W // 2
        return x, x_down, H_new, W_new

class SwinUpBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "特征图尺寸不匹配"
        shortcut = x
        
        # 窗口注意力
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # Padding+窗口划分
        x_padded, (pad_h, pad_w) = pad_to_window_size(x, self.window_size)
        H_padded, W_padded = x_padded.shape[1], x_padded.shape[2]
        x_windows = window_partition(x_padded, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # 注意力计算
        attn_windows = self.attn(x_windows)
        # 合并窗口+去padding
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_padded, W_padded)
        x = unpad_from_window_size(x, pad_h, pad_w)
        x = x.view(B, H * W, C)
        # 残差
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.window_size = 8  # 核心修改：7→8

        # 编码器
        self.patch_embed = PatchEmbed(img_size, 4, in_chans, embed_dim)  # [B, 64×64=4096, 96]
        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0], self.window_size)      # 64→32×32, 96→192
        self.stage2 = SwinStage(2*embed_dim, depths[1], num_heads[1], self.window_size)    # 32→16×16, 192→384
        self.stage3 = SwinStage(4*embed_dim, depths[2], num_heads[2], self.window_size)    # 16→8×8, 384→768
        self.stage4 = SwinStage(8*embed_dim, depths[3], num_heads[3], self.window_size)    # 8→4×4, 768→1536

        # 解码器：修正维度和形状转换逻辑
        self.up4 = nn.Sequential(
            nn.LayerNorm(1536),
            nn.Linear(1536, 768),
        )
        self.up4_conv = nn.ConvTranspose2d(768, 768, 2, stride=2)  # 4×4→8×8
        self.up_block4 = SwinUpBlock(768, num_heads[2], self.window_size)
        
        self.up3 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 384),
        )
        self.up3_conv = nn.ConvTranspose2d(384, 384, 2, stride=2)  # 8×8→16×16
        self.up_block3 = SwinUpBlock(384, num_heads[1], self.window_size)
        
        self.up2 = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
        )
        self.up2_conv = nn.ConvTranspose2d(192, 192, 2, stride=2)  # 16×16→32×32
        self.up_block2 = SwinUpBlock(192, num_heads[0], self.window_size)
        
        self.up1_conv = nn.ConvTranspose2d(192, 96, 2, stride=2)  # 32×32→64×64
        self.final_up = nn.ConvTranspose2d(96, 96, 4, stride=4)  # 64×64→256×256
        self.outc = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"输入尺寸{H}×{W}≠{self.img_size}×{self.img_size}"

        # -------------------------- 编码器 --------------------------
        # PatchEmbed: [B,3,256,256] → [B, 64×64=4096, 96]
        x_embed = self.patch_embed(x)  
        # Stage1: 64×64→32×32, 96→192（64是8的倍数，无padding）
        x1, x2, H2, W2 = self.stage1(x_embed, 64, 64)  # x1:[B,4096,96], x2:[B,1024,192]
        # Stage2: 32×32→16×16, 192→384
        x2, x3, H3, W3 = self.stage2(x2, H2, W2)       # x2:[B,1024,192], x3:[B,256,384]
        # Stage3: 16×16→8×8, 384→768
        x3, x4, H4, W4 = self.stage3(x3, H3, W3)       # x3:[B,256,384], x4:[B,64,768]
        # Stage4: 8×8→4×4, 768→1536
        x4, x5, H5, W5 = self.stage4(x4, H4, W4)       # x4:[B,64,768], x5:[B,16,1536]

        # -------------------------- 解码器 --------------------------
        # Up4: 4×4→8×8, 1536→768
        x = self.up4(x5)  # [B,16,768]
        x = x.view(B, 4, 4, 768).permute(0, 3, 1, 2)  # [B,768,4,4]
        x = self.up4_conv(x)  # [B,768,8,8]
        x = x.permute(0,2,3,1).reshape(B, -1, 768)  # [B,64,768]
        x = x + x4  # 残差连接
        x = self.up_block4(x, 8, 8)  # 8×8特征图，8×8窗口

        # Up3: 8×8→16×16, 768→384
        x = self.up3(x)  # [B,64,384]
        x = x.view(B, 8, 8, 384).permute(0, 3, 1, 2)  # [B,384,8,8]
        x = self.up3_conv(x)  # [B,384,16,16]
        x = x.permute(0,2,3,1).reshape(B, -1, 384)  # [B,256,384]
        x = x + x3  # 残差连接
        x = self.up_block3(x, 16, 16)

        # Up2: 16×16→32×32, 384→192
        x = self.up2(x)  # [B,256,192]
        x = x.view(B, 16, 16, 192).permute(0, 3, 1, 2)  # [B,192,16,16]
        x = self.up2_conv(x)  # [B,192,32,32]
        x = x.permute(0,2,3,1).reshape(B, -1, 192)  # [B,1024,192]
        x = x + x2  # 残差连接
        x = self.up_block2(x, 32, 32)

        # Up1: 32×32→64×64→256×256
        x = x.view(B, 32, 32, 192).permute(0, 3, 1, 2)  # [B,192,32,32]
        x = self.up1_conv(x)  # [B,96,64,64]
        x = self.final_up(x)  # [B,96,256,256]
        
        out = self.outc(x)  # [B,1,256,256]
        return out

