import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V3_Large_Weights

# ---------------------- 基础FCN（保留） ----------------------
class FCN8s_Baseline(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN8s_Baseline, self).__init__()
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        self.features1 = nn.Sequential(*features[:17])  # 下采样2倍
        self.features2 = nn.Sequential(*features[17:24]) # 下采样4倍
        self.features3 = nn.Sequential(*features[24:])   # 下采样8倍
        
        self.upsample8 = nn.ConvTranspose2d(512, num_classes, kernel_size=8, stride=8, padding=0)
        self.upsample4 = nn.ConvTranspose2d(512, num_classes, kernel_size=4, stride=4, padding=0)
        self.upsample2 = nn.ConvTranspose2d(256, num_classes, kernel_size=2, stride=2, padding=0)
        
        self.conv1x1_3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        
        out3 = self.conv1x1_3(x3)
        out3_up = self.upsample8(out3)
        
        out2 = self.conv1x1_2(x2)
        out2_up = self.upsample4(out2)
        
        out1_up = self.upsample2(x1)
        
        out = out3_up + out2_up + out1_up
        return out

# ---------------------- 基础U-Net（保留） ----------------------
class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

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

# ---------------------- 新增：1. U-Net-ResNet18（ResNet18作为编码器） ----------------------
class UNet_ResNet18(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet_ResNet18, self).__init__()
        # 加载预训练ResNet18作为编码器
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 提取ResNet18的特征层（去掉全连接层）
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B,64,H/2,W/2]
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # [B,64,H/4,W/4]
        self.encoder3 = resnet.layer2                                         # [B,128,H/8,W/8]
        self.encoder4 = resnet.layer3                                         # [B,256,H/16,W/16]
        self.encoder5 = resnet.layer4                                         # [B,512,H/32,W/32]

        # 解码器（适配ResNet特征通道数）
        self.up1 = Up(512, 256)    # 512→256
        self.up2 = Up(256, 128)    # 256→128
        self.up3 = Up(128, 64)     # 128→64
        self.up4 = Up(64, 64)      # 64→64
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器前向（ResNet18特征提取）
        x1 = self.encoder1(x)  # [B,64,H/2,W/2]
        x2 = self.encoder2(x1) # [B,64,H/4,W/4]
        x3 = self.encoder3(x2) # [B,128,H/8,W/8]
        x4 = self.encoder4(x3) # [B,256,H/16,W/16]
        x5 = self.encoder5(x4) # [B,512,H/32,W/32]

        # 解码器前向（跳连对应ResNet特征层）
        x = self.up1(x5, x4)   # 512+256 → 256
        x = self.up2(x, x3)    # 256+128 → 128
        x = self.up3(x, x2)    # 128+64 → 64
        x = self.up4(x, x1)    # 64+64 → 64
        logits = self.outc(x)  # 64→1（回归输出）
        return logits

# ---------------------- 新增：2. MobileNetv3 + U-Net（轻量级） ----------------------
class MobileNetV3_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(MobileNetV3_UNet, self).__init__()
        # 加载预训练MobileNetV3-Large（轻量级，也可换Small）
        mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        features = mobilenet.features  # MobileNetV3的特征提取层

        # 提取MobileNetV3的关键特征层（按下采样倍数划分）
        self.encoder1 = nn.Sequential(*features[:4])   # [B,24,H/2,W/2]
        self.encoder2 = nn.Sequential(*features[4:7])  # [B,40,H/4,W/4]
        self.encoder3 = nn.Sequential(*features[7:11]) # [B,80,H/8,W/8]
        self.encoder4 = nn.Sequential(*features[11:16])# [B,112,H/16,W/16]
        self.encoder5 = nn.Sequential(*features[16:])  # [B,960,H/32,W/32]

        # 适配MobileNet通道数的解码器
        self.up1 = Up(960, 112)  # 960→112
        self.up2 = Up(112*2, 80) # 112+112→80
        self.up3 = Up(80*2, 40)  # 80+80→40
        self.up4 = Up(40*2, 24)  # 40+40→24
        self.outc = nn.Conv2d(24, n_classes, kernel_size=1)  # 输出层

    def forward(self, x):
        # 编码器前向
        x1 = self.encoder1(x)  # [B,24,H/2,W/2]
        x2 = self.encoder2(x1) # [B,40,H/4,W/4]
        x3 = self.encoder3(x2) # [B,80,H/8,W/8]
        x4 = self.encoder4(x3) # [B,112,H/16,W/16]
        x5 = self.encoder5(x4) # [B,960,H/32,W/32]

        # 解码器前向（上采样+拼接）
        x = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)
        x = self.up1.conv(x)    # 复用DoubleConv

        x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up2.conv(x)

        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up3.conv(x)

        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up4.conv(x)

        # 恢复到原始输入尺寸 + 输出
        x = F.interpolate(x, size=x.shape[2:]*2, mode='bilinear', align_corners=True)
        logits = self.outc(x)
        return logits

# ---------------------- 新增：3. Swin-Unet（Transformer版） ----------------------
# Swin Transformer基础模块
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # 分块合并
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2C]
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
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, num_heads, window_size) for _ in range(depth)])
        self.patch_merge = PatchMerging(dim)

    def forward(self, x, H, W):
        for block in self.blocks:
            x = block(x)
        x_down = self.patch_merge(x, H, W)
        H, W = H // 2, W // 2
        return x, x_down, H, W

class SwinUpBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # 编码器
        self.patch_embed = PatchEmbed(img_size, 4, in_chans, embed_dim)
        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0])
        self.stage2 = SwinStage(2*embed_dim, depths[1], num_heads[1])
        self.stage3 = SwinStage(4*embed_dim, depths[2], num_heads[2])
        self.stage4 = SwinStage(8*embed_dim, depths[3], num_heads[3])

        # 解码器
        self.up4 = nn.Linear(8*embed_dim, 4*embed_dim, bias=False)
        self.up_block4 = SwinUpBlock(4*embed_dim, num_heads[2])
        self.up3 = nn.Linear(4*embed_dim, 2*embed_dim, bias=False)
        self.up_block3 = SwinUpBlock(2*embed_dim, num_heads[1])
        self.up2 = nn.Linear(2*embed_dim, embed_dim, bias=False)
        self.up_block2 = SwinUpBlock(embed_dim, num_heads[0])

        # 输出头（恢复图像尺寸）
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(embed_dim//2, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码器前向
        x_embed = self.patch_embed(x)  # [B, 64*64, 96] (256/4=64)
        H1, W1 = 64, 64
        x1, x2, H2, W2 = self.stage1(x_embed, H1, W1)  # x1:[B,64*64,96], x2:[B,32*32,192]
        x2, x3, H3, W3 = self.stage2(x2, H2, W2)       # x2:[B,32*32,192], x3:[B,16*16,384]
        x3, x4, H4, W4 = self.stage3(x3, H3, W3)       # x3:[B,16*16,384], x4:[B,8*8,768]
        x4, _, _, _ = self.stage4(x4, H4, W4)          # x4:[B,8*8,768]

        # 解码器前向
        x = self.up4(x4)  # [B,8*8,384]
        x = x + x3        # 跳连
        x = self.up_block4(x)

        x = self.up3(x)   # [B,8*8,192] → 上采样到[B,16*16,192]
        x = x.view(B, 8, 8, -1).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(16,16), mode='bilinear', align_corners=True)
        x = x.permute(0,2,3,1).reshape(B, -1, 192)
        x = x + x2        # 跳连
        x = self.up_block3(x)

        x = self.up2(x)   # [B,16*16,96] → 上采样到[B,32*32,96]
        x = x.view(B,16,16,-1).permute(0,3,1,2)
        x = F.interpolate(x, size=(32,32), mode='bilinear', align_corners=True)
        x = x.permute(0,2,3,1).reshape(B,-1,96)
        x = x + x1        # 跳连
        x = self.up_block2(x)

        # 恢复到原始图像尺寸
        x = x.view(B, 32, 32, -1).permute(0, 3, 1, 2)  # [B,96,32,32]
        x = self.up1(x)   # [B,48,64,64]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        logits = self.final_conv(x)  # [B,1,256,256]
        return logits