import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ImprovedAttentionCNN(nn.Module):
    """ResNet18 + CBAM Attention - EXACT match to trained model"""
    def __init__(self, num_classes=5, pretrained=False, dropout_rate=0.5):
        super(ImprovedAttentionCNN, self).__init__()
        
        # Load ResNet18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Add CBAM attention modules
        self.attention1 = CBAM(64, reduction_ratio=8)
        self.attention2 = CBAM(128, reduction_ratio=8)
        self.attention3 = CBAM(256, reduction_ratio=8)
        self.attention4 = CBAM(512, reduction_ratio=8)
        
        self.avgpool = resnet.avgpool
        
        # Enhanced classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x