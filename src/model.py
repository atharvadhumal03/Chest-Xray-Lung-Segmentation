"""
File containing the UNet architecture

"""
import torch
import torch.nn as nn


# Double Convolution Layer
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


 # Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Upsample: halves channels, doubles spatial size
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # After concatenating skip connection, in_channels doubles
        self.conv = DoubleConv(out_channels*2, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Initial double conv (no maxpool)
        self.encoder1 = DoubleConv(in_channels, 64)
        
        # 4 encoder blocks (MaxPool + DoubleConv)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        
        # Bottleneck
        self.bottleneck = Encoder(512, 1024)
        
        # 4 decoder blocks
        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)
        
        # Final 1x1 conv
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        skip1 = self.encoder1(x)      
        skip2 = self.encoder2(skip1)  
        skip3 = self.encoder3(skip2)  
        skip4 = self.encoder4(skip3)  
        
        # Bottleneck
        x = self.bottleneck(skip4)
        
        # Decoder - skip connections passed here
        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        
        return self.final_conv(x)