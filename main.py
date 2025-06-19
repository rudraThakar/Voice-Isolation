class SpectrogramUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(2, 128)
        self.enc2 = DownBlock(128, 128)
        self.enc3 = DownBlock(128, 128)
        self.enc4 = DownBlock(128, 256)
        self.enc5 = DownBlock(256, 256)

        self.middle = ResidualBlock(256, 256)

        self.dec5 = UpBlock(256, 256)
        self.dec4 = UpBlock(256, 256)
        self.dec3 = UpBlock(256, 128)
        self.dec2 = UpBlock(128, 128)
        self.dec1 = ResidualBlock(128, 2)  # output back to 2 channels

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        mid = self.middle(e5)

        d5 = self.dec5(mid, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        out = self.dec1(d2 + e1)  # add final skip connection

        return out
