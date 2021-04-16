import torch
import torch.nn as nn

class Gmodel(nn.Module):
    def __init__(self, noise_channel):
        super(Gmodel, self).__init__()
        self.nc = noise_channel
        self.conv= nn.Sequential(
            nn.ConvTranspose2d(self.nc,512,kernel_size=4, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512,256,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64,1,kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)
class Dmodel(nn.Module):
    def __init__(self):
        super(Dmodel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,kernel_size =4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32, kernel_size = 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64, kernel_size = 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128, kernel_size = 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,1,kernel_size=4, stride=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.conv(x)




if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_g = Gmodel(100).to(device)
    model_d = Dmodel().to(device)
    # summary(model_g, input_size=(100, 1, 1))
    summary(model_d, input_size=(1, 64, 64))