import torch
from torch import nn
class Encoder(torch.nn.Module):
    def __init__(self, D_in, latent_size):#3,64
        super(Encoder, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=D_in, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),#32,127,127
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),#64,63,63
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),#128,31,31
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),#256,15,15
        )
        self.enc_mu=nn.Linear(256*15*15,latent_size)
        self.enc_log_sigma=nn.Linear(256*15*15,latent_size)

    def forward(self, x):
        x=self.feature_extraction(x) # 256,15,15
        # print("encoder特征提取层的输出形状:{}".format(x.shape))
        x=x.view(-1,256*15*15)
        # print("view输出形状:{}".format(x.shape))
        mu = self.enc_mu(x)
        # print("mu输出形状:{}".format(mu.shape))
        log_sigma = self.enc_log_sigma(x)
        z = self.reparameterize(mu,log_sigma)
        # print("log_sigma输出形状:{}".format(log_sigma.shape))

        return mu,log_sigma,z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z