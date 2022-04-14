import torch.nn as nn
import torch
import pdb
class encoder(nn.Module):
    def __init__(self, inputchannel=3, pathchannel=64):
        super(encoder, self).__init__()
        self.main1 = nn.Sequential(
            nn.Conv2d(inputchannel, pathchannel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel)
        )
        self.main2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel, pathchannel * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 2)
        )
        self.main3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel * 2, pathchannel * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 4)
        )
        self.main4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel * 4, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main5 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel * 8, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main6 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel * 8, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main7 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(pathchannel * 8, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

    def forward(self, secret):        ###########################################################
        save_secret_feature = []
        secret_ouput1 = self.main1(secret)
        save_secret_feature.append(secret_ouput1)
        secret_ouput2 = self.main2(secret_ouput1)
        save_secret_feature.append(secret_ouput2)
        secret_ouput3 = self.main3(secret_ouput2)
        save_secret_feature.append(secret_ouput3)
        secret_ouput4 = self.main4(secret_ouput3)
        save_secret_feature.append(secret_ouput4)
        secret_ouput5 = self.main5(secret_ouput4)
        save_secret_feature.append(secret_ouput5)
        secret_ouput6 = self.main6(secret_ouput5)
        save_secret_feature.append(secret_ouput6)
        secret_ouput7 = self.main7(secret_ouput6)
        save_secret_feature.append(secret_ouput7)



        #pdb.set_trace()
        return save_secret_feature


net = encoder()
print(net)







