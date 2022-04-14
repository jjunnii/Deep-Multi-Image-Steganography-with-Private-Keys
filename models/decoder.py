import torch.nn as nn
import torch

class decoder(nn.Module):
    def __init__(self, inputchannel=2048, pathchannel=64, outputchannel=3, outputfunction=nn.Sigmoid):
        super(decoder, self).__init__()
        self.main1 = nn.Sequential(
            nn.ConvTranspose2d(inputchannel, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel *40, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel *40, pathchannel * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 8)
        )
        self.main4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel * 40, pathchannel * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 4)
        )
        self.main5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel * 20, pathchannel * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel * 2)
        )
        self.main6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel * 10, pathchannel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(pathchannel)
        )
        self.main7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(pathchannel * 5, outputchannel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, concat, memory, memory2, memory3, memory4):
        concat_ouput1 = self.main1(concat)
        skipconnection1 = torch.cat([concat_ouput1, memory[5], memory2[5], memory3[5], memory4[5]], 1)
        concat_ouput2 = self.main2(skipconnection1)
        skipconnection2 = torch.cat([concat_ouput2, memory[4], memory2[4], memory3[4], memory4[4]], 1)
        concat_ouput3 = self.main3(skipconnection2)
        skipconnection3 = torch.cat([concat_ouput3, memory[3], memory2[3], memory3[3], memory4[3]], 1)
        concat_ouput4 = self.main4(skipconnection3)
        skipconnection4 = torch.cat([concat_ouput4, memory[2], memory2[2], memory3[2], memory4[2]], 1)
        concat_ouput5 = self.main5(skipconnection4)
        skipconnection5 = torch.cat([concat_ouput5, memory[1], memory2[1], memory3[1], memory4[1]], 1)
        concat_ouput6 = self.main6(skipconnection5)
        skipconnection6 = torch.cat([concat_ouput6, memory[0], memory2[0], memory3[0], memory4[0]], 1)
        concat_ouput7 = self.main7(skipconnection6)

        return concat_ouput7

  