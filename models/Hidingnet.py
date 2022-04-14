import torch.nn as nn
import torch
import decoder



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
            nn.Conv2d(pathchannel * 8, pathchannel * 8, kernel_size=4, stride=2, padding=1)

        )
        self.main8 =nn.ReLU(True)
        self.decoder = decoder()


    def forward(self, cover, secret1, secret2, secret3):
        save_cover_feature = []
        cover_ouput1 = self.main1(cover)
        save_cover_feature.append(cover_ouput1)
        cover_ouput2 = self.main2(cover_ouput1)
        save_cover_feature.append(cover_ouput2)
        cover_ouput3 = self.main3(cover_ouput2)
        save_cover_feature.append(cover_ouput3)
        cover_ouput4 = self.main4(cover_ouput3)
        save_cover_feature.append(cover_ouput4)
        cover_ouput5 = self.main5(cover_ouput4)
        save_cover_feature.append(cover_ouput5)
        cover_ouput6 = self.main6(cover_ouput5)
        save_cover_feature.append(cover_ouput6)
        cover_ouput7 = self.main7(cover_ouput6)
        cover_ouput8 = self.main8(cover_ouput7)
        ############################################################################
        save_secret_feature1 = []
        secret_ouput1 = self.main1(secret1)
        save_secret_feature1.append(secret_ouput1)
        secret_ouput2 = self.main2(secret_ouput1)
        save_secret_feature1.append(secret_ouput2)
        secret_ouput3 = self.main3(secret_ouput2)
        save_secret_feature1.append(secret_ouput3)
        secret_ouput4 = self.main4(secret_ouput3)
        save_secret_feature1.append(secret_ouput4)
        secret_ouput5 = self.main5(secret_ouput4)
        save_secret_feature1.append(secret_ouput5)
        secret_ouput6 = self.main6(secret_ouput5)
        save_secret_feature1.append(secret_ouput6)
        secret_ouput7 = self.main7(secret_ouput6)
        secret_output7 = secret_ouput7.clone()
        save_secret_feature1.append(secret_ouput7)
        secret_ouput8 = self.main8(secret_output7)
        save_secret_feature1.append(secret_ouput8)
        ############################################################################
        save_secret_feature2 = []
        secret2_ouput1 = self.main1(secret2)
        save_secret_feature2.append(secret2_ouput1)
        secret2_ouput2 = self.main2(secret2_ouput1)
        save_secret_feature2.append(secret2_ouput2)
        secret2_ouput3 = self.main3(secret2_ouput2)
        save_secret_feature2.append(secret2_ouput3)
        secret2_ouput4 = self.main4(secret2_ouput3)
        save_secret_feature2.append(secret2_ouput4)
        secret2_ouput5 = self.main5(secret2_ouput4)
        save_secret_feature2.append(secret2_ouput5)
        secret2_ouput6 = self.main6(secret2_ouput5)
        save_secret_feature2.append(secret2_ouput6)
        secret2_ouput7 = self.main7(secret2_ouput6)
        secret2_output7 = secret2_ouput7.clone()
        save_secret_feature2.append(secret2_ouput7)
        secret2_ouput8 = self.main8(secret2_output7)
        #############################################################################
        save_secret_feature3 = []
        secret3_ouput1 = self.main1(secret3)
        save_secret_feature3.append(secret3_ouput1)
        secret3_ouput2 = self.main2(secret3_ouput1)
        save_secret_feature3.append(secret3_ouput2)
        secret3_ouput3 = self.main3(secret3_ouput2)
        save_secret_feature3.append(secret3_ouput3)
        secret3_ouput4 = self.main4(secret3_ouput3)
        save_secret_feature3.append(secret3_ouput4)
        secret3_ouput5 = self.main5(secret3_ouput4)
        save_secret_feature3.append(secret3_ouput5)
        secret3_ouput6 = self.main6(secret3_ouput5)
        save_secret_feature3.append(secret3_ouput6)
        secret3_ouput7 = self.main7(secret3_ouput6)
        secret3_output7 = secret3_ouput7.clone()
        save_secret_feature3.append(secret3_ouput7)
        secret3_ouput8 = self.main8(secret3_output7)

        concat_image = torch.cat([cover_ouput8, secret_ouput8, secret2_ouput8, secret3_ouput8], 1)
        final_img = self.decoder(concat_image)

        return final_img, save_cover_feature, save_secret_feature1, save_secret_feature2, save_secret_feature3









