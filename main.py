# encoding: utf-8

import matplotlib.pyplot as plt
import argparse
import os
import shutil
import socket
import time
import math
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder, MyImageFolders
from models.RevealNet import RevealNet
import torchvision.models
from models.preprocessing import preprocessing
from models.Hidingnet import encoder
from skimage.util import random_noise

DATA_DIR = '/home/user/Desktop/jjun/trainset'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--encoder', default='',
                    help="path to decoder (to continue training)")
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--discriminator', default='',
                    help="path to discriminator (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--betaG', type=float, default=0.001,
                    help='hyper parameter of betaG')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--enet_path', default='', help='test mode, you need give the enet_weight dirs in this param')
parser.add_argument('--rnet_path', default='', help='test mode, you need give the rnet_weight dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')



def calc_psnr1(sr, hr, scale, rgb_range, dataset=None):
    #if hr.nelement() == 1: return 0
    diff = (sr - hr) / rgb_range
    if dataset =='benchmark':
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def calculate_mse(img1, img2):
    mse = []
    for i1, i2 in zip(img1, img2):
        err = np.mean((i1.astype(np.float64) - i2.astype(np.float64)) ** 2)

        mse.append(err)
    return mse


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1: np.ndarray, img2: np.ndarray):
    return [ssim(i1, i2) for i1, i2 in zip(img1, img2)]


def calculate_psnr(img1, img2):
    psnr = []
    for i1, i2 in zip(img1, img2):
        err = np.mean((i1.astype(np.float64) - i2.astype(np.float64)) ** 2)
        if err == 0:
            return float('inf')
        psnr.append(20 * math.log10(255.0 / math.sqrt(err)))

    return psnr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  
    cur_work_dir, mainfile = os.path.split(main_file_path) 

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main():
    ############### define global parameters ###############
    global opt, optimizerH1, optimizerE1, optimizerR1, writer, logPath, schedulerH1, schedulerR1, schedulerE1, val_loader, smallestLoss

    #################  output configuration   ###############
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if opt.test == '':
        # tensorboardX writer
        writer = SummaryWriter(comment='**' + opt.remark)
        ##############   get dataset   ############################
        traindir = os.path.join(DATA_DIR, 'train')
        valdir = os.path.join(DATA_DIR, 'val')
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),  # resize to a given size
                transforms.ToTensor(),
            ]))
        val_dataset = MyImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert train_dataset
        assert val_dataset
    else:

        testdir = opt.test
        test_dataset = MyImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert test_dataset

    Enet1 = encoder()
    Enet1.cuda()
    if opt.enet_path != "":
        Enet1.load_state_dict(torch.load(opt.enet_path))
    if opt.ngpu > 1:
        Enet1 = torch.nn.DataParallel(Enet1).cuda()
    print_network(Enet1)
    Rnet1 = RevealNet(output_function=nn.Sigmoid)
    Rnet1.cuda()
    Rnet1.apply(weights_init)
    if opt.rnet_path != '':
        Rnet1.load_state_dict(torch.load(opt.rnet_path))
    if opt.ngpu > 1:
        Rnet1 = torch.nn.DataParallel(Rnet1).cuda()

    print_network(Rnet1)

    # MSE loss
    criterion = nn.MSELoss().cuda()
    # BCE loss

    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerE1 = optim.Adam(Enet1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerE1 = ReduceLROnPlateau(optimizerE1, mode='min', factor=0.2, patience=5, verbose=True)
        optimizerR1 = optim.Adam(Rnet1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR1 = ReduceLROnPlateau(optimizerR1, mode='min', factor=0.2, patience=8, verbose=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers))
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers))
        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(opt.niter):
            ######################## train ##########################################
            train(train_loader, epoch, Enet1=Enet1, Rnet1=Rnet1, criterion=criterion)

            ####################### validation  #####################################
            val_hloss1, val_rloss1, val_rloss2, val_rloss3, val_Rsumloss, val_sumloss = validation(val_loader, epoch,
                                                                                                   Enet1=Enet1,                                                           
                                                                                                   Rnet1=Rnet1,
                                                                                                   criterion=criterion)

            ####################### adjust learning rate ############################
            schedulerE1.step(val_sumloss)
            schedulerR1.step(val_Rsumloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss
                # do checkPointing
                torch.save(Enet1.state_dict(),
                           '%s/netE1_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_hloss1))
                torch.save(Rnet1.state_dict(),
                           '%s/netR1_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_rloss1))

        writer.close()

    # test mode
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers))
        test(test_loader, 0, Enet1=Enet1, Rnet1=Rnet1, criterion=criterion)
        print(
            "##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def train(train_loader, epoch, Enet1, Rnet1, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses1 = AverageMeter()  # record loss of H-net
    Rlosses1 = AverageMeter()  # record loss of R-net
    Rlosses2 = AverageMeter()
    Rlosses3 = AverageMeter()
    SumLosses = AverageMeter()  # record Hloss + �*Rloss

    # switch to train mode
    Enet1.train()
    Rnet1.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Enet1.zero_grad()
        Rnet1.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 4)  # get true batch size of this step

        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img1 = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        secret_img2 = all_pics[this_batch_size * 2:this_batch_size * 3, :, :, :]
        secret_img3 = all_pics[this_batch_size * 3:this_batch_size * 4, :, :, :]



        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img1 = secret_img1.cuda()
            secret_img2 = secret_img2.cuda()
            secret_img3 = secret_img3.cuda()

        container_img, memory1, memory2, memory3, memory4 = Enet1(cover_img, secret_img1, secret_img2, secret_img3)



        errH1 = criterion(container_img, cover_img)  # loss between cover and container
        Hlosses1.update(errH1.data, this_batch_size)

        secretkey1 = F.interpolate(memory2[6], (256, 256))
        secretkey2 = F.interpolate(memory3[6], (256, 256))
        secretkey3 = F.interpolate(memory4[6], (256, 256))


        rev_secret_img1 = Rnet1(container_img, secretkey1)
        rev_secret_img2 = Rnet1(container_img, secretkey2)
        rev_secret_img3 = Rnet1(container_img, secretkey3)

        #############################################################################################

        #########################################################################################################

        # put concatenated image into R-net and get revealed secret image

        errR1 = criterion(rev_secret_img1, secret_img1)
        errR2 = criterion(rev_secret_img2, secret_img2)
        errR3 = criterion(rev_secret_img3, secret_img3)

        Rlosses1.update(errR1.data, this_batch_size)
        Rlosses2.update(errR2.data, this_batch_size)
        Rlosses3.update(errR3.data, this_batch_size)

        betaerrR_secret1 = opt.beta * errR1
        betaerrR_secret2 = opt.beta * errR2
        betaerrR_secret3 = opt.beta * errR3

        err_sum = errH1 + betaerrR_secret1 + betaerrR_secret2 + betaerrR_secret3
        SumLosses.update(err_sum.data, this_batch_size)

        err_sum.backward()
        optimizerE1.step()
        optimizerR1.step()

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H1: %.4f Loss_R1: %.4f Loss_R2: %.4f  Loss_R3: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses1.val, Rlosses1.val, Rlosses2.val, Rlosses3.val, SumLosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        # genereate a picture every resultPicFrequency steps
        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img1, rev_secret_img1.data,
                            secret_img2, rev_secret_img2.data, secret_img3, rev_secret_img3.data, epoch, i,
                            opt.trainpics)

    # epcoh log
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH1_lr = %.8f      optimizerE1_lr = %.8f      optimizerR1_lr = %.8f" % (
        optimizerH1.param_groups[0]['lr'], optimizerE1.param_groups[0]['lr'], optimizerR1.param_groups[0]['lr'],) + "\n"
    epoch_log = epoch_log + "epoch_Hloss1=%.6f\tepoch_Rloss1=%.6f\tepoch_Rloss2=%.6f\tepoch_Rloss3=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses1.avg, Rlosses1.avg, Rlosses2.avg, Rlosses3.avg, SumLosses.avg)
    print_log(epoch_log, logPath)

    if not opt.debug:
        # record lr
        writer.add_scalar("lr/H_lr", optimizerH1.param_groups[0]['lr'], epoch)

        writer.add_scalar("lr/R_lr", optimizerR1.param_groups[0]['lr'], epoch)

        writer.add_scalar("lr/beta", opt.beta, epoch)
        # record loss
        writer.add_scalar('train/R_loss', Rlosses1.avg, epoch)

        writer.add_scalar('train/H_loss', Hlosses1.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader, epoch, Enet1, Rnet1, criterion):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Enet1.eval()
    Rnet1.eval()

    Hlosses1 = AverageMeter()  # record loss of H-net
    Rlosses1 = AverageMeter()  # record loss of R-net
    Rlosses2 = AverageMeter()
    Rlosses3 = AverageMeter()

    for i, data in enumerate(val_loader, 0):

        Enet1.zero_grad()
        Rnet1.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 4)  # get true batch size of this step

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img1 = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        secret_img2 = all_pics[this_batch_size * 2:this_batch_size * 3, :, :, :]
        secret_img3 = all_pics[this_batch_size * 3:this_batch_size * 4, :, :, :]


        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img1 = secret_img1.cuda()
            secret_img2 = secret_img2.cuda()
            secret_img3 = secret_img3.cuda()

        container_img, memory1, memory2, memory3, memory4 = Enet1(cover_img, secret_img1, secret_img2, secret_img3)

        errH1 = criterion(container_img, cover_img)  # loss between cover and container
        Hlosses1.update(errH1.data, this_batch_size)

        secretkey1 = F.interpolate(memory2[6], (256, 256))
        secretkey2 = F.interpolate(memory3[6], (256, 256))
        secretkey3 = F.interpolate(memory4[6], (256, 256))


        rev_secret_img1 = Rnet1(container_img, secretkey1)
        rev_secret_img2 = Rnet1(container_img, secretkey2)
        rev_secret_img3 = Rnet1(container_img, secretkey3)

        #############################################################################################

        #########################################################################################################



        errR1 = criterion(rev_secret_img1, secret_img1)
        errR2 = criterion(rev_secret_img2, secret_img2)
        errR3 = criterion(rev_secret_img3, secret_img3)

        Rlosses1.update(errR1.data, this_batch_size)
        Rlosses2.update(errR2.data, this_batch_size)
        Rlosses3.update(errR3.data, this_batch_size)

        if i % 50 == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img1, rev_secret_img1.data,
                            secret_img2, rev_secret_img2.data, secret_img3, rev_secret_img3.data, epoch, i,
                            opt.validationpics)

    val_hloss1 = Hlosses1.avg
    val_rloss1 = Rlosses1.avg
    val_rloss2 = Rlosses2.avg
    val_rloss3 = Rlosses3.avg
    val_sumloss = val_hloss1 + opt.beta * val_rloss1 + opt.beta * val_rloss2 + opt.beta * val_rloss3
    val_Rsumloss = opt.beta * val_rloss1 + opt.beta * val_rloss2 + opt.beta * val_rloss3

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss1 = %.6f\t val_Rloss1 = %.6f\t val_Rloss2 = %.6f\t val_Rloss3 = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss1, val_rloss1, val_rloss2, val_rloss3, val_sumloss, val_time)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses1.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses1.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print(
        "#################################################### validation end ########################################################")
    return val_hloss1, val_rloss1, val_rloss2, val_rloss3, val_Rsumloss, val_sumloss


def test(test_loader, epoch, Enet1, Rnet1, criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Enet1.eval()
    Rnet1.eval()

    Hlosses1 = AverageMeter()  # record loss of H-net

    Rlosses1 = AverageMeter()  # record loss of R-net
    Rlosses2 = AverageMeter()
    Rlosses3 = AverageMeter()
    for i, data in enumerate(test_loader, 0):

        Enet1.zero_grad()
        Rnet1.zero_grad()

        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 4)  # get true batch size of this step

        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img1 = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        secret_img2 = all_pics[this_batch_size * 2:this_batch_size * 3, :, :, :]
        secret_img3 = all_pics[this_batch_size * 3:this_batch_size * 4, :, :, :]

  

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img1 = secret_img1.cuda()
            secret_img2 = secret_img2.cuda()
            secret_img3 = secret_img3.cuda()

        container_img, memory1, memory2, memory3, memory4 = Enet1(cover_img, secret_img1, secret_img2, secret_img3)


        errH1 = criterion(container_img, cover_img)  # loss between cover and container
        Hlosses1.update(errH1.data, this_batch_size)

        secretkey1 = F.interpolate(memory2[5], (256, 256))
        secretkey2 = F.interpolate(memory3[5], (256, 256))
        secretkey3 = F.interpolate(memory4[5], (256, 256))
        

        rev_secret_img1 = Rnet1(container_img, secretkey1)
        rev_secret_img2 = Rnet1(container_img, secretkey2) 
        rev_secret_img3 = Rnet1(container_img, secretkey3)


        #############################################################################################

        #########################################################################################################

        # put concatenated image into R-net and get revealed secret image

        errR1 = criterion(rev_secret_img1, secret_img1)
        errR2 = criterion(rev_secret_img2, secret_img2)
        errR3 = criterion(rev_secret_img3, secret_img3)

        Rlosses1.update(errR1.data, this_batch_size)
        Rlosses2.update(errR2.data, this_batch_size)
        Rlosses3.update(errR3.data, this_batch_size)

        cover_imgs = cover_img.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0
        container_imgs = container_img.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0

        secret_imgs1 = secret_img1.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0
        secret_imgs2 = secret_img2.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0
        secret_imgs3 = secret_img3.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0

        rev_secret_imgs1 = rev_secret_img1.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0
        rev_secret_imgs2 = rev_secret_img2.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0
        rev_secret_imgs3 = rev_secret_img3.cpu().detach().permute(0, 2, 3, 1).numpy() * 255.0


        cover_container_psnr1 = calc_psnr1(cover_img, container_img, 0, 1, dataset=None)
        cover_container_ssim1 = calculate_ssim(cover_imgs, container_imgs)


        secret_reveal_psnr1 = calc_psnr1(rev_secret_img1, secret_img1, 0, 1, dataset=None)
        secret_reveal_ssim1 = calculate_ssim(rev_secret_imgs1, secret_imgs1)
        secret_reveal_psnr2 = calc_psnr1(rev_secret_img2, secret_img2, 0, 1, dataset=None)
        secret_reveal_ssim2 = calculate_ssim(rev_secret_imgs2, secret_imgs2)
        secret_reveal_psnr3 = calc_psnr1(rev_secret_img3, secret_img3, 0, 1, dataset=None)
        secret_reveal_ssim3 = calculate_ssim(rev_secret_imgs3, secret_imgs3)

        container_psnr1 = []
        container_ssim1 = []


        secret_psnr1 = []
        secret_ssim1 = []
        secret_psnr2 = []
        secret_ssim2 = []
        secret_psnr3 = []
        secret_ssim3 = []

        container_psnr1.append(cover_container_psnr1)
        secret_psnr1.append(secret_reveal_psnr1)
        secret_psnr2.append(secret_reveal_psnr2)
        secret_psnr3.append(secret_reveal_psnr3)

        container_ssim1.append(cover_container_ssim1)
        secret_ssim1.append(secret_reveal_ssim1)
        secret_ssim2.append(secret_reveal_ssim2)
        secret_ssim3.append(secret_reveal_ssim3)

        if i % 50 == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img1, rev_secret_img1.data,
                            secret_img2, rev_secret_img2.data, secret_img3, rev_secret_img3.data, epoch, i,
                            opt.testPics)

    val_hloss1 = Hlosses1.avg
    val_rloss1 = Rlosses1.avg
    val_rloss2 = Rlosses2.avg
    val_rloss3 = Rlosses3.avg
    val_sumloss = val_hloss1 + val_hloss1 + opt.beta * val_rloss1 + opt.beta * val_rloss2 + opt.beta * val_rloss3

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss1 = %.6f\t val_Rloss1 = %.6f\tval_Rloss2 = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss1, val_rloss1, val_rloss2, val_sumloss, val_time)
    print_log(val_log, logPath)
    print(np.mean(container_psnr1), np.mean(container_ssim1))
    print(np.mean(secret_psnr1), np.mean(secret_ssim1))
    print(np.mean(secret_psnr2), np.mean(secret_ssim2))
    print(np.mean(secret_psnr3), np.mean(secret_ssim3))


    print(
        "#################################################### test end ########################################################")
    return val_hloss1, val_rloss1, val_rloss2, val_rloss3, val_sumloss


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv1, RevSecImg1, secretLabelv2, RevSecImg2,
                    secretLabelv3, RevSecImg3, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        secretFrames1 = secretLabelv1.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames2 = secretLabelv2.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames3 = secretLabelv3.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        revSecFrames1 = RevSecImg1.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames2 = RevSecImg2.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames3 = RevSecImg3.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        showContainer = torch.cat(
            [originalFrames, containerFrames, secretFrames1, revSecFrames1, secretFrames2, revSecFrames2, secretFrames3,
             revSecFrames3], 0)

        # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
        resultImg = showContainer
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
