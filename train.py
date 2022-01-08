from SRGAN import pytorch_ssim
from SRGAN.data_utils import TrainDatasetFromFolder
from SRGAN.data_utils import ValDatasetFromFolder
from SRGAN.data_utils import display_transform
from SRGAN.loss import GeneratorLoss
from SRGAN.model import Generator
from SRGAN.model import Discriminator

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as utils
import torch.optim as optim
import torch.utils.data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from math import log10
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Super Resolution Models")
    parser.add_argument("--dataset-path", required=True, type=str, help="path to training dataset")
    parser.add_argument("--crop_size", default=88, type=int, help="training images crop size")
    parser.add_argument(
        "--upscale_factor", default=4, choices=[2, 4, 8],
        type=int, help="super resolution upscale factor"
    )
    parser.add_argument("--epochs", default=30, type=int, help="# of epochs")
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--save-path", default="models", type=str, help="path to save training weight")
    args = parser.parse_args()
    # print(args)
    # exit(0)

    DATASET_PATH   = args.dataset_path
    CROP_SIZE      = args.crop_size
    UPSCALE_FACTOR = args.upscale_factor
    NUM_EPOCHS     = args.epochs
    BATCH_SIZE     = args.batch_size
    SAVE_PATH      = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Private attribute
    GEN_WEIGHT = "models/5fold_gen_SRx8.pth"
    DIS_WEIGHT = "models/5fold_dis_SRx8.pth"
    VAL_IMAGE_SAVE_FREQ = 50
    VAL_IMAGE_SAVE_N    = 1
    FOLD_ID = 0

    TRAIN_PATH = "DS_tv/train"
    VALID_PATH = "DS_tv/valid"
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(VALID_PATH, exist_ok=True)
    img_names = os.listdir(DATASET_PATH)
    # train_imgs, valid_imgs = train_test_split(img_names, test_size=0.2)
    # for img_name in train_imgs:
    #     shutil.copyfile(os.path.join(DATASET_PATH, img_name), os.path.join(TRAIN_PATH, img_name))
    # for img_name in valid_imgs:
    #     shutil.copyfile(os.path.join(DATASET_PATH, img_name), os.path.join(VALID_PATH, img_name))
    kf = KFold(n_splits=5, shuffle=True, random_state=2022)
    tra_val_pair = [fold for fold in kf.split(img_names)]
    for img_id in tra_val_pair[FOLD_ID][0]:
        img_name = img_names[img_id]
        shutil.copyfile(os.path.join(DATASET_PATH, img_name), os.path.join(TRAIN_PATH, img_name))
    for img_id in tra_val_pair[FOLD_ID][1]:
        img_name = img_names[img_id]
        shutil.copyfile(os.path.join(DATASET_PATH, img_name), os.path.join(VALID_PATH, img_name))


    train_set = TrainDatasetFromFolder(TRAIN_PATH, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    valid_set = ValDatasetFromFolder(VALID_PATH, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=valid_set, num_workers=4, batch_size=1, shuffle=True)

    netG = Generator(UPSCALE_FACTOR)
    if GEN_WEIGHT is not None:
        netG.load_state_dict(torch.load(GEN_WEIGHT))
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    if DIS_WEIGHT is not None:
        netD.load_state_dict(torch.load(DIS_WEIGHT))
    print("# discriminator parameters:", sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    monitor = "ssim"
    mode    = "max"
    bound   = np.inf if mode == "min" else (-np.inf)
    save_model_log = ""

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()


            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = "outputs/images"
        os.makedirs(out_path, exist_ok=True)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])

            if epoch % VAL_IMAGE_SAVE_FREQ == 0 and epoch != 0:
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)

                for img_idx in range(min(VAL_IMAGE_SAVE_N, len(val_images))):
                    image = val_images[img_idx]
                    img_name = "SRF_{}_ep{}_{}.png".format(UPSCALE_FACTOR, epoch, img_idx)
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, os.path.join(out_path, img_name), padding=5)

        # save model parameters
        if ((mode == "min" and valing_results[monitor] < bound) or \
            (mode == "max" and valing_results[monitor] > bound)):
            save_model_log = "Epoch {} : {} improved from {} to {}\n".format(
                epoch, monitor, bound, valing_results[monitor])
            print(save_model_log)
            bound = valing_results[monitor]
            torch.save(
                netG.state_dict(),
                os.path.join(SAVE_PATH, "netG_SRx{}.pth".format(UPSCALE_FACTOR))
            )
            torch.save(
                netD.state_dict(),
                os.path.join(SAVE_PATH, "netD_SRx{}.pth".format(UPSCALE_FACTOR))
            )

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = "outputs/statistics"
            os.makedirs(out_path, exist_ok=True)

            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(
                os.path.join(out_path, "SRF_{}_train_results.csv".format(UPSCALE_FACTOR)), index_label="Epoch"
            )
    print("Last save model log: {}".format(save_model_log))
