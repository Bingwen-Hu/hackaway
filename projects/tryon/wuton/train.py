import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from dataset import CPDataset, CPDataLoader
from models import SiameseUnetGenerator, VGGLoss, Discriminator



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model):
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c = inputs['c']
        im_mask = inputs['im_mask']
        c_gt = inputs['c_gt']
        im = inputs['im']

        grid, theta = model(im_mask, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')


        loss = criterionL1(warped_cloth, c_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            statedict = model.cpu().state_dict()
            torch.save(statedict, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def train_wuton(opt, train_loader, model, dmodel):
    model.train().cuda()
    dmodel.train().cuda()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    BCE_stable = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    doptimizer = torch.optim.SGD(dmodel.parameters(), lr=0.001)

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c = inputs['c'].cuda()
        im_mask = inputs['im_mask'].cuda()
        c_gt = inputs['c_gt'].cuda()
        im = inputs['im'].cuda()

        warp_person, warp_cloth = model(im_mask, c, training=True)
        pair_person, unpair_person = warp_person.split(3, dim=1)

        # adversal training D
        fake = dmodel(unpair_person.detach())
        real = dmodel(pair_person.detach())
        gan_target = torch.ones(real.shape[0], 16*12).cuda()
        D_loss = BCE_stable((real - fake).view(real.shape[0], -1), gan_target)
        doptimizer.zero_grad()
        D_loss.backward()
        doptimizer.step()

        # G
        G_loss = BCE_stable((fake.detach() - real.detach()).view(real.shape[0], -1), gan_target)
        loss_l1 = criterionL1(warp_cloth, c_gt)
        loss_vgg = criterionVGG(pair_person, im)
        loss = loss_l1 + loss_vgg + G_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            statedict = model.cpu().state_dict()
            torch.save(statedict, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # create model & train & save the final checkpoint
    model = SiameseUnetGenerator(opt.fine_height, opt.fine_width, opt.grid_size)
    dmodel = Discriminator(3, 64)
    train_wuton(opt, train_loader, model, dmodel)


    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
