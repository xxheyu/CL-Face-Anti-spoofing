from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
plt.switch_backend('agg') 


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def ImageShow(data_loader, encoder, decoder, opt, epoch, save=True):
    encoder.eval()
    decoder.eval()

    fake_img, real_img = [], []
    for i, (images, labels, idxs) in enumerate(data_loader):
        if i > 1:
            break
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        batch_size = labels.size(0)
        # inference
        _, features, feat_enc = encoder(images.detach())
        feat_enc = feat_enc[5].view(batch_size, 1, 32, 64)
        out = decoder(feat_enc)
        fake_img.append(vutils.make_grid(out.detach().cpu(), padding=2, normalize=True))
        real_img.append(vutils.make_grid(images.detach().cpu(), padding=2, normalize=True))

    # Plot the fake images from the first epoch
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(fake_img[-1], (1, 2, 0)))
    

    # Plot the real images from the first epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(real_img[-1], (1, 2, 0)))
    if save:
        plt.savefig('./figures/images/alpha_5/real_fake_epoch_{epoch}.jpg'.format(epoch=epoch))
        print('**********************')
        print('images saved')
    else:
        plt.show()
   
    plt.close()
   

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def PSNR(fake_img, ori_img):
    MSE = nn.MSELoss()
    batch_size = fake_img.size(0)
    return - 10 * MSE(fake_img.cuda(), ori_img.cuda()).log10()


def ReconstructionErrorHist(data_loader, encoder, decoder, opt, epoch, save=True):
    encoder.eval()
    decoder.eval()

    all_labels = []
    fake_img, real_img = torch.Tensor(), torch.Tensor()
    for i, (images, labels, idxs) in enumerate(data_loader):
        batch_size = labels.size(0)
        # Modify labels first
        for ind, k in enumerate(labels):
            if k in opt.original_index:
                labels[ind] = opt.original_index.index(k)
            else:
                labels[ind] = len(opt.original_index) # label as last label

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        all_labels.append(labels.data.cpu().numpy())

        # inference
        _, features, feat_enc = encoder(images.detach())
        feat_enc = feat_enc[5].view(batch_size, 1, 32, 64)
        out = decoder(feat_enc)
        # for renconstruction error histogram
        real_img = torch.cat([real_img, images.detach().cpu()], dim=0)
        fake_img = torch.cat([fake_img, out.detach().cpu()], dim=0)

    test_labels = np.concatenate(all_labels, 0)

    MSE = nn.MSELoss()
    bsz = fake_img.size(0)
    match_err = []
    unmatch_err = []

    for i in range(bsz):
        if test_labels[i] == len(opt.original_index):
            #unmatch_err.append(torch.mean(torch.abs(fake_img[i] - real_img[i])).data.cpu().numpy())
            unmatch_err.append(MSE(fake_img[i], real_img[i]).data.cpu().numpy())
        else:
            #match_err.append(torch.mean(torch.abs(fake_img[i] - real_img[i])).data.cpu().numpy())
            match_err.append(MSE(fake_img[i], real_img[i]).data.cpu().numpy())

    match_err = np.array(match_err)
    unmatch_err = np.array(unmatch_err)
    # print('**********************')
    # print('size of matching pairs is {size}'.format(size=match_err.size))
    # print('size of unmatching pairs is {size}'.format(size=unmatch_err.size))

    # plot histogram of reconstruction error
    bins_1 = np.linspace(min(match_err), max(match_err), 300)
    bins_2 = np.linspace(min(unmatch_err), max(unmatch_err), 200)
    plt.hist(match_err, bins_1, facecolor='g', label='Known')
    plt.hist(unmatch_err, bins_2, facecolor='r', label='Unknown')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Histogram')
    plt.legend()
    if save:
        plt.savefig('./figures/hist/alpha_5/hist_epoch_{epoch}.jpg'.format(epoch=epoch))
        print('**********************')
        print('histogram saved')
        print('**********************')
    else:
        plt.show()
    
    plt.close()
    