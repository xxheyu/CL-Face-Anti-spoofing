import os
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

def sort_result(features, feat_bank, label_bank, opt, top_k=100):
    score = torch.matmul(features, feat_bank.transpose(0, 1))
    batch_size = features.size(0)
    batch_labels = np.array([[]] * (opt.num_classes)).T

    for i in range(batch_size):
        sorted_score, indices = torch.sort(score[i], descending=True)
        labels = label_bank[indices]
        ans = [0 for m in opt.original_index]
        count = [0 for n in opt.original_index]

        for j in range(top_k):
            ans[labels[j]] += sorted_score[j]
            count[labels[j]] += 1

        for k in range(len(count)):
            if count[k] == 0:
                count[k] = 1

        test_labels = np.array(ans) / np.array(count)
        #test_labels = np.array(ans) / top_k
        # osr = np.max(test_labels) * (-1) + 1
        # test_labels = np.append(test_labels, osr)
        # print(test_labels)
        # one-hot labels
        # max_prob = np.max(test_labels)
        # if max_prob < 0.94:
        #     for m in range(test_labels.size):
        #         if m == opt.num_classes:
        #             test_labels[m] = 1
        #         else:
        #             test_labels[m] = 0

        batch_labels = np.vstack((batch_labels, test_labels))
    # print('test outputs')
    # print(batch_labels)
    return batch_labels

def save_txt(all_path, outputs, AUC=None):
    if AUC is not None:
        txtName = "./val_result/exp5/val_{}.txt".format(AUC[-1])
    else:
        txtName = "./val_result/exp5/test_result.txt"

    with open(txtName, 'a+') as f:
        for i, path in enumerate(all_path):
            newline = path.split()[0] + ' ' + str(outputs[i]) + '\n'
            f.write(newline)
    f.close()
    print('**********************')
    print('txt saved to {}'.format(txtName))
    print('**********************')
    return txtName


def sort_txt(txtName):
    with open(txtName) as f:
        data = []
        line = []
        for item in f.readlines():
            line.append(item.strip(''))
            data.append(int(item.split('.')[0]))

    indices = sorted(range(len(data)), key=lambda k: data[k])

    with open(txtName, "w") as f:
        for i in range(len(indices)):
            newline = line[indices[i]]
            f.write(newline)
        # for i, path in enumerate(all_path):
        #     newline = path + ' ' + str(outputs[i]) + '\n'
        #     f.write(newline)
    f.close()

def rewrite_txt(txtName):
    with open(txtName) as f:
        line = []
        for item in f.readlines():
            name = str(int(item.split('/')[-1].split('.')[0]) + 4645) + '.png'
            line.append(name + ' '+ item.split(' ')[-1])

    with open(txtName, "w") as f:
        for i in range(len(line)):
            newline = line[i]
            f.write(newline)
    f.close()

def val(test_loader, ordered_train_loader, model, epoch, opt):
    """one epoch training"""
    model.eval()

    feat_bank, label_bank = torch.Tensor(), torch.Tensor().type(torch.int64)
    for i, (images, labels, path) in enumerate(ordered_train_loader):
        # modify labels with their new indexes - we are not using all the labels anymore at the training
        for ind, label in enumerate(labels):
            labels[ind] = opt.original_index.index(label)

        # images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        # inference
        _, features = model(images.detach())

        # all features and labels in training set
        feat_bank = torch.cat([feat_bank, features.cpu().data], dim=0)
        label_bank = torch.cat([label_bank, labels], dim=0)

        if i % 50 == 0:
            print(i, '/', len(ordered_train_loader))

    all_outputs, all_labels = [], []
    all_path = ()
    all_preds = np.array([] * 1).T
    for i, (images, labels, path) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # idxs   = idxs.cuda(non_blocking=True)
        bsz = images.shape[0]

        # One-hot labels
        labels = torch.nn.functional.one_hot(labels, num_classes=len(opt.original_index))
        all_labels.append(labels.data.cpu().numpy())

        # inference
        _, features = model(images.detach())
        outputs = sort_result(features.cpu().data, feat_bank, label_bank, opt)
        all_outputs.append(outputs)
        all_preds = np.hstack((all_preds, outputs[:,1]))
        all_path = all_path + tuple(path)
        


    test_labels = np.concatenate(all_labels, 0)
    test_outputs = np.concatenate(all_outputs, 0)

    AUC = metrics.roc_auc_score(test_labels, test_outputs, average=None)
    print("AU-ROC scores for each class. The last one is open set identification score:")
    print(AUC)
    txtName = save_txt(all_path, all_preds, AUC)
    sort_txt(txtName)

    score = AUC[-1]

    return score


def test(test_loader, ordered_train_loader, model, epoch, opt):
    """one epoch training"""
    model.eval()

    feat_bank, label_bank = torch.Tensor(), torch.Tensor().type(torch.int64)
    for i, (images,labels, path) in enumerate(ordered_train_loader):
        #if i > 2:
        #    break
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        # inference
        _, features = model(images.detach())

        # all features and labels in training set
        feat_bank = torch.cat([feat_bank, features.cpu().data], dim=0)
        label_bank = torch.cat([label_bank, labels], dim=0)

        if i % 50 == 0:
            print(i, '/', len(ordered_train_loader))

    all_outputs, all_labels = [], []
    all_path = ()
    all_preds = np.array([] * 1).T
    print('testing ... ...')
    for i, (images, path) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)

        bsz = images.shape[0]
        # inference
        _, features = model(images.detach())
        outputs = sort_result(features.cpu().data, feat_bank, label_bank, opt)
        all_outputs.append(outputs)
        all_preds = np.hstack((all_preds, outputs[:, 1]))
        all_path = all_path + tuple(path)
        
        if i % 50 == 0:
            print(i, '/', len(test_loader))

    txtName = save_txt(all_path, all_preds)
    rewrite_txt(txtName)


if __name__ == '__main__':
    # features = torch.rand(6, 128)
    # feat_bank = torch.rand(20, 128)
    # labels = np.random.randint(0, 6, size=20)
    # label_bank = torch.tensor(labels)
    #
    # outputs = sort_result(features, feat_bank, label_bank)
    sort_txt()
