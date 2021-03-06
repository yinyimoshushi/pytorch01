import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentaion import eval_semantic_segmentation
from FCN import FCN
import cfg

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True)
val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True)

fcn = FCN(12)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def train(model):
    best = [0]
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print("Eopch is [{}/{}]".format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0

        for i, sample in enumerate(train_data):
            img_data = Variable(sample["img"].to(device))
            img_label = Variable(sample["label"].to(device))

            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metric["mean_class_accuracy"]
            train_miou += eval_metric["miou"]
            train_class_acc += eval_metric["class_accuracy"]

            print("|batch{}/{}|batch_loss {: .8f}".format(i+1,len(train_data),loss.item()))

        metric_description = '|Train Acc|: {:.5f} |Train Mean IOU |: {: .5f}\n|Train_class_acc| :{:}'.format(
            train_acc/len(train_data),
            train_miou/len(train_data),
            train_class_acc/len(train_data)
        )

        print(metric_description)

        if max(best) <= train_miou/len(train_data):
            best.append(train_miou/len(train_data))
            t.save(fcn.state_dict(),'{}.pth'.format(epoch))

def evalate(model):
    net = model.eval()

    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0
    t.set_grad_enabled(False)

    for i, sample in enumerate(val_data):
        img_data = Variable(sample["img"].to(device))
        img_label = Variable(sample["label"].to(device))

        out = net(img_data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out,img_label)
        eval_loss = loss.item() + eval_loss

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = img_label.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metric = eval_semantic_segmentation(pre_label, true_label)
        eval_acc += eval_metric["mean_class_accuracy"]
        eval_miou += eval_metric["miou"]
        eval_class_acc += eval_metric["class_accuracy"]

    metric_description = '|Evalate Acc|: {:.5f} |Evalate Mean IOU |: {: .5f}\n|Evalate_class_acc| :{:}'.format(
        eval_acc / len(val_data),
        eval_miou / len(val_data),
        eval_class_acc / len(val_data)
    )

    print(metric_description)



if __name__ == '__main__':
    train(fcn)
    evalate(fcn)


