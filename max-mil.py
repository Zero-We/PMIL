import sys
import os
import numpy as np
import argparse
import random
import openslide
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models
from sklearn.metrics import roc_curve, auc
from model import resnet34
from collections import OrderedDict
import time
from Early_Stopping import EarlyStopping


np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='the standard instance-space Max-MIL')
parser.add_argument('--train_lib', type=str, default='lib/train.ckpt', help='lib to save wsi id of train set')
parser.add_argument('--val_lib', type=str, default='lib/val.ckpt', help='lib to save wsi id of valid set')
parser.add_argument('--test_lib', type=str, default='lib/test.ckpt', help='lib to save wsi id of test set')
parser.add_argument('--output', type=str, default='result', help='output directory')
parser.add_argument('--feat_dir', type=str, default='', help='path to save features')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
parser.add_argument('--nepochs', type=int, default=30, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--test_every', default=5, type=int, help='test on val every')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--mil_model', type=str, default='model/mil.pth', help='path to pretrained model')
parser.add_argument('--n', default=1000, type=int, help='select top n tiles')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--patch_dir', type=str, default='')
parser.add_argument('--save_model', default=False, action='store_true')
parser.add_argument('--save_feat', default=False, action='store_true')
parser.add_argument('--feat_format', type=str, choices = ['.csv', '.npy', '.pt'], default='.csv')
parser.add_argument('--load_model', default=False, action='store_true')
parser.add_argument('--is_test', default=False, action='store_true')
parser.add_argument('--save_index', default=False, action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0,1,2,3])


global args, best_acc
args = parser.parse_args()
torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

def main():
    best_acc = 0
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if args.load_model:
        ch = torch.load(args.mil_model, map_location='cpu')
        model.load_state_dict(ch["state_dict"], strict=False)
    model.to(device)

    model = nn.DataParallel(model, device_ids=args.device_ids)


    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1, last_epoch=-1)
    cudnn.benchmark = True

    # normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # load data
    inference_dset = Inferencedataset(args.train_lib, trans)
    inference_loader = torch.utils.data.DataLoader(
        inference_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = Inferencedataset(args.val_lib, trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
    test_dset = Inferencedataset(args.test_lib, trans)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    columns = []
    for i in range(1, 513):
        columns.append('feature' + str(i))

    epoch = 0
    # initialize the model saving path
    model_save_path = args.mil_model
    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=3, verbose=True)
    if not args.is_test:
        #loop throuh epochs
        for epoch in range(args.nepochs):
            begin = time.time()
            
            if epoch == args.nepochs-1 and args.save_feat:
                probs, features = inference(epoch, inference_loader, model, args.save_feat)
            else:
                probs, _ = inference(epoch, inference_loader, model)

            topk = group_argtopk(np.array(inference_dset.slideIDX), probs, args.k)
            t_data = inference_dset.maketraindata(topk)
            train_dset = Traindataset(t_data, args.train_lib, trans)
            train_loader = torch.utils.data.DataLoader(
                 train_dset,
                 batch_size=128, shuffle=True,
                 num_workers=args.workers, pin_memory=False)

            if epoch == args.nepochs-1 and args.save_index:
                topn = group_argtopk(np.array(inference_dset.slideIDX), probs, args.n)
                inference_dset.savetopndata(topn, filename=f"select_train")
                if args.save_feat:
                    slideIDX = np.array(train_dset.slideIDX)
                    for i, slide in enumerate(train_dset.slidenames):
                    slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                    feature = features[slideIDX == i, :]
                    
                    # save features
                    if args.feat_format == '.csv':
                        df = pd.DataFrame(feature, columns=columns)
                        df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                    elif args.feat_format == '.npy':
                        np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                    elif args.feat_format == '.pt':
                        feature = torch.from_numpy(feature)
                        torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                        
            train_dset.shuffletraindata()
            loss = train(epoch, train_loader, model, criterion, optimizer)
            end = time.time()
            usetime = end-begin
            print('Training\tEpoch: [{}/{}]\tLoss: {}\tUsetime: {:.4f}'.format(epoch+1, args.nepochs, loss, usetime))
            scheduler.step()


            #Validation
            if args.val_lib and (epoch+1) % args.test_every == 0:
                if epoch == args.nepochs-1 and args.save_feat:
                    probs, features = inference(epoch, val_loader, model, args.save_feat)
                else:
                    probs, _ = inference(epoch, val_loader, model)
                topk = group_argtopk(np.array(val_dset.slideIDX), probs, args.k)
                maxs = group_max(np.array(val_dset.slideIDX), probs, len(val_dset.targets))
                t_data = val_dset.maketraindata(topk)
                train_dset = Traindataset(t_data, args.val_lib, trans)
                train_loader = torch.utils.data.DataLoader(
                    train_dset,
                    batch_size=128, shuffle=True,
                    num_workers=args.workers, pin_memory=False)
                val_loss = train(epoch, train_loader, model, criterion, optimizer)
                if epoch == args.nepochs-1 and args.save_index:
                    topn = group_argtopk(np.array(val_dset.slideIDX), probs, args.n)
                    val_dset.savetopndata(topn, filename=f"select_val")
                    if args.save_feat:
                        slideIDX = np.array(val_dset.slideIDX)
                        for i, slide in enumerate(val_dset.slidenames):
                            slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                            feature = features[slideIDX == i, :]
                            
                            # save features
                            if args.feat_format == '.csv':
                                df = pd.DataFrame(feature, columns=columns)
                                df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                            elif args.feat_format == '.npy':
                                np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                            elif args.feat_format == '.pt':
                                feature = torch.from_numpy(feature)
                                torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                fpr, tpr, thresh = roc_curve(y_true=val_dset.targets, y_score=maxs, pos_label=1)
                roc_auc = auc(fpr, tpr)
                err, fpr, fnr = calc_err(pred, val_dset.targets)

                print("Validation Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
                print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}'.format(epoch+1, args.nepochs, err, fpr, fnr))

                if 1 - err >= best_acc:
                    best_acc = 1 - err

            if args.test_lib and (epoch+1) % args.test_every == 0:
                ## Test
                if epoch == args.nepochs-1 and args.save_feat:
                    probs, features = inference(epoch, test_loader, model, args.save_feat)
                else:
                    probs, _ = inference(epoch, test_loader, model)
                maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))
                if epoch == args.nepochs-1 and args.save_index:
                    topn = group_argtopk(np.array(test_dset.slideIDX), probs, args.n)
                    test_dset.savetopndata(topn, filename=f"select_test")
                    if args.save_feat:
                        slideIDX = np.array(test_dset.slideIDX)
                        for i, slide in enumerate(val_dset.slidenames):
                            slidename = '.'.join(os.path.basename(slide).split('.')[0:2])
                            feature = features[slideIDX == i, :]
                            
                            # save features
                            if args.feat_format == '.csv':
                                df = pd.DataFrame(feature, columns=columns)
                                df.to_csv(os.path.join(args.feat_dir, f'{slidename}.csv'))
                            elif args.feat_format == '.npy':
                                np.save(os.path.join(args.feat_dir, f'{slidename}.npy'), feature)
                            elif args.feat_format == '.pt':
                                feature = torch.from_numpy(feature)
                                torch.save(feature, os.path.join(args.feat_dir, f'{slidename}.pt'))
                
                pred = [1 if x >= 0.5 else 0 for x in maxs]
                fpr, tpr, thresh = roc_curve(y_true=test_dset.targets, y_score=maxs, pos_label=1)
                roc_auc = auc(fpr, tpr)
                err, fpr, fnr = calc_err(pred, test_dset.targets)
                print("Test Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
                print('Test\tError: {}\tFPR: {}\tFNR: {}'.format(err, fpr, fnr))

                ## early stop
                early_stopping(val_loss, best_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                ## Save the model
                if args.save_model:
                    obj = {
                        'state_dict': model.module.state_dict(),
                        'best_acc': best_acc,
                    }
                    torch.save(obj, model_save_path)

    ## Test
    ch = torch.load(model_save_path, map_location='cpu')
    print(ch['best_acc'])
    model.load_state_dict(ch["state_dict"], strict=False)
    probs = inference(epoch, test_loader, model)
    maxs = group_max(np.array(test_dset.slideIDX), probs, len(test_dset.targets))
    if args.save_index:
        topn = group_argtopk(np.array(test_dset.slideIDX), probs, args.n)
        test_dset.savetopndata(topn, filename=f"select_test")
    pred = [1 if x >= 0.5 else 0 for x in maxs]
    fpr, tpr, thresh = roc_curve(y_true=test_dset.targets, y_score=maxs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    err, fpr, fnr = calc_err(pred, test_dset.targets)
    print("Test Accuracy: {:.4f}\t AUC: {:.4f}".format(1 - err, roc_auc))
    miss_wsi(test_dset, pred)



def miss_wsi(dset, pred):
    slidenames = dset.slidenames
    wsi_names = [os.path.basename(slidename).split('.')[0] for slidename in slidenames]
    wsi_names = np.array(wsi_names)
    targets = np.array(dset.targets)
    pred = np.array(pred)
    miss_wsi_name = wsi_names[pred!=targets]
    print("Misclassify {} wsi".format(miss_wsi_name.shape[0]))
    print(miss_wsi_name)


def inference(run, loader, model, save_feat=False):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    prefetcher = data_prefetcher(loader)
    features = torch.Tensor()
    with torch.no_grad():
        input = prefetcher.next()
        i = 0
        while input is not None:
            if i % 1000 == 999:
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(loader)))

            input = input.to(device)
            output, feature = model(input)
            output = F.softmax(output, dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
            input = prefetcher.next()
            i += 1
            if save_feat:
                features = torch.cat((features, feature.cpu()), dim=0)
    return probs.cpu().numpy(), features.cpu().numpy()


def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output, _ = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
        torch.cuda.empty_cache()
    return running_loss/len(loader.dataset)


def pred(run, loader, model, criterion, optimizer):
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output, _ = model(input)
            loss = criterion(output, target)
            running_loss += loss.item()*input.size(0)
            torch.cuda.empty_cache()
    return running_loss/len(loader.dataset)


def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input



class Inferencedataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
        lib = torch.load(libraryfile, map_location='cpu')
        slides = []
        wsidirs = []
        for i,name in enumerate(lib['slides']):
            wsiname = os.path.basename(name)
            wsiname = wsiname.split('.')[0]
            wsidir = os.path.join(args.patch_dir, wsiname)
            wsidirs.append(wsidir)
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.wsidirs = wsidirs
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.level = 0
        self.size = 256

    def savetopndata(self,idxs, filename):
        slides = []
        grids = []
        targets = []
        topngrid = [self.grid[x] for x in idxs]
        topnid = [self.slideIDX[x] for x in idxs]
        topngrid = np.array(topngrid)
        topnid = np.array(topnid)
        for i in range(len(self.slidenames)):
            slides.append(self.slidenames[i])
            grid = topngrid[topnid==i]
            grid = grid.tolist()
            grids.append(grid)
            targets.append(self.targets[i])
        torch.save({
            'slides': slides,
            'grid': grids,
            'gridIDX': list(topnid),
            'targets': targets},
            os.path.join(args.output, f'{filename}.ckpt'))
            
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
        return self.t_data
        
    def __getitem__(self,index):
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        wsidir = self.wsidirs[slideIDX]
        img_path = os.path.join(wsidir, f"{coord[0]}_{coord[1]}.jpg")
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.grid)



class Traindataset(data.Dataset):
    def __init__(self, data, libraryfile='', transform=None):
        lib = torch.load(libraryfile, map_location='cpu')
        self.t_data = data
        wsidirs = []
        for i,name in enumerate(lib['slides']):
            wsiname = os.path.basename(name)
            wsiname = wsiname.split('.')[0]
            wsidir = os.path.join(args.patch_dir, wsiname)
            wsidirs.append(wsidir)
        #Flatten grid
        grid = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
        self.wsidirs = wsidirs
        self.grid = grid
        self.transform = transform

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        slideIDX, coord, target = self.t_data[index]
        wsidir = self.wsidirs[slideIDX]
        img_path = os.path.join(wsidir, f"{coord[0]}_{coord[1]}.jpg")
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.t_data)

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        temp = resnet34(pretrained=True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        self.feature_extractor = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x), x


if __name__ == '__main__':
    main()