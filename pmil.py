import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
import argparse
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.model_pmil import IClassifier, BClassifier, PBMIL, Euclidean_Similarity


np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Generating slide embedding features through prototypes and classification')
parser.add_argument('--train_lib', type=str, default='lib/train.ckpt',
                    help='lib to save wsi id of train set')
parser.add_argument('--val_lib', type=str, default='lib/val.ckpt',
                    help='lib to save wsi id of valid set')
parser.add_argument('--test_lib', type=str, default='lib/test.ckpt',
                    help='lib to save wsi id of test set')
parser.add_argument('--feature_dir', type=str, default='feat',
                    help='feature directory')
parser.add_argument('--output', type=str, default='result', help='output directory')
parser.add_argument('--num_cluster', type=int, help='number of cluster')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--nepochs', type=int, default=30, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--weights', default=0.5, type=float,
                    help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--global_cluster', type=str, default='cluster/prototypes_features_40x256.npy')
parser.add_argument('--pmil_model', type=str, default='model/pmil_model.pth', help='path to pretrained model')
parser.add_argument('--mil_model', type=str, default='model/mil.pth')
parser.add_argument('--s', default=5, type=int, help='how many top k patchess to consider')
parser.add_argument('--save_model', default=False, action='store_true')
parser.add_argument('--load_model', default=False, action='store_true')
parser.add_argument('--is_test', default=False, action='store_true')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--suffix', type=str, default='.csv')

global args
args = parser.parse_args()
torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


def main():
    best_acc = 0

    ## Loading prototype feature
    if os.path.basename(args.global_cluster).split('.')[-1] == 'npy':
        data = np.load(args.global_cluster)
        prototypes_features = data
    elif os.path.basename(args.global_cluster).split('.')[-1] == 'npz':
        data = np.load(args.global_cluster)
        prototypes_features = data['feature']
    args.num_cluster = prototypes_features.shape[0]
    print(args.num_cluster)
    prototypes_features = torch.from_numpy(prototypes_features)
    prototypes_features = prototypes_features.cuda()

    ## Creating model
    i_classifier = IClassifier(mil_model=args.mil_model, feature_size=512, out_class=2).to(device)
    b_classifier = BClassifier(num_cluster=args.num_cluster, feature_size=512, input_size=128, output_class=2, dropout=0).to(device)
    model = PBMIL(i_classifier, b_classifier).to(device)
    if args.load_model:
        ch = torch.load(args.pmil_model, map_location='cpu')
        model.load_state_dict(ch['state_dict'])
    if args.weights == 0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1 - args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(weight=w).cuda()

    ## Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1, last_epoch=-1)
    cudnn.benchmark = True

    # Loading dataset
    if args.train_lib:
        train_dset = GlobalDataset(feature_dir=args.feature_dir, libraryfile=args.train_lib, is_train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=1, shuffle=True,
            num_workers=1, pin_memory=False)
    if args.val_lib:
        val_dset = GlobalDataset(feature_dir=args.feature_dir, libraryfile=args.val_lib, is_train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=False)
    test_dset = GlobalDataset(feature_dir=args.feature_dir, libraryfile=args.test_lib, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False)

    prototype = prototypes_features
    # loop throuh epochs
    if not args.is_test:
        for epoch in range(args.nepochs):
            ## train
            prototype = prototypes_features
            loss, acc = train(epoch, prototype, train_loader, model, criterion, optimizer)
            scheduler.step()

            ## Validation
            val_loss, probs, val_acc, val_auc, idx, codings = test(epoch, prototype, val_loader, model, criterion)
            targets = np.array(test_dset.targets)
            print('Testing Loss: {:.4f}\tAcc: {:.4f}\tAUC: {:.4f}'.format(val_loss, val_acc, val_auc))

            ## Save best model
            if val_acc >= best_acc:
                best_acc = val_acc
                best_auc = val_auc
                best_loss = val_loss
                print("Best acc: {:.4f} | Best auc: {:.4f}".format(best_acc, best_auc))

                if args.save_model:
                    obj = {
                        'epoch': epoch + 1,
                        'i_state_dict': model.i_classifier.state_dict(),
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'best_auc': best_auc,
                        'idx': idx,
                        'features': codings,
                        'targets': targets
                    }
                    torch.save(obj, args.pmil_model)

    ## Testing
    ch = torch.load(args.pmil_model, map_location='cpu')
    model.load_state_dict(ch['state_dict'])
    test_loss, probs, test_acc, auc, idx, codings = test(0, prototype, test_loader, model, criterion)
    print('Testing Loss: {:.4f}\tAcc: {:.4f}\tAUC: {:.4f}'.format(test_loss, test_acc, auc))


def train(run, prototype, loader, model, criterion, optimizer):
    model.eval()
    running_loss = 0.
    running_bag_loss = 0.
    running_max_loss = 0.
    probs = []
    targets = []
    for i, (input, target, patient_id) in enumerate(loader):
        input = input[0]
        input = input.cuda()
        target = target.cuda()

        ins_prediction, bag_prediction, _ = model(input, prototype)

        score = F.softmax(ins_prediction, dim=1)
        _, max_id = torch.max(score[:, 1], 0)

        max_prediction = ins_prediction[max_id, :]
        bag_loss = criterion(bag_prediction.view(1, -1), target)
        max_loss = criterion(max_prediction.view(1, -1), target)
        loss = bag_loss + max_loss

        running_bag_loss += bag_loss.item()
        running_max_loss += max_loss.item()

        train_prediction = F.softmax(bag_prediction.detach(),  dim=1).squeeze().cpu().numpy()
        probs.append(train_prediction[1])
        targets.append(target.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        torch.cuda.empty_cache()
    targets = np.array(targets)
    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    _, _, threshold_optimal = optimal_thresh(fpr, tpr, thresholds)
    preds = [1 if x >= threshold_optimal else 0 for x in probs]
    preds = np.array(preds)
    eq = np.equal(targets, preds)
    acc = float(eq.sum()) / targets.shape[0]

    running_loss = running_loss / len(loader.dataset)
    running_max_loss = running_max_loss / len(loader.dataset)
    running_bag_loss = running_bag_loss / len(loader.dataset)
    print('Training\tEpoch: [{}/{}]\tMax Loss: {:.4f}\tBag Loss: {:.4f}\tAcc: {:.4f}AUC: {:.4f}'.format(run + 1,
                                                                                                        args.nepochs,
                                                                                                        running_max_loss,
                                                                                                        running_bag_loss,
                                                                                                        acc, roc_auc))
    return running_loss, acc


def test(run, prototype, loader, model, criterion):
    model.eval()
    running_loss = 0.
    running_bag_loss = 0.
    running_max_loss = 0.
    probs = []
    targets = []
    ids = []
    codings = []
    with torch.no_grad():
        for i, (input, target, patient_id) in enumerate(loader):
            input = input[0]
            input = input.cuda()
            target = target.cuda()
            ins_prediction, bag_prediction, coding = model(input, prototype)
            codings.append(coding.cpu().numpy())
            score = F.softmax(ins_prediction, dim=1)
            _, max_id = torch.max(score[:, 1], 0)
            _, id = torch.topk(score[:, 1], dim=0, k=5)

            ids.append(id)
            max_prediction = ins_prediction[max_id, :]
            bag_loss = criterion(bag_prediction.view(1, -1), target)
            max_loss = criterion(max_prediction.view(1, -1), target)
            loss = bag_loss + max_loss
            running_bag_loss += bag_loss.item()
            running_max_loss += max_loss.item()
            test_prediction = F.softmax(bag_prediction.detach(), dim=1).squeeze().cpu().numpy()
            probs.append(test_prediction[1])
            targets.append(target.item())

            running_loss += loss.item()
            torch.cuda.empty_cache()

    codings = np.array(codings)
    targets = np.array(targets)
    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    _, _, threshold_optimal = optimal_thresh(fpr, tpr, thresholds)
    preds = [1 if x >= threshold_optimal else 0 for x in probs]
    preds = np.array(preds)
    eq = np.equal(targets, preds)
    acc = float(eq.sum()) / targets.shape[0]
    running_loss = running_loss / len(loader.dataset)
    return running_loss, np.array(probs), acc, roc_auc, ids, codings


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]



class GlobalDataset(data.Dataset):
    columns = []
    for i in range(1, 513):
        columns.append('feature' + str(i))

    def __init__(self, feature_dir='', libraryfile='', is_train=True):
        self.feature_dir = feature_dir
        self.is_train = is_train
        # self.feature_files = os.listdir(self.feature_dir)
        lib = torch.load(libraryfile)
        self.slidenames = [os.path.basename(slide).split('.')[0] for slide in lib['slides']]
        gt_targets = lib['targets']
        self.targets = self._get_label(gt_targets)
        self.micrometastasis = np.load('micrometastasis/micro_wsi_names_trainset.npy')
        self.micrometastasis = self.micrometastasis.tolist()
        self.features = self._get_features()

    def _get_label(self, gt_targets):
        targets = []
        for slidename in self.slidenames:
            idx = self.slidenames.index(slidename)
            targets.append(gt_targets[idx])
        return targets

    def _get_features(self):
        all_features = []
        for i, slidename in enumerate(self.slidenames):
            if args.suffix == '.csv':
                df = pd.read_csv(os.path.join(self.feature_dir, slidename + '.csv'))
                features = df[self.columns].values
            elif args.suffix == '.npy':
                features = np.load(os.path.join(self.feature_dir, slidename + '.npy'))
            features = features.astype(np.float32)
            all_features.append(features)
        return all_features

    def __getitem__(self, index):
        features = self.features[index]
        target = self.targets[index]
        patient_id = self.slidenames[index]
        return features, target, patient_id

    def __len__(self):
        return len(self.slidenames)


if __name__ == '__main__':
    main()
