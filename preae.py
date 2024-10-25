import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva

# the basic autoencoder
class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,  n_dec_1, n_dec_2,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder configuration
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)
        # decoder configuration
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)
        # degree
        #self.v = v

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# pre-train the autoencoder model
def pretrain_ae(model, dataset, args, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)#256
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            #x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #scheduler.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            #eva(y, y_pred, epoch)


        torch.save(model.state_dict(), 'D:/scAFC/scAFC-master/pkl/preae_{}.pkl'.format(args.name))

def create_label_mapping(labels):
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_to_int[label] for label in labels])
    return mapped_labels, label_to_int

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Bladder')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--preae_path', type=str, default='pkl')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    #device = torch.device("cpu")

    args.pretrain_path = 'D:/scAFC/scAFC-master/pkl/{}.pkl'.format(args.name)

    if args.name == 'Muraro':
        args.n_clusters = 10
        args.n_input = 2200
        args.epochs = 50
        args.lr = 0.0001

    if args.name == 'Darmanis':
        args.n_clusters = 8
        args.n_input = 6199
        args.epochs = 50
        args.lr = 0.0001

    if args.name == 'Pollen':
        args.n_clusters = 11
        args.n_input = 6347
        args.epochs = 60
        args.lr = 0.0001

    if args.name == 'Wang':
        args.n_clusters = 7
        args.n_input = 6702
        args.epochs = 100#300
        args.lr = 0.001

    if args.name == 'Baron':
        args.n_clusters = 14
        args.n_input = 1864
        args.epochs = 100
        args.lr = 0.001

    if args.name == 'Melanoma':
        args.n_clusters = 9
        args.n_input = 5072
        args.epochs = 120
        args.lr = 0.001

    if args.name == 'Romanov':
        args.n_clusters = 7
        args.n_input = 3878
        args.epochs = 80
        args.lr = 0.0005

    if args.name == 'Bladder':
        args.n_clusters = 4
        args.n_input = 2183
        args.epochs = 50
        args.lr = 0.001

    if args.name == 'Diaphragm':
        args.n_clusters = 5
        args.n_input = 4167
        args.epochs =100
        args.lr = 0.0001

    if args.name == 'Deng':
        args.lr = 0.0001
        args.epochs = 80
        args.n_clusters = 6
        args.n_input = 5605
   
    if args.name == 'Tosches':
        args.lr = 0.0001
        args.epochs = 40
        args.n_clusters = 15
        args.n_input = 2753
    

    x_path = 'D:/scAFC/scAFC-master/data/{}.txt'.format(args.name)
    y_path = 'D:/scAFC/scAFC-master/data/{}_labels.txt'.format(args.name)

    x = np.loadtxt(x_path, dtype=float)
    y = np.loadtxt(y_path, dtype=int)

    model = AE(
            n_enc_1=1024,
            n_enc_2=128,
            n_dec_1=128,
            n_dec_2=1024,
            n_input=args.n_input,
            n_z=16,).cuda()

    dataset = LoadDataset(x)
    pretrain_ae(model, dataset, args, y)
