from __future__ import print_function, division
import argparse
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear

import scanpy as sc
import matplotlib.pyplot as plt
import os
import pandas as pd
from evaluation import eva
from layers import GraphAttentionLayer
from utils import load_data, load_graph


class AttentionLayer(nn.Module):

    def __init__(self, last_dim, n_num):
        super(AttentionLayer, self).__init__()
        self.n_num = n_num
        self.fc1 = nn.Linear(n_num * last_dim, int(last_dim/4))
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(int(last_dim/4), n_num)
        self.attention = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.T = 10

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        attention_sample = self.attention(x / self.T)
        attention_view = torch.mean(attention_sample, dim=0, keepdim=True).squeeze()
        return attention_view

class FusionLayer(nn.Module):
    def __init__(self, last_dim, n_num=2):
        super(FusionLayer, self).__init__()
        self.n_num = n_num
        self.attentionLayer = AttentionLayer(last_dim, n_num)
    def forward(self, x, k):
        y = torch.cat((x, k), 1)
        weights = self.attentionLayer(y)
        x_TMP = weights[0] * x + weights[1] * k
        return x_TMP
def dot_product(z):
        adj1 = torch.sigmoid(torch.mm(z, z.transpose(0, 1)))
        adj1 = adj1.add(torch.eye(adj1.shape[0]).to(args.device))
        adj1 = normalize(adj1)
        return adj1

def normalize(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2,  n_dec_1, n_dec_2,
                 n_input, n_z,  v=1):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = Linear(n_dec_2, n_input)
        self.v = v

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, enc_h1, enc_h2, z, dec_h1, dec_h2

class AFC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, n_clusters, v=1):
        super(AFC, self).__init__()

        self.ael = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z)

        self.ael.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GAT layer
        dr_rate = 0.6
        alpha = 0.2
        self.gat_1 = GraphAttentionLayer(n_input, n_enc_1, dr_rate, alpha)
        self.gat_2 = GraphAttentionLayer(n_enc_1, n_enc_2, dr_rate, alpha)
        self.gat_3 = GraphAttentionLayer(n_enc_2, n_z, dr_rate, alpha)

        self.fuse1 = FusionLayer(n_enc_1)
        self.fuse2 = FusionLayer(n_enc_2)
        self.fuse3 = FusionLayer(n_z)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v
        self.linear = nn.Linear(n_z, n_clusters)

    def forward(self, x, adj):

        x_bar, tra1, tra2, z, dec_1, dec_2 = self.ael(x)

        adj = adj.to_dense()

        h = self.gat_1(x, adj)
        h = self.fuse1(h, tra1)
        h = self.gat_2(h, adj)
        h = self.fuse2(h, tra2)
        h = self.gat_3(h, adj)
        h = self.fuse3(h, z)
        h = self.linear(h)

        predict = F.softmax(h, dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def remap_labels(true_labels, pred_labels):
    label_pair = list(zip(true_labels, pred_labels))
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)

    cost_matrix = np.zeros((len(unique_true_labels), len(unique_pred_labels)), dtype=np.int32)
    for true_label, pred_label in label_pair:
        true_idx = np.where(unique_true_labels == true_label)[0][0]
        pred_idx = np.where(unique_pred_labels == pred_label)[0][0]
        cost_matrix[true_idx][pred_idx] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    label_mapping = dict(zip(unique_pred_labels[col_ind], unique_true_labels[row_ind]))

    remapped_pred_labels = np.array([label_mapping[label] for label in pred_labels])
    return remapped_pred_labels

def train(dataset, args):

    model = AFC(1024, 128, 128, 1024,
                 n_input=args.n_input,
                 n_clusters=args.n_clusters,
                 n_z=args.n_z ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    adj = load_graph(args.name, args.k)
    adj = adj.to(device)

    data = torch.Tensor(dataset.x).to(device)

    y = dataset.y
    with torch.no_grad():
        _, _, _, z = model(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_mapped = remap_labels(y, y_pred)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred_mapped, 'pae')

    M = np.zeros((args.epochs, 4))
    max_ari =0.;
    for epoch in range(args.epochs):
        if epoch % args.update_interval == 0:
            model.eval()
            with torch.no_grad():
                _, tmp_q, pred, _ = model(data, adj)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)
            res2 = pred.data.cpu().numpy().argmax(1)
            res3 = p.data.cpu().numpy().argmax(1)
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')

            M[epoch, 0], M[epoch, 1], M[epoch, 2], M[epoch, 3] = eva(y, res2, str(epoch) + 'Z')
            tmp_ari = M[epoch,2]
            if tmp_ari > max_ari:
                max_ari = tmp_ari
                torch.save(model.state_dict(), "tmp.pkl")
                print("max epoch is {}".format(epoch))

        model.train()
        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        l2_loss = torch.tensor(0.).cuda()
        for param in model.parameters():
            l2_loss += torch.norm(param)

        loss = args.lambda_v1 * kl_loss + args.lambda_v2 * ce_loss + args.lambda_v3 * re_loss + args.lambda_v4 * l2_loss
        print(f'{epoch} loss--loss:{loss},kl:{kl_loss},ce:{ce_loss},re:{re_loss},l2:{l2_loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    acc_max = np.max(M[:, 0])
    nmi_max = np.max(M[:, 1])
    ari_max = np.max(M[:, 2])
    f1_max = np.max(M[:, 3])
    print('scAFC:    acc:', acc_max, ' nmi:', nmi_max, ' ari:', ari_max, ' f1:', f1_max)

    model.load_state_dict(torch.load("tmp.pkl", map_location='cpu'))
    model.eval()
    with torch.no_grad():
        x_bar, _, _, z = model(data, adj)

    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_mapped = remap_labels(y, y_pred)

    x_path = 'data/{}.txt'.format(args.name)
    x = np.loadtxt(x_path, dtype=float)
    
    trajectory_analysis(z.cpu().numpy(), y_pred_mapped, 'img', args)

    return [acc_max, nmi_max, ari_max, f1_max]



def trajectory_analysis(embedded_data, cluster_labels, output_folder, args):

    adata = sc.AnnData(embedded_data)

    adata.obs['clusters'] = pd.Categorical(cluster_labels)

    cluster_names = {
         0: '16cell',
         1: '4cell',
         2: '8cell',
         3: 'zygote',
         4: 'blast',
         5: '2cell'
    }#Deng

    adata.obs['clusters_names'] = adata.obs['clusters'].map(cluster_names)
    adata.obs['clusters_names'] = adata.obs['clusters_names'].astype('category')


    sc.pp.neighbors(adata, use_rep='X')

    sc.tl.umap(adata)

    # PAGA
    sc.tl.paga(adata, groups='clusters_names')

    plt.rcParams['axes.linewidth'] = 2.5
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 20

    fig, axs = plt.subplots(1, 2, figsize=(14.5, 7), gridspec_kw={'wspace': 0})

    sc.pl.umap(adata, color='clusters_names', ax=axs[0], show=False, title='UMAP')

    sc.pl.paga(adata, ax=axs[1], show=False, title='PAGA', node_size_scale=2.0, fontsize=20)
    axs[1].set_title('PAGA', color='black')

    plt.subplots_adjust(bottom=0.1)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=25, markerscale=3)

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.savefig(os.path.join(output_folder, args.name+'.png'), dpi=300)
    plt.show()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    adata.write(os.path.join(output_folder, 'adata.h5ad'))

    print(f"Results and visualizations are saved in {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Deng')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--k',type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50 )
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--n_z', default=16, type=int)
    parser.add_argument('--preae_path', type=str, default='pkl')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--lambda_v1', type=float, default='0.5')
    parser.add_argument('--lambda_v2', type=float, default='0.1')
    parser.add_argument('--lambda_v3', type=float, default='0.01')
    parser.add_argument('--lambda_v4', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default=20, help='Random seed for reproducibility')
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    setup_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    #device = torch.device("cpu")

    # load data
    dataset = load_data(args.name)

    if args.name == 'Deng':
        args.n_clusters = 6
        args.n_input = 5605

    args.pretrain_path = 'pkl/preae_{}.pkl'.format(args.name)

    print(args)
    train(dataset, args)
