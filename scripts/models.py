import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
import pandas as pd
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from model_transformer import *

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score, f1_score

from tqdm import trange
from scipy.stats import pearsonr
import time
from scipy.special import softmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MISSING_NUM = -100


class BasicBlock(nn.Module):
    def __init__(self, hidden_width, module_length=2):
        super(BasicBlock, self).__init__()
        self.hidden = nn.ModuleList()
        for k in range(module_length):
            self.hidden.append(
                nn.Sequential(nn.Linear(hidden_width, hidden_width)
                              )
            )

    def forward(self, x):
        activation = torch.relu
        identity = x
        out = x
        for layer in self.hidden:
            out = activation(layer(out))
        out += identity

        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = torch.mean(x, dim=1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = torch.max(x, dim=1)[0]
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(1).expand_as(x)
        return x * scale


def logistic(x):
    return 1 / (1 + torch.exp(-x))


def corr_loss(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    loss = 50 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))

    return loss


class BottleNeck(nn.Module):
    def __init__(self, hidden_width, group=20):
        super(BottleNeck, self).__init__()
        self.groups = nn.ModuleList()
        for g in range(group):
            group_layers = nn.ModuleList()
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_width, hidden_width // group)
                              ))
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_width // group, hidden_width // group)
                              ))
            self.groups.append(group_layers)

    def forward(self, x):
        activation = logistic
        identity = x
        out = []
        for group_layers in self.groups:
            group_out = x
            for layer in group_layers:
                group_out = activation(layer(group_out))

            out.append(group_out)
        out = torch.cat(out, dim=1)
        out += identity
        return out


class DeepOmicNetBaseline(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width):
        super(DeepOmicNetBaseline, self).__init__()

        self.input = nn.Linear(in_dim, hidden_width)
        # self.hidden = nn.Linear(hidden_width, hidden_width // 2)
        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        # x = activation(self.hidden(x))
        x = self.output(x)
        return x


class DeepMultiOmicPathwayNet(nn.Module):
    def __init__(self, hidden_width, num_omics, out_dim, gene_to_id, id_to_gene, pathway_dict, non_cancer_genes,
                 equal_width=False, only_cancer_genes=False):
        """
        :param hidden_width:
        :param out_dim:
        :param gene_to_id:
        :param id_to_gene:
        :param pathway_dict: {string: list}
        """
        super(DeepMultiOmicPathwayNet, self).__init__()
        self.pathway_dict = pathway_dict
        self.pathway_layers = nn.ModuleDict()
        self.non_cancer_genes = non_cancer_genes
        self.gene_to_id = gene_to_id
        self.id_to_gene = id_to_gene
        self.num_omics = num_omics
        self.non_cancer_width = hidden_width
        self.only_cancer_genes = only_cancer_genes

        pathway_total_width = 0
        for key in self.pathway_dict:
            num_genes_in_pathway = len(self.pathway_dict[key])
            if not equal_width:
                pathway_width = num_genes_in_pathway
            else:
                pathway_width = 1
            self.pathway_layers[key] = nn.Linear(num_genes_in_pathway * num_omics, pathway_width)
            pathway_total_width += pathway_width

        self.non_cancer_layer = nn.Linear(len(non_cancer_genes) * num_omics, self.non_cancer_width)
        if self.only_cancer_genes:
            self.output = nn.Linear(pathway_total_width, out_dim)
        else:
            self.output = nn.Linear(pathway_total_width + self.non_cancer_width, out_dim)

    def forward(self, x, mode='training'):
        """

        :param mode:
        :param x: shape [N, G, C]
        :return:
        """
        pathway_x = []
        activation = logistic
        for key in self.pathway_dict:
            gene_ids = [self.gene_to_id[x] for x in self.pathway_dict[key]]
            tmp = self.pathway_layers[key](
                x[:, gene_ids, :].reshape(-1, len(gene_ids) * self.num_omics))
            tmp = tmp / torch.linalg.norm(tmp, dim=1, keepdim=True).expand_as(tmp)
            pathway_x.append(tmp)  # shape:[N,Gs,3] -> [N,Gs]
        pathway_x = torch.cat(pathway_x, dim=1)  # [N,G]

        non_cancer_gene_x = None
        if self.only_cancer_genes:
            x = self.output(activation(pathway_x))
        else:
            non_cancer_gene_ids = [self.gene_to_id[x] for x in self.non_cancer_genes]
            non_cancer_gene_x = self.non_cancer_layer(
                x[:, non_cancer_gene_ids, :].reshape(-1, len(non_cancer_gene_ids) * self.num_omics))
            x = self.output(activation(torch.cat([pathway_x, non_cancer_gene_x], dim=1)))

        if mode == "training":
            return x
        elif mode == 'explain':
            pathway_x.retain_grad()
            if non_cancer_gene_x is not None:
                non_cancer_gene_x.retain_grad()
            return x, pathway_x, non_cancer_gene_x
        else:
            raise Exception


class DeepOmicNetS(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_size=2):
        super(DeepOmicNetS, self).__init__()

        self.input = nn.Linear(in_dim, hidden_width)
        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BasicBlock(hidden_width))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class DeepOmicNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width, hidden_size=2, group=20):
        super(DeepOmicNet, self).__init__()

        self.input = nn.Linear(in_dim, hidden_width)
        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BottleNeck(hidden_width, group=group))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x = activation(self.input(x))
        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class DeepMultiOmicNet(nn.Module):
    def __init__(self, in_dim, num_omics, out_dim, hidden_width, hidden_size=2, group=20):
        super(DeepMultiOmicNet, self).__init__()
        # hard-code two streams
        self.hidden_width = hidden_width
        self.stream_0 = nn.ModuleList()
        self.stream_1 = nn.ModuleList()
        self.num_omics = num_omics
        for i in range(num_omics):
            self.stream_0.append(nn.Linear(in_dim, hidden_width))
            self.stream_1.append(nn.Linear(in_dim, hidden_width))

        self.hidden_0 = nn.ModuleList()
        self.hidden_1 = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden_0.append(BottleNeck(hidden_width, group=group))
            self.hidden_1.append(BottleNeck(hidden_width, group=group))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x_0 = []
        x_1 = []
        for i in range(self.num_omics):
            x_0.append(self.stream_0[i](x[:, i, :]))
            x_1.append(self.stream_1[i](x[:, i, :]))
        x_0 = torch.sum(torch.stack(x_0, dim=1), dim=1)
        x_1 = torch.sum(torch.stack(x_1, dim=1), dim=1)
        x_0 = activation(x_0)
        x_1 = activation(x_1)
        for layer in self.hidden_0:
            x_0 = activation(layer(x_0))
        for layer in self.hidden_1:
            x_1 = activation(layer(x_1))
        x = x_0 + x_1
        x = self.output(x)
        return x


class MultiOmicDrugResXNNV2(nn.Module):
    """
    one hidden layer then combine
    """

    def __init__(self, in_dim, num_omics, out_dim, hidden_width, hidden_size=2, group=20):
        super(MultiOmicDrugResXNNV2, self).__init__()
        # hard-code two streams
        self.hidden_width = hidden_width
        self.input = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.in_dim = in_dim
        self.num_omics = num_omics
        for i in range(num_omics):
            self.input.append(nn.Linear(in_dim, hidden_width))
        for i in range(in_dim):
            self.merge.append(nn.Linear(num_omics, 1))

        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BottleNeck(hidden_width, group=group))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic
        x_0 = []
        for i in range(self.num_omics):
            x_0.append(self.input[i](x[:, i, :]))
        # x_0 now has 3 x num_genes
        x_0 = torch.stack(x_0, dim=1)  # N x 3 x protein
        x_0 = activation(x_0)
        merge_values = []
        for i in range(self.hidden_width):
            merge_values.append(torch.flatten(activation(self.merge[i](x_0[:, :, i]))))  # N x 3 x 3000 -> list of N

        merge_values = torch.stack(merge_values, dim=1)  # N x 3000

        for layer in self.hidden:
            x_0 = activation(layer(merge_values))
        x = self.output(x_0)
        return x


class OmicLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, num_omics):
        super().__init__()
        self.size_in, self.num_omics = size_in, num_omics
        weights = torch.Tensor(num_omics, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_in)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x = (x * self.weights.t()).sum(dim=2)
        return torch.add(w_times_x, self.bias)  # w times x + b


class DeepMultiOmicNetV3S(nn.Module):
    """
    directly combine genes then linears
    """

    def __init__(self, in_dim, num_omics, out_dim, hidden_width, hidden_size=2):
        super(DeepMultiOmicNetV3S, self).__init__()
        # hard-code two streams
        self.hidden_width = hidden_width
        self.in_dim = in_dim
        self.num_omics = num_omics
        self.omic_layer = nn.Conv1d(num_omics, 4, kernel_size=1)
        self.hidden_0 = nn.Linear(in_dim * 4, hidden_width)

        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BasicBlock(hidden_width))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic

        # merge_values = []
        # for i in range(self.in_dim):
        #     merge_values.append(torch.flatten(activation(self.merge[i](x[:, :, i]))))  # N x 3 x 3000 -> list of N
        x = x.permute(0, 2, 1)
        x = activation(self.omic_layer(x))
        x = torch.flatten(x, start_dim=1)
        # merge_values = torch.stack(merge_values, dim=1)  # N x 3000

        x = activation(self.hidden_0(x))

        # for layer in self.hidden:
        #     x = activation(layer(x))
        x = self.output(x)
        return x


class DeepMultiOmicNetV3(nn.Module):
    """
    directly combine genes then linears
    """

    def __init__(self, in_dim, num_omics, out_dim, hidden_width, hidden_size=2, group=20):
        super(DeepMultiOmicNetV3, self).__init__()
        # hard-code two streams
        self.hidden_width = hidden_width
        self.merge = nn.ModuleList()
        self.in_dim = in_dim
        self.num_omics = num_omics
        self.hidden_0 = nn.Linear(in_dim, hidden_width)

        for i in range(in_dim):
            self.merge.append(nn.Linear(num_omics, 1))

        self.hidden = nn.ModuleList()
        for k in range(hidden_size):
            self.hidden.append(BottleNeck(hidden_width, group=group))

        self.output = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        activation = logistic

        merge_values = []
        for i in range(self.in_dim):
            merge_values.append(torch.flatten(activation(self.merge[i](x[:, i, :]))))

        merge_values = torch.stack(merge_values, dim=1)  # N x 3000

        x = activation(self.hidden_0(merge_values))

        for layer in self.hidden:
            x = activation(layer(x))
        x = self.output(x)
        return x


class DOIT(nn.Module):
    def __init__(self, num_omics, out_dim, gene_to_id, id_to_gene, pathway_dict, non_cancer_genes, dim,
                 depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0., only_cancer_genes=False):
        super().__init__()
        self.pathway_dict = pathway_dict
        self.pathway_layers = nn.ModuleDict()
        self.non_cancer_genes = non_cancer_genes
        self.gene_to_id = gene_to_id
        self.id_to_gene = id_to_gene
        self.num_omics = num_omics
        self.only_cancer_genes = only_cancer_genes

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        for key in self.pathway_dict:
            num_genes_in_pathway = len(self.pathway_dict[key])
            pathway_width = dim
            self.pathway_layers[key] = nn.Linear(num_genes_in_pathway * num_omics, pathway_width)

        self.non_cancer_layer = nn.Linear(len(non_cancer_genes) * num_omics, dim)

        self.pathway_embedding = nn.Parameter(torch.randn(1, len(pathway_dict) + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, out_dim)
        )

    def forward(self, x):
        pathway_x = []
        for key in self.pathway_dict:
            gene_ids = [self.gene_to_id[x] for x in self.pathway_dict[key]]
            tmp = self.pathway_layers[key](
                x[:, gene_ids, :].reshape(-1, len(gene_ids) * self.num_omics))
            # tmp = tmp / torch.linalg.norm(tmp, dim=1, keepdim=True).expand_as(tmp)
            pathway_x.append(tmp)  # shape:[b,g,3] -> [b,g]
        pathway_x = torch.stack(pathway_x, dim=1)  # [b, p, g]
        if self.only_cancer_genes:
            x = pathway_x
        else:
            non_cancer_gene_ids = [self.gene_to_id[x] for x in self.non_cancer_genes]
            non_cancer_gene_x = self.non_cancer_layer(
                x[:, non_cancer_gene_ids, :].reshape(-1, len(non_cancer_gene_ids) * self.num_omics))
            non_cancer_gene_x = non_cancer_gene_x.unsqueeze(1)
            x = torch.cat([pathway_x, non_cancer_gene_x], dim=1)  # [b, p+1, g]

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pathway_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class SingleOmicDataset(Dataset):
    def __init__(self, data_df, purpose_data_df, mode, logger=None):
        assert mode in ['train', 'val', 'test']

        self.df = np.nan_to_num(data_df, nan=0)
        self.purpose_data = np.nan_to_num(purpose_data_df, nan=MISSING_NUM)

        assert self.df.shape[0] == self.purpose_data.shape[0], f"{self.df.shape[0]}, {self.purpose_data.shape[0]}"
        self.mode = mode
        if logger:
            logger.info(f"mode: {mode}, df shape: {self.df.shape}, purpose_data shape: {self.purpose_data.shape}")

    def __getitem__(self, index):
        """ Returns: tuple (sample, target) """
        data = self.df[index, :]
        if len(self.purpose_data.shape) > 1:
            target = self.purpose_data[index, :]  # the first col is cell line name
        else:
            target = self.purpose_data[index]

        # no other preprocessing for now

        return data, target

    def __len__(self):
        return self.df.shape[0]


def get_multiomic_df(df, omics_types):
    gene_columns = [x for x in df.columns if 'tissue' not in x]
    tissue_df = df[[x for x in df.columns if 'tissue' in x]].values
    genes = np.unique(([x.split("_")[0] for x in gene_columns]))

    not_covered = [f"{x}_{omic}" for x in genes for omic in omics_types if f"{x}_{omic}" not in gene_columns]

    df_zeros = pd.DataFrame(np.zeros((df.shape[0], len(not_covered))), columns=not_covered, index=df.index)
    df_combined = pd.concat([df, df_zeros], axis=1)
    df_multiomic = np.zeros((df.shape[0], len(genes), len([x for x in omics_types if
                                                           x != 'tissue'])))
    for i in range(len(genes)):
        df_multiomic[:, i, :] = df_combined[[f"{genes[i]}_{omic}" for omic in omics_types]].values

    genes_to_id = dict(zip(genes, range(len(genes))))
    id_to_genes = dict(zip(range(len(genes)), genes))
    return df_multiomic, genes_to_id, id_to_genes, tissue_df


class MultiOmicDataset(Dataset):
    def __init__(self, df, purpose_data_df, mode, omics_types, logger=None, with_tissue=False):
        assert mode in ['train', 'val', 'test']
        self.df, self.genes_to_id, self.id_to_genes, self.tissue_df = get_multiomic_df(df, omics_types)
        self.omics_types = omics_types
        self.purpose_data = np.nan_to_num(purpose_data_df, nan=MISSING_NUM)
        self.with_tissue = with_tissue

        assert self.df.shape[0] == self.purpose_data.shape[0], f"{self.df.shape[0]}, {self.purpose_data.shape[0]}"
        self.mode = mode
        if logger:
            logger.info(f"mode: {mode}, df shape: {self.df.shape}, purpose_data shape: {self.purpose_data.shape}")

    def __getitem__(self, index):
        """ Returns: tuple (sample, target) """
        data = self.df[index, :, :]  # (b, genes, omics)
        tissue_data = self.tissue_df[index, :]
        target = self.purpose_data[index, :]
        if self.with_tissue:
            return data, tissue_data, target
        else:
            return data, target

    def __len__(self):
        return self.df.shape[0]


class MultiOmicMulticlassDataset(Dataset):
    def __init__(self, df, purpose_data_df, mode, omics_types, class_name_to_id, logger=None):
        assert mode in ['train', 'val', 'test']
        self.df, self.genes_to_id, self.id_to_genes, self.tissue_df = get_multiomic_df(df, omics_types)
        self.df = np.nan_to_num(self.df, nan=0)
        self.omics_types = omics_types
        self.purpose_data = np.nan_to_num(purpose_data_df, nan=MISSING_NUM)
        self.class_name_to_id = class_name_to_id

        assert self.df.shape[0] == self.purpose_data.shape[0], f"{self.df.shape[0]}, {self.purpose_data.shape[0]}"
        self.mode = mode
        if logger:
            logger.info(f"mode: {mode}, df shape: {self.df.shape}, purpose_data shape: {self.purpose_data.shape}")
            logger.info(self.class_name_to_id)

    def __getitem__(self, index):
        """ Returns: tuple (sample, target) """
        data = self.df[index, :, :]  # (b, genes, omics)
        target = self.purpose_data[index, :]
        target_id = np.array([self.class_name_to_id[x] for x in target])
        return data, target_id

    def __len__(self):
        return self.df.shape[0]


class AverageMeter:
    ''' Computes and stores the average and current value '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_r2 = AverageMeter()
    avg_mae = AverageMeter()
    avg_rmse = AverageMeter()
    avg_corr = AverageMeter()

    model.train()

    end = time.time()
    lr_str = ''

    for i, data in enumerate(train_loader):

        if len(data) == 2:
            (input_, targets) = data
            output = model(input_.float().to(device))
        elif len(data) == 3:
            (input_, tissue_x, targets) = data
            output = model(input_.float().to(device), tissue_x.float().to(device))
        else:
            raise Exception

        output[targets == MISSING_NUM] = MISSING_NUM

        loss = criterion(output, targets.float().to(device))
        targets = targets.cpu().numpy()

        confs = output.detach().cpu().numpy()
        if not np.isinf(confs).any() and not np.isnan(confs).any():
            try:
                avg_r2.update(np.median(
                    [r2_score(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])
                     for i in range(confs.shape[1])]))
                avg_mae.update(np.median(
                    [mean_absolute_error(targets[targets[:, i] != MISSING_NUM, i],
                                         confs[targets[:, i] != MISSING_NUM, i])
                     for i in range(confs.shape[1])]))
                avg_rmse.update(np.median(
                    [mean_squared_error(targets[targets[:, i] != MISSING_NUM, i],
                                        confs[targets[:, i] != MISSING_NUM, i],
                                        squared=True)
                     for i in range(confs.shape[1])]))
                avg_corr.update(np.median(
                    [pearsonr(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])[0]
                     for i in range(confs.shape[1])][0]))
            except ValueError:
                logger.info("skipping training score")

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(f'{epoch} \t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'corr {avg_corr.val:.4f} ({avg_corr.avg:.4f})\t'
                f'R2 {avg_r2.val:.4f} ({avg_r2.avg:.4f})\t'
                f'MAE {avg_mae.val:.4f} ({avg_mae.avg:.4f})\t'
                f'RMSE {avg_rmse.val:.4f} ({avg_rmse.avg:.4f})\t' + lr_str)

    return avg_r2.avg


def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_confs, all_targets = [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if len(data) == 2:
                (input_, target) = data
                output = model(input_.float().to(device))
            elif len(data) == 3:
                (input_, tissue_type, target) = data
                output = model(input_.float().to(device), tissue_type.float().to(device))
            else:
                raise Exception

            # output = model(input_.float().to(device))
            all_confs.append(output)

            if target is not None:
                all_targets.append(target)

    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    targets = targets.cpu().numpy()
    confs = confs.cpu().numpy()

    return confs, targets


def validate(val_loader, model, val_drug_ids, run=None, epoch=None, val_score_dict=None):
    confs, targets = inference(val_loader, model)

    r2_avg, mae_avg, rmse_avg, corr_avg = None, None, None, None
    if not np.isinf(confs).any() and not np.isnan(confs).any():
        if 'drug_id' in val_score_dict:
            val_score_dict['drug_id'].extend(val_drug_ids)
        elif 'Gene' in val_score_dict:
            val_score_dict['Gene'].extend(val_drug_ids)
        else:
            raise
        val_score_dict['run'].extend([run] * len(val_drug_ids))
        val_score_dict['epoch'].extend([epoch] * len(val_drug_ids))

        r2 = [r2_score(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])
              for i in range(confs.shape[1])]
        r2_avg = np.median(r2)

        mae = [mean_absolute_error(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])
               for i in range(confs.shape[1])]
        mae_avg = np.median(mae)

        rmse = [mean_squared_error(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i],
                                   squared=False)
                for i in range(confs.shape[1])]
        rmse_avg = np.median(rmse)

        corr = [pearsonr(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])[0]
                for i in range(confs.shape[1])]
        corr_avg = np.median(corr)

        val_score_dict['mae'].extend(mae)
        val_score_dict['rmse'].extend(rmse)
        val_score_dict['corr'].extend(corr)
        val_score_dict['r2'].extend(r2)

    return r2_avg, mae_avg, rmse_avg, corr_avg


def get_model_filename(drug_id):
    drug_name = drug_id.replace(';', '_')
    drug_name = drug_name.replace('/', '_')
    drug_name = drug_name.replace(' ', '')
    drug_name = drug_name.replace('(', '')
    drug_name = drug_name.replace(')', '')
    drug_name = drug_name.replace('+', '_')
    drug_name = drug_name.replace(',', '_')
    return drug_name


def train_loop(epochs, train_loader, val_loader, model, criterion, optimizer, logger, stamp,
               configs,
               lr_scheduler=None,
               val_drug_ids=None,
               run=None, val_score_dict=None, id_to_class_name=None):
    train_res = []
    val_res = []
    if configs['task'] == 'regression':
        best_r2 = 0.05
        for epoch in trange(1, epochs + 1):
            if lr_scheduler:
                logger.info(f"learning rate: {lr_scheduler.get_last_lr()}")
            train_score = train(train_loader,
                                model,
                                criterion,
                                optimizer,
                                epoch,
                                logger)

            train_res.append(train_score)
            if lr_scheduler:
                lr_scheduler.step()

            r2, mae, rmse, corr = validate(val_loader, model, val_drug_ids, run=run, epoch=epoch,
                                           val_score_dict=val_score_dict)

            if r2 and mae and rmse and corr:
                logger.info(f"Epoch {epoch} validation corr:{corr:4f}, R2:{r2:4f}, MAE:{mae:4f}, RMSE:{rmse:4f}")
            else:
                logger.info(f"Epoch {epoch} validation Inf")
            if configs['save_checkpoints'] and best_r2 < r2:
                best_r2 = max(best_r2, r2)
                if len(val_drug_ids) == 1:
                    model_path = f"{configs['work_dir']}/{stamp}_{get_model_filename(val_drug_ids[0])}.pth"
                else:
                    model_path = f"{configs['work_dir']}/{stamp}{configs['suffix']}.pth"
                torch.save(model.state_dict(), model_path)
        return None

    elif configs['task'] == 'classification':
        best_auc = 0.71
        criterion = nn.BCEWithLogitsLoss()
        for epoch in trange(1, epochs + 1):
            if lr_scheduler:
                logger.info(f"learning rate: {lr_scheduler.get_lr()}")
            train_score = train_cls(train_loader,
                                    model,
                                    criterion,
                                    optimizer,
                                    epoch,
                                    logger)

            train_res.append(train_score)
            if lr_scheduler:
                lr_scheduler.step()

            accuracy, auc = validate_cls(val_loader, model, val_drug_ids, run=run, epoch=epoch,
                                         val_score_dict=val_score_dict)
            if accuracy:
                logger.info(f"Epoch {epoch} validation accuracy:{accuracy:4f}, AUC:{auc:4f}")
            else:
                logger.info(f"Epoch {epoch} validation Inf")
            if configs['save_checkpoints'] and auc > best_auc:
                best_auc = max(best_auc, auc)
                if len(val_drug_ids) == 1:
                    model_path = f"{configs['work_dir']}/{stamp}_{get_model_filename(val_drug_ids[0])}.pth"
                else:
                    model_path = f"{configs['work_dir']}/{stamp}{configs['suffix']}.pth"
                torch.save(model.state_dict(), model_path)

            torch.cuda.empty_cache()
        return None

    elif configs['task'] == 'multiclass':
        best_avg_acc = 0.7
        criterion = nn.CrossEntropyLoss()
        all_val_res = []
        for epoch in trange(1, epochs + 1):
            if lr_scheduler:
                logger.info(f"learning rate: {lr_scheduler.get_lr()}")
            train_score = train_cls_multiclass(train_loader,
                                               model,
                                               criterion,
                                               optimizer,
                                               epoch,
                                               logger)

            train_res.append(train_score)
            if lr_scheduler:
                lr_scheduler.step()

            top1_acc, top3_acc, f1, roc_auc, val_res_perclass = validate_cls_multiclass(val_loader, model, run=run,
                                                                                        epoch=epoch,
                                                                                        val_score_dict=val_score_dict)
            all_val_res.append(val_res_perclass)
            logger.info(
                f"Epoch {epoch} validation top1_acc:{top1_acc:4f}, f1:{f1:4f}, AUC:{roc_auc:4f}")
            avg_acc = np.mean([top1_acc, top3_acc])
            if configs['save_checkpoints'] and avg_acc > best_avg_acc:
                best_avg_acc = max(best_avg_acc, avg_acc)
                if len(val_drug_ids) == 1:
                    model_path = f"{configs['work_dir']}/{stamp}_{get_model_filename(val_drug_ids[0])}.pth"
                else:
                    model_path = f"{configs['work_dir']}/{stamp}{configs['suffix']}.pth"
                torch.save(model.state_dict(), model_path)

            torch.cuda.empty_cache()
        return pd.concat(all_val_res)
    else:
        raise Exception


def train_cls(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_accuracy = AverageMeter()
    avg_auc = AverageMeter()

    model.train()

    end = time.time()
    lr_str = ''

    for i, data in enumerate(train_loader):

        if len(data) == 2:
            (input_, targets) = data
            output = model(input_.float().to(device))
        elif len(data) == 3:
            (input_, tissue_x, targets) = data
            output = model(input_.float().to(device), tissue_x.float().to(device))
        else:
            raise Exception

        output[targets == MISSING_NUM] = MISSING_NUM

        loss = criterion(output, targets.float().to(device))
        targets = targets.cpu().numpy()

        confs = torch.sigmoid(output).detach().cpu().numpy()
        predicts = (confs > 0.5).astype(int)

        avg_auc.update(np.median(
            [roc_auc_score(targets[targets[:, i] != MISSING_NUM, i],
                           confs[targets[:, i] != MISSING_NUM, i])
             for i in range(confs.shape[1])]))

        avg_accuracy.update(np.median(
            [accuracy_score(targets[targets[:, i] != MISSING_NUM, i],
                            predicts[targets[:, i] != MISSING_NUM, i])
             for i in range(predicts.shape[1])]))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(f'{epoch} \t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'avg_accuracy {avg_accuracy.val:.4f} ({avg_accuracy.avg:.4f})\t'
                f'avg_auc {avg_auc.val:.4f} ({avg_auc.avg:.4f})\t' + lr_str)

    return avg_accuracy.avg


def validate_cls(val_loader, model, val_drug_ids, run=None, epoch=None, val_score_dict=None):
    confs, targets = inference(val_loader, model)
    confs = torch.from_numpy(confs)
    confs = torch.sigmoid(confs).numpy()

    predicts = (confs > 0.5).astype(int)
    acc_avg, auc_avg = None, None
    if not np.isinf(confs).any() and not np.isnan(confs).any():
        if 'drug_id' in val_score_dict:
            val_score_dict['drug_id'].extend(val_drug_ids)
        elif 'Gene' in val_score_dict:
            val_score_dict['Gene'].extend(val_drug_ids)
        else:
            raise Exception

        val_score_dict['run'].extend([run] * len(val_drug_ids))
        val_score_dict['epoch'].extend([epoch] * len(val_drug_ids))
        accuracy = [accuracy_score(targets[targets[:, i] != MISSING_NUM, i], predicts[targets[:, i] != MISSING_NUM, i])
                    for i in range(predicts.shape[1])]
        acc_avg = np.median(accuracy)

        auc = [roc_auc_score(targets[targets[:, i] != MISSING_NUM, i], confs[targets[:, i] != MISSING_NUM, i])
               for i in range(confs.shape[1])]
        auc_avg = np.median(auc)

        val_score_dict['accuracy'].extend(accuracy)
        val_score_dict['auc'].extend(auc)

    return acc_avg, auc_avg


def train_cls_multiclass(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_top3_acc = AverageMeter()
    avg_f1 = AverageMeter()
    avg_auc = AverageMeter()

    model.train()

    end = time.time()
    lr_str = ''

    for i, data in enumerate(train_loader):
        (input_, targets) = data
        output = model(input_.float().to(device))

        loss = criterion(output, targets.flatten().long().to(device))
        output = torch.softmax(output, dim=-1)
        targets = targets.cpu().numpy()

        confs = output.detach().cpu().numpy()
        predicts = np.argsort(-confs, axis=1)
        targets = targets.flatten()
        top1_acc = np.sum((predicts[:, 0] == targets)) / targets.shape[0]
        top3_acc = np.sum(np.any(predicts[:, :3] == np.expand_dims(targets, axis=1), axis=1)) / targets.shape[0]
        f1 = f1_score(targets, predicts[:, 0], average='macro')
        # roc_auc = roc_auc_score(targets, confs, multi_class='ovo')

        avg_top1_acc.update(top1_acc)
        avg_top3_acc.update(top3_acc)
        avg_f1.update(f1)
        # avg_auc.update(roc_auc)

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(f'{epoch} \t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'avg_top1_acc {avg_top1_acc.val:.4f} ({avg_top1_acc.avg:.4f})\t'
                f'avg_f1 {avg_f1.val:.4f} ({avg_f1.avg:.4f})' + lr_str)

    return avg_top1_acc.avg


def validate_cls_multiclass(val_loader, model, run=None, epoch=None, val_score_dict=None):
    confs, targets = inference(val_loader, model)
    predicts = np.argsort(-confs, axis=1)
    confs = softmax(confs, axis=-1)
    val_res_perclass = {}

    val_score_dict['run'].append(run)
    val_score_dict['epoch'].append(epoch)
    targets = targets.flatten()
    top1_acc = np.sum((predicts[:, 0] == targets)) / targets.shape[0]

    if np.unique(targets).size > 2:
        top3_acc = np.sum(np.any(predicts[:, :3] == np.expand_dims(targets, axis=1), axis=1)) / targets.shape[0]
        f1 = f1_score(targets, predicts[:, 0], average='macro')
        roc_auc = roc_auc_score(targets, confs, multi_class='ovo')
    else:
        top3_acc = 1
        f1 = f1_score(targets, predicts[:, 0])
        roc_auc = roc_auc_score(targets, confs[:, 1])

    val_res_perclass['run'] = [run] * len(predicts)
    val_res_perclass['epoch'] = [epoch] * len(predicts)
    val_res_perclass['y_pred'] = predicts[:, 0]
    val_res_perclass['y_true'] = targets
    for i in range(confs.shape[1]):
        val_res_perclass[f"feature_{i}"] = confs[:, i]

    val_score_dict['top1_acc'].append(top1_acc)
    val_score_dict['top3_acc'].append(top3_acc)
    val_score_dict['f1'].append(f1)
    val_score_dict['roc_auc'].append(roc_auc)

    return top1_acc, top3_acc, f1, roc_auc, pd.DataFrame(val_res_perclass)
