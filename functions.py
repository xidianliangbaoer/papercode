import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn.manifold import TSNE
# plt.rcParams["font.sans-serif"]=["SimHei"]
# plt.rcParams["axes.unicode_minus"]=False
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import models

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """

    feature_extractor.eval()
    all_features = []
    all_label = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0]
            label = data[1]
            all_label.append(label)
            data = inputs.reshape([inputs.shape[0], 1, inputs.shape[1]])
            data = data.type(torch.cuda.FloatTensor)
            inputs = data.to(device)
            feature = feature_extractor(inputs).cpu()
            all_features.append(feature)
    return torch.cat(all_features, dim=0),torch.cat(all_label, dim=0)

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str,src_label: torch.Tensor,tar_label: torch.Tensor):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    tsne = TSNE(n_components=2, random_state=33)
    src_sne = tsne.fit_transform(source_feature)
    tar_sne = tsne.fit_transform(target_feature)

    tar_feature_xdf = tar_sne[:, 0]
    tar_feature_ydf = tar_sne[:, 1]
    temp = np.where(tar_label == 0)[0]
    tar_l0_x = tar_feature_xdf[np.where(tar_label == 0)[0]]
    tar_l0_y = tar_feature_ydf[np.where(tar_label == 0)[0]]
    tar_l1_x = tar_feature_xdf[np.where(tar_label == 1)[0]]
    tar_l1_y = tar_feature_ydf[np.where(tar_label == 1)[0]]
    tar_l2_x = tar_feature_xdf[np.where(tar_label == 2)[0]]
    tar_l2_y = tar_feature_ydf[np.where(tar_label == 2)[0]]
    plt.scatter(tar_l0_x, tar_l0_y)
    plt.scatter(tar_l1_x, tar_l1_y)
    plt.scatter(tar_l2_x, tar_l2_y)
    plt.legend(['0', '1', '2'])
    # plt.title(title)
    # plt.savefig(title+'png')
    # plt.show()
    fname = filename + '_目标域各类分布'
    plt.title(fname)
    plt.savefig(fname + '.png')
    # plt.show()
    plt.close('all')


    src_feature_xdf = src_sne[:, 0]
    src_feature_ydf = src_sne[:, 1]
    src_l0_x = src_feature_xdf[np.where(src_label == 0)[0]]
    src_l0_y = src_feature_ydf[np.where(src_label == 0)[0]]
    src_l1_x = src_feature_xdf[np.where(src_label == 1)[0]]
    src_l1_y = src_feature_ydf[np.where(src_label == 1)[0]]
    src_l2_x = src_feature_xdf[np.where(src_label == 2)[0]]
    src_l2_y = src_feature_ydf[np.where(src_label == 2)[0]]
    plt.scatter(src_l0_x,src_l0_y)
    plt.scatter(src_l1_x,src_l1_y)
    plt.scatter(src_l2_x,src_l2_y)
    plt.legend(['0', '1', '2'])
    fname = filename + '_源域各类分布'
    plt.title(fname)
    plt.savefig(fname+'.png')
    # plt.show()
    plt.close('all')

    plt.scatter(src_feature_xdf, src_feature_ydf)
    plt.scatter(tar_feature_xdf, tar_feature_ydf)
    plt.legend(['src', 'tar'])
    fname = filename + '_源域目标域分布'
    plt.title(fname)
    # plt.show()
    plt.savefig(fname+'.png')
    plt.close('all')

def init_model(modelname,store_path,device):
    model = getattr(models, modelname)()
    if store_path is not None:
        model.load(store_path)
    return model.to(device)
