import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
from wmamba import WRBlock

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        name = config.name

        if name in ('Toronto_3D', 'Toronto3D'):

            self.class_weights = DP.get_class_weights('Toronto_3D')
            self.fc0 = nn.Linear(12, 8)
        elif name == 'SemanticKITTI':

            self.class_weights = DP.get_class_weights('SemanticKITTI')
            self.fc0 = nn.Linear(3, 8)
        elif name == 'Sensaturban':

            self.class_weights = DP.get_class_weights('Sensaturban')
            self.fc0 = nn.Linear(12, 8)
        elif name == 'S3DIS':

            self.class_weights = DP.get_class_weights('S3DIS')
            self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)
        else:
            raise ValueError(f"Unsupported dataset: {name}")

        self.fc0_acti = nn.LeakyReLU()
        self.fc0_bath = nn.BatchNorm1d(self.fc0.out_features, eps=1e-6, momentum=0.99)
        nn.init.constant_(self.fc0_bath.weight, 1.0)
        nn.init.constant_(self.fc0_bath.bias,   0.0)

        C0 = self.fc0.out_features

        self.mrffi_blocks = nn.ModuleList()
        self.mrffi_fuse   = nn.ModuleList()
        for i in range(config.num_layers):
            C = 2 * config.d_out[i]
            self.mrffi_blocks.append(WRBlock(channels=C, prior_channels=6))
            self.mrffi_fuse.append(
                pt_utils.Conv2d(C + 6, C, kernel_size=(1,1), bn=True)
            )

        self.dilated_res_blocks = nn.ModuleList()
        d_in = C0
        for i in range(config.num_layers):
            d_out = config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out, idx=i))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(config.num_layers):
            if j < config.num_layers - 1:
                d_in = d_out + 2 * config.d_out[-j-2]
                d_out = 2 * config.d_out[-j-2]
            else:

                d_in = 4 * config.d_out[0]
                d_out = 2 * config.d_out[0]
            self.decoder_blocks.append(
                pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)
            )

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64,   32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1),
                                   bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']
        wavelet_prior = features[:, :, 6:12] if self.config.name in ('Toronto_3D', 'Toronto3D') else None
        features = self.fc0(features)
        features = self.fc0_acti(features)
        features = features.transpose(1, 2)
        features = self.fc0_bath(features)
        features = features.unsqueeze(3)

        f_encoder_list = []
        for i in range(self.config.num_layers):

            f_enc = self.dilated_res_blocks[i](
                features,
                end_points['xyz'][i],
                end_points['neigh_idx'][i],
                None if wavelet_prior is None else wavelet_prior[:, :features.size(2), :]
            )

            mr_in = f_enc.squeeze(-1)

            mr_out = self.mrffi_blocks[i](
                mr_in,
                end_points['xyz'][i],
                end_points['order_h'][i],
                end_points['order_t'][i],
                None if wavelet_prior is None else wavelet_prior[:, :f_enc.size(2), :]
            )

            mr_out = mr_out.permute(0,2,1).unsqueeze(-1)

            mr_fused = self.mrffi_fuse[i](mr_out)
            N_i = f_enc.size(2)

            hf = mr_fused[:, :, :N_i, :]
            tf = mr_fused[:, :, N_i:, :]
            order_h_inv = torch.argsort(end_points['order_h'][i], dim=1)
            order_t_inv = torch.argsort(end_points['order_t'][i], dim=1)
            channels = mr_fused.size(1)
            hf = torch.gather(hf, 2, order_h_inv[:, None, :, None].expand(-1, channels, -1, 1))
            tf = torch.gather(tf, 2, order_t_inv[:, None, :, None].expand(-1, channels, -1, 1))
            mr_fused = 0.5 * (hf + tf)
            f_enc = f_enc + mr_fused

            f_sampled = self.random_sample(f_enc, end_points['sub_idx'][i])

            features = f_sampled

            if i == 0:
                f_encoder_list.append(f_enc.squeeze(-1))

            f_encoder_list.append(f_sampled.squeeze(-1))

        bottleneck = f_encoder_list[-1]
        features = self.decoder_0(bottleneck.unsqueeze(-1))

        for j in range(self.config.num_layers):

            interp = self.nearest_interpolation(
                features,
                end_points['interp_idx'][-j - 1]
            )

            skip_feat = f_encoder_list[-j - 2].unsqueeze(-1)

            concat = torch.cat([skip_feat, interp], dim=1)
            features = self.decoder_blocks[j](concat)

        x = self.fc1(features)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        logits = x.squeeze(3)

        end_points['logits'] = logits
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))

        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))

        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features

def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points

class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(
        labels_valid,
        pred_valid,
        labels=np.arange(self.cfg.num_classes)
        )
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list

class MLPMultiScaleWaveletBlock(nn.Module):

    def __init__(self, channels, levels=2, wavelet='haar', mode='zero'):
        super().__init__()

        from pytorch_wavelets import DWT1DForward
        self.dwt = DWT1DForward(J=levels, wave=wavelet, mode=mode)

        self.detail_convs = nn.ModuleList([
            pt_utils.Conv2d(
                in_size=channels,
                out_size=channels,
                kernel_size=(1,1),
                bn=True
            )
            for _ in range(levels)
        ])

        self.low_conv = pt_utils.Conv2d(
            in_size=channels,
            out_size=channels,
            kernel_size=(1,1),
            bn=True
        )

        self.fuse = pt_utils.Conv2d(
            in_size=channels,
            out_size=channels,
            kernel_size=(1,1),
            bn=True
        )

    def forward(self, x, knn_idx=None):
        """
        WaveletMLPBlock 的前向，不再强制要求 knn_idx
        Args:
            x:       Tensor of shape [B, C, N, 1]
            knn_idx: （可选）Tensor of shape [B, N, k]，目前未用到
        Returns:
            out:     Tensor of shape [B, C, N, 1]
        """

        B, C, N, _ = x.shape

        x1 = x.squeeze(-1)

        Yl, Yh = self.dwt(x1)

        low = Yl.unsqueeze(-1)

        low_up = F.interpolate(
            low,
            size=(N, 1),
            mode='nearest'
        )
        low_feat = self.low_conv(low_up)

        details = torch.zeros_like(low_feat)
        for j, detail in enumerate(Yh):

            det = detail.unsqueeze(-1)
            det_up = F.interpolate(
                det,
                size=(N, 1),
                mode='nearest'
            )
            details = details + self.detail_convs[j](det_up)

        combined = low_feat + details
        out = self.fuse(combined)

        return out

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out, idx):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1,1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx, wavelet_prior=None):

        f_pc   = self.mlp1(feature)

        f_pc   = self.lfa(xyz, f_pc, neigh_idx, wavelet_prior)
        f_pc   = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)

class Building_block(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.wavelet_mlp = pt_utils.Conv2d(6, d_out//2, kernel_size=(1,1), bn=True)
        self.neighbour_mlp = pt_utils.Conv2d(d_out, d_out//2, kernel_size=(1,1), bn=True)

        self.att_pooling_1 = Att_pooling(d_out + d_out//2, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx, wavelet_prior=None):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        if wavelet_prior is not None:
            wavelet_neighbours = self.gather_neighbour(wavelet_prior, neigh_idx)
            wavelet_neighbours = wavelet_neighbours.permute((0, 3, 1, 2))
            f_wavelet = self.wavelet_mlp(wavelet_neighbours)
            f_neighbours = self.neighbour_mlp(f_neighbours)
            f_concat = torch.cat([f_wavelet, f_xyz, f_neighbours], dim=1)
        else:
            f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        return features

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg

def compute_loss(end_points, cfg, device):

    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)

    unique, counts = torch.unique(labels, return_counts=True)
    freqs = counts.float() / labels.shape[0]
    eps   = 0.02

    batch_weights = torch.ones(cfg.num_classes, device=device)
    batch_weights[unique] = 1.0 / (freqs + eps)

    batch_weights = batch_weights.clamp(min=1.0, max=10.0)

    loss = get_loss(
        logits,
        labels,
        batch_weights.cpu().numpy(),
        device
    )

    end_points['valid_logits'] = logits
    end_points['valid_labels'] = labels
    end_points['loss']         = loss

    return loss, end_points

def get_loss(logits, labels, pre_cal_weights, device):

    class_weights = torch.from_numpy(pre_cal_weights).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.reshape([-1]), reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
