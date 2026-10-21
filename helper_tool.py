from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from sklearn.neighbors import KDTree


class ConfigSemanticKITTI:
    k_n = 16
    num_layers = 4
    num_points = 4096 * 11
    num_classes = 19
    sub_grid_size = 0.06

    batch_size = 6
    val_batch_size = 20
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4]
    d_out = [16, 64, 128, 256]
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5
    max_epoch = 100
    learning_rate = 3e-3
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

class ConfigSensaturban:
    k_n = 16
    num_layers = 5
    num_points = 65536
    num_classes = 13
    sub_grid_size = 0.2

    batch_size = 4
    val_batch_size = 14
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]

    noise_init = 3.5
    max_epoch = 100
    learning_rate = 1e-3
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log_SensatUrban'
    saving = True
    saving_path = None


class ConfigS3DIS:
    k_n = 16
    num_layers = 5
    num_points = 60960
    num_classes = 13
    sub_grid_size = 0.04

    batch_size = 4
    val_batch_size = 20
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]


    noise_init = 3.5
    max_epoch = 64
    learning_rate = 1e-3
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigToronto_3D:

    dataset_root    = '/home/cheng/WMLP_NET/data/Toronto_3D'

    k_n = 16
    num_layers = 5
    num_points = 53960
    num_classes = 8
    sub_grid_size = 0.06

    batch_size = 4
    val_batch_size = 15
    train_steps = 500
    val_steps = 25

    sub_sampling_ratio = [4,4,4,4,2]
    d_out = [16,64,128,256,512]

    noise_init = 3.5
    max_epoch = 64
    learning_rate = 1e-3
    lr_decays = {i: 0.95 for i in range(0, 500)}
    weight_decay       = 1e-4

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

class ConfigSemantic3D:
    k_n = 16
    num_layers = 5
    num_points = 65536
    num_classes = 8
    sub_grid_size = 0.06

    batch_size = 6
    val_batch_size = 16
    train_steps = 500
    val_steps = 100

    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]

    noise_init = 3.5
    max_epoch = 100
    learning_rate = 1e-4
    lr_decays = {i: 0.95 for i in range(0, 500)}

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8


class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF
        inst_label = label >> 16
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        if test_scan_num != 'None':
            test_file_list = np.concatenate(test_file_list, axis=0)
        else:
            test_file_list = None
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3 在这些点上搜索
        :param query_pts: points you want to know the neighbour index, B*N2*3 以这些点为中心做KNN
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """


        if support_pts.ndim == 3:
            B, N1, _ = support_pts.shape
            N2 = query_pts.shape[1]
            neighbor_idx = np.zeros((B, N2, k), dtype=np.int32)
            for b in range(B):
                tree = KDTree(support_pts[b], leaf_size=40)
                _, idxs = tree.query(query_pts[b], k=k)
                neighbor_idx[b] = idxs
            return neighbor_idx


        else:
            tree = KDTree(support_pts, leaf_size=40)
            _, idxs = tree.query(query_pts, k=k)
            return idxs.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):

        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """


        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)


        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)


        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)


        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):

        num_per_class = []
        if dataset_name == 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name == 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name == 'Toronto_3D':
            num_per_class = np.array([41697357, 1745448, 6572572, 19136493, 674897, 897825, 4634634, 374721],
                                     dtype=np.int32)
        elif dataset_name == 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        elif dataset_name == 'Sensaturban':
            num_per_class = np.array([431, 332, 401, 202, 71, 232, 1.5,
                                      301, 161, 91, 91, 5, 131], dtype=np.int32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)


class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)


        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp


            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins
