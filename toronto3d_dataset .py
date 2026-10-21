from os.path import join, basename
import glob
import os
import pickle
import time
import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset
from helper_tool import DataProcessing as DP, ConfigToronto_3D as cfg
from helper_ply import read_ply


class Toronto_3D(Dataset):
    def __init__(self, val_split=2):

        self.name = 'Toronto_3D'
        cfg.name = 'Toronto_3D'
        self.path = cfg.dataset_root


        self.label_to_names = {
            0: 'Ground', 1: 'Road_markings', 2: 'Natural',
            3: 'Building', 4: 'Utility_line', 5: 'Pole',
            6: 'Car', 7: 'Fence'
        }

        self.label_values = sorted(self.label_to_names.keys())

        self.num_classes = len(self.label_values)

        self.label_to_idx = {lab: i for i, lab in enumerate(self.label_values)}


        cfg.ignored_label_inds = []
        cfg.num_classes        = self.num_classes


        cfg.class_weights = DP.get_class_weights('Toronto_3D').astype(np.float32)

        self.val_split = val_split


        self.all_files = sorted(glob.glob(join(self.path, 'original_ply', '*.ply')))
        self.size = len(self.all_files)


        prefix = f"L{self.val_split:03d}"

        self.val_files = [
            f for f in self.all_files
            if os.path.basename(f).startswith(prefix)
        ]

        self.train_files = [
            f for f in self.all_files
            if not os.path.basename(f).startswith(prefix)
        ]


        self.val_proj    = []
        self.val_labels  = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees  = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names  = {'training': [], 'validation': []}


        self.load_sub_sampled_clouds(cfg.sub_grid_size)


        self.knn_idx_global = {'training': [], 'validation': []}
        knn_dir = join(self.path, f'input_{cfg.sub_grid_size:.3f}', 'knn')
        for split in ['training', 'validation']:
            for name in self.input_names[split]:
                with open(join(knn_dir, f'{name}_knn.pkl'), 'rb') as f:
                    knn_cache = pickle.load(f)
                self.knn_idx_global[split].append(np.asarray(knn_cache['idx'], dtype=np.int32))


        self.hilbert_wavelet_features = {'training': [], 'validation': []}
        hilbert_dir = join(self.path, f'input_{cfg.sub_grid_size:.3f}', 'wavelet_features')
        for split in ['training', 'validation']:
            for name in self.input_names[split]:
                with open(join(hilbert_dir, f'{name}_wavelet_hilbert.pkl'), 'rb') as f:
                    feats_h = pickle.load(f)
                self.hilbert_wavelet_features[split].append(feats_h)


        self.full_hilbert_orders = {'training': [], 'validation': []}
        order_dir = join(self.path, f'input_{cfg.sub_grid_size:.3f}')
        for split in ['training', 'validation']:
            for name in self.input_names[split]:
                pkl_path = join(order_dir, f'{name}_order.pkl')
                with open(pkl_path, 'rb') as f:
                    orders = pickle.load(f)
                self.full_hilbert_orders[split].append(orders)

        print(f"Size of training : {len(self.input_names['training'])}")
        print(f"Size of validation : {len(self.input_names['validation'])}")


    def load_sub_sampled_clouds(self, sub_grid_size):
        """
        从 input_{sub_grid_size} 目录加载下采样点云、KDTree，
        并为验证（或测试）集准备投影索引。
        标签已经在预处理阶段写成了 0..7。
        """
        tree_path = join(self.path, f'input_{sub_grid_size:.3f}')

        for file_path in self.train_files + self.val_files:
            start = time.time()
            name = basename(file_path)[:-4]
            split = 'validation' if file_path in self.val_files else 'training'


            ply_file = join(tree_path, f'{name}.ply')
            data = read_ply(ply_file)
            pts = np.vstack([data['x'], data['y'], data['z']]).T
            colors = np.vstack([data['red'], data['green'], data['blue']]).T
            labels = data['class'].astype(np.int32)


            assert labels.min() >= 0 and labels.max() < self.num_classes, \
                f"标签越界: [{labels.min()}, {labels.max()}]"


            tree = KDTree(pts)


            self.input_trees[split].append(tree)
            self.input_colors[split].append(colors)
            self.input_labels[split].append(labels)
            self.input_names[split].append(name)


            mb = colors.shape[0] * (4*7) * 1e-6
            print(f"{name}_KDTree.pkl {mb:.1f} MB loaded in {time.time()-start:.1f}s")


        print("\nPreparing reprojected indices for validation")
        for file_path in self.val_files:
            start = time.time()
            name = basename(file_path)[:-4]
            proj_file = join(tree_path, f'{name}_proj.pkl')
            with open(proj_file, 'rb') as f:
                proj_idx, proj_labels = pickle.load(f)

            self.val_proj.append(proj_idx)
            self.val_labels.append(proj_labels)

            print(f"{name}_proj.pkl done in {time.time()-start:.1f}s")


    def __getitem__(self, idx):
        pass

    def __len__(self):

        return self.size


class Toronto_3DSampler(Dataset):
    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}
        self.sample_radius = 15.0
        self.num_point = cfg.num_points


        self.class_freq_weights = None
        if split == 'training':
            all_labels = np.hstack(self.dataset.input_labels['training'])
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            freqs = counts / counts.sum()
            self.class_freq_weights = np.zeros(self.dataset.num_classes, dtype=np.float32)
            for lbl, f in zip(unique_labels, freqs):
                self.class_freq_weights[lbl] = f

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]


    def __getitem__(self, item):

        pts_big, lbl_big, idx_big, cloud_idx = \
            self.spatially_regular_gen(item, self.split)

        if self.split == 'training':
            cls_list, num_list = self.get_sample_num(lbl_big.cpu().numpy())
            all_pts, all_lbls, all_idxs = [], [], []
            pts_big_np = pts_big.cpu().numpy()
            lbl_big_np = lbl_big.cpu().numpy()
            idx_big_np = idx_big.cpu().numpy()
            for cls, num in zip(cls_list, num_list):
                mask = (lbl_big_np == cls)
                cls_pts = pts_big_np[mask]
                cls_idxs = idx_big_np[mask]
                pts2, lbl2, idx2 = self.get_points_by_class(cls_pts, cls_idxs, num, cls)
                all_pts.append(pts2)
                all_lbls.append(lbl2)
                all_idxs.append(idx2)
            points_xyzrgb = torch.from_numpy(np.vstack(all_pts)).float()
            labels = torch.from_numpy(np.hstack(all_lbls)).long()
            queried_idx = torch.from_numpy(np.hstack(all_idxs)).long()
        else:
            points_xyzrgb = pts_big
            labels = lbl_big
            queried_idx = idx_big.long()
        room          = int(cloud_idx.item())


        sel        = queried_idx.cpu().numpy().astype(np.int64)
        wave_h_all = self.dataset.hilbert_wavelet_features[self.split][room]
        wave_h_sel = wave_h_all[sel]


        xyzrgb_np = points_xyzrgb.cpu().numpy()
        feat_np   = np.concatenate([xyzrgb_np, wave_h_sel], axis=1)
        feat      = torch.from_numpy(feat_np).float()


        perm        = torch.randperm(feat.size(0))
        feat        = feat[perm]
        labels      = labels[perm]
        queried_idx = queried_idx[perm]


        return feat, labels, queried_idx, cloud_idx

    def __len__(self):

        return self.num_per_epoch


    def get_sample_num(self, labels):
        total_points = labels.shape[0]

        cls_idx_list, cls_num_list = np.unique(labels, return_counts=True)


        sample_ratio_list = [(cls_num / total_points) ** 0.5
                             for cls_num in cls_num_list]
        total_ratio = sum(sample_ratio_list)

        sample_num_list = [
            int(self.num_point * ratio / total_ratio)
            for ratio in sample_ratio_list
        ]

        sample_num_list[-1] = self.num_point - sum(sample_num_list[:-1])
        assert sample_num_list[-1] > 0, \
            f'sample_num must > 0, but got {sample_num_list[-1]}'
        return cls_idx_list, sample_num_list


    def spatially_regular_gen(self, item, split):


        cloud_idx = int(np.argmin(self.min_possibility[split]))


        point_ind = np.argmin(self.possibility[split][cloud_idx])


        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)


        center_point = points[point_ind, :].reshape(1, -1)


        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)


        if len(points) < cfg.num_points:

            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:

            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]


        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]


        if split == 'training':
            queried_pt_weight = self.class_freq_weights[queried_pc_labels]


        dists = np.sum(
            np.square((points[queried_idx] - pick_point).astype(np.float32)),
            axis=1
        )

        dist_weights = 1.0 - dists / max(float(np.max(dists)), 1e-12)
        if split == 'training':
            delta = dist_weights * queried_pt_weight
        else:
            delta = dist_weights


        self.possibility[split][cloud_idx][queried_idx] += delta

        self.min_possibility[split][cloud_idx] = float(
            np.min(self.possibility[split][cloud_idx])
        )


        if len(points) < cfg.num_points:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)


        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(np.asarray(queried_idx, dtype=np.int64)).long()
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int64)).long()

        points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

        return points, queried_pc_labels, queried_idx, cloud_idx


    def get_points_by_class(self, class_pts, class_idxs, sample_num, cls_label):
        """
        类别感知球形区域采样（Ball Query）实现：
          - class_pts: np.ndarray [M,6]
          - class_idxs: np.ndarray [M]
          - sample_num: int
          - cls_label: int
        返回:
          pts:   [sample_num,6]
          lbls:  [sample_num]
          idxs:  [sample_num]
        """
        M = class_pts.shape[0]
        assert M > 0, f'no points for class {cls_label}'


        if M < sample_num:
            sel = np.random.choice(M, sample_num, replace=True)
            return (
                class_pts[sel],
                np.full(sample_num, cls_label, dtype=np.int64),
                class_idxs[sel]
            )


        max_tries = 5

        radius = self.sample_radius

        for _ in range(max_tries):

            center = class_pts[np.random.randint(M), :3]

            dists = np.linalg.norm(class_pts[:, :3] - center[None, :], axis=1)

            idxs = np.nonzero(dists <= radius)[0]
            if idxs.size >= sample_num:
                break


        if idxs.size >= sample_num:
            sel = np.random.choice(idxs, sample_num, replace=False)
        else:
            sel = np.random.choice(idxs, sample_num, replace=True)

        return (
            class_pts[sel],
            np.full(sample_num, cls_label, dtype=np.int64),
            class_idxs[sel]
        )


    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):
        """
        进行下采样和KNN的索引记录，为后面网络做准备
        """
        batch_size, N, _ = batch_xyz.shape

        batch_pc_idx = batch_pc_idx.astype(np.int64)

        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)

        input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []

        for i in range(cfg.num_layers):
            if i == 0:

                neighbour_idx_list = []
                for b in range(batch_size):
                    room_id    = int(np.asarray(batch_cloud_idx[b]).item())
                    global_knn = self.dataset.knn_idx_global[self.split][room_id]
                    sel_idx    = batch_pc_idx[b]


                    ng_global = global_knn[sel_idx]


                    order      = np.argsort(sel_idx)
                    sorted_sel = sel_idx[order]


                    flat_ng = ng_global.ravel()
                    pos     = np.searchsorted(sorted_sel, flat_ng)


                    pos_safe = pos.copy()
                    pos_safe[pos_safe >= N] = 0


                    mask_oob      = pos >= N
                    mask_mismatch = sorted_sel[pos_safe] != flat_ng
                    mask_invalid  = mask_oob | mask_mismatch


                    pos_safe[mask_invalid] = 0


                    local_flat = order[pos_safe]


                    ng_local = local_flat.reshape(N, -1)
                    neighbour_idx_list.append(ng_local)

                neighbour_idx = np.stack(neighbour_idx_list, axis=0)
            else:

                neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)


            num_pts = batch_xyz.shape[1]
            mi, ma  = neighbour_idx.min(), neighbour_idx.max()
            if mi < 0 or ma >= num_pts:
                raise IndexError(
                    f"Layer {i} neighbour_idx 越界：min={mi}, max={ma}, N={num_pts}"
                )


            sub_N      = num_pts // cfg.sub_sampling_ratio[i]
            sub_points = batch_xyz[:, :sub_N, :]
            pool_i     = neighbour_idx[:, :sub_N, :]
            up_i       = DP.knn_search(sub_points, batch_xyz, 1)

            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)


            batch_xyz = sub_points


        inputs = input_points + input_neighbors + input_pools + input_up_samples
        inputs += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]
        return inputs


    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = zip(*batch)
        B = len(batch)

        selected_pc     = np.stack(selected_pc, axis=0)
        selected_labels = np.stack(selected_labels, axis=0)
        selected_idx    = np.stack(selected_idx, axis=0)
        cloud_ind       = np.stack(cloud_ind, axis=0)


        xyz   = selected_pc[:, :, :3]
        feats = selected_pc[:, :, 3:]


        flat = self.tf_map(xyz, feats, selected_labels, selected_idx, cloud_ind)


        L = cfg.num_layers
        inputs = {
            'xyz': [],
            'neigh_idx': [],
            'sub_idx': [],
            'interp_idx': []
        }

        for i in range(L):
            inputs['xyz'].append(    torch.from_numpy(flat[i]).float())
            inputs['neigh_idx'].append(torch.from_numpy(flat[L + i]).long())
            inputs['sub_idx'].append( torch.from_numpy(flat[2 * L + i]).long())
            inputs['interp_idx'].append(torch.from_numpy(flat[3 * L + i]).long())


        inputs['features']    = torch.from_numpy(flat[4 * L    ]).float()
        inputs['labels']      = torch.from_numpy(flat[4 * L + 1]).long()
        inputs['input_inds']  = torch.from_numpy(flat[4 * L + 2]).long()
        inputs['cloud_inds']  = torch.from_numpy(flat[4 * L + 3]).long()


        inputs['knn_idx'] = inputs['neigh_idx'][0]


        B = inputs['cloud_inds'].shape[0]


        inputs['input_inds']  = torch.from_numpy(flat[4 * L + 2]).long()
        inputs['cloud_inds']  = torch.from_numpy(flat[4 * L + 3]).long()


        level_inds = {0: inputs['input_inds']}
        for i in range(1, L):

            N_i = inputs['xyz'][i].shape[1]

            level_inds[i] = level_inds[i-1][:, :N_i]

        for i in range(L):
            inputs[f'input_inds_{i}'] = level_inds[i]


        inputs['order_h'], inputs['order_t'] = [], []
        for lvl in range(L):
            oh_list, ot_list = [], []
            for b in range(B):
                room = inputs['cloud_inds'][b].item()
                orders = self.dataset.full_hilbert_orders[self.split][room]
                full_h, full_t = orders['order_h'], orders['order_t']

                inds = inputs[f'input_inds_{lvl}'][b].cpu().numpy()
                inv_h = np.empty_like(full_h); inv_h[full_h] = np.arange(len(full_h))
                inv_t = np.empty_like(full_t); inv_t[full_t] = np.arange(len(full_t))
                oh_list.append(torch.from_numpy(np.argsort(inv_h[inds])).long())
                ot_list.append(torch.from_numpy(np.argsort(inv_t[inds])).long())
            inputs['order_h'].append(torch.stack(oh_list, 0))
            inputs['order_t'].append(torch.stack(ot_list, 0))

        return inputs


if __name__ == '__main__':
    dataset = Toronto_3D(val_split=2)
    dataset_train = Toronto_3DSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=dataset_train.collate_fn)

    for data in dataloader:

        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(features.shape)
        print(labels.shape)
        print(idx.shape)
        print(cloud_idx.shape)
        break
