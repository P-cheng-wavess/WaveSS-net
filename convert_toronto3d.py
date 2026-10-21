import os
import glob
import pickle
import numpy as np
import pandas as pd
import pywt
from sklearn.neighbors import KDTree
from os.path import join, dirname, abspath
from hilbertcurve.hilbertcurve import HilbertCurve
from helper_ply import read_ply, write_ply
from helper_tool import DataProcessing as DP

# ---------------- 配置 ----------------
BASE_DIR     = dirname(abspath(__file__))
# 请修改为你的 Toronto_3D 原始点云目录
DATASET_ROOT = 'yours'  



# — 下采样 & KNN 参数 — #
SUB_GRID  = 0.06   # 网格下采样大小
LEAF_SZ   = 40     # KDTree leaf size
K_NEIGH   = 16     # KNN 邻居数
SIGMA     = 0.2    # 加权高斯 sigma
# 输出目录
ORIG_PLY_DIR = join(DATASET_ROOT, 'original_ply')
SUB_DIR      = join(DATASET_ROOT, f'input_{SUB_GRID:.3f}')
KNN_DIR      = join(SUB_DIR, 'knn')
WAVE_DIR     = join(SUB_DIR, 'wavelet_features')

for d in [ORIG_PLY_DIR, SUB_DIR, KNN_DIR, WAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- 类别映射 ----
# 用你的类别名称列表替换下面这一行
gt_names = ['Unclassified', 'Ground', 'Road_markings', 'Natural', 'Building', 'Utility_line', 'Pole', 'Car','Fence']
gt2label  = {c:i for i,c in enumerate(gt_names)}

# 下采样 & KNN 参数
SUB_GRID  = 0.06   # 网格下采样大小
LEAF_SZ   = 40     # KDTree leaf size
K_NEIGH   = 16     # KNN 邻居数
SIGMA     = 0.2    # 加权高斯 sigma

def build_weighted_knn(xyz, k=K_NEIGH, sigma=SIGMA, leaf_size=LEAF_SZ):
    tree = KDTree(xyz, leaf_size=leaf_size)
    dists, idxs = tree.query(xyz, k=k+1)
    idxs  = idxs[:,1:]
    dists = dists[:,1:]
    w = np.exp(- (dists**2) / (2 * sigma**2))
    w /= (w.sum(axis=1, keepdims=True) + 1e-12)
    return idxs.astype(np.int32), w.astype(np.float32)

def convert_pc2ply(scan_path):
    """转换单个扫描文件（假设为 .txt 或 .csv 格式）为 .ply，并生成下采样、KDTree、KNN 及小波特征。"""
    # —— 1. 读取原始点云 + 标签 —— #
    ext = os.path.splitext(scan_path)[1].lower()
    if ext == '.ply':
        ply = read_ply(scan_path)
        print(f">>> PLY fields: {ply.dtype.names}")
        xyz    = np.vstack([ply['x'], ply['y'], ply['z']]).T.astype(np.float32)
        colors = np.vstack([ply['red'], ply['green'], ply['blue']]).T.astype(np.uint8)

        # —— 明确读取标签字段 —— #
        if 'scalar_Label' not in ply.dtype.names:
            raise KeyError(f"在 {scan_path} 中未找到 'scalar_Label'，可用字段：{ply.dtype.names}")
        labels = ply['scalar_Label'].astype(np.uint8)
    else:
        data = pd.read_csv(scan_path, header=None, delim_whitespace=True).values
        xyz    = data[:, :3].astype(np.float32)
        colors = data[:, 3:6].astype(np.uint8)
        labels = data[:, 6].astype(np.uint8)

    # 去掉全局最小值（可选）
    xyz -= xyz.min(axis=0, keepdims=True)

    # 输出原始 ply
    room = os.path.splitext(os.path.basename(scan_path))[0]
    out_ori = join(ORIG_PLY_DIR, f'{room}.ply')
    write_ply(out_ori,
              (xyz, colors, labels),
              ['x','y','z','red','green','blue','class'])
    print(f'> [{room}] 原始 ply 保存: {xyz.shape[0]} pts')

    # —— 2. 网格下采样 —— #
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, SUB_GRID)
    # 如果下采样后颜色仍在 0-255 范围内，则除以 255 → [0,1]
    sub_colors = sub_colors.astype(np.float32) / 255.0
    out_sub = join(SUB_DIR, f'{room}.ply')
    write_ply(out_sub,
              (sub_xyz, sub_colors, sub_labels),
              ['x','y','z','red','green','blue','class'])
    print(f'> [{room}] 下采样 ply 保存: {sub_xyz.shape[0]} pts')

    # —— 3. 构建 KDTree & 投影索引 —— #
    tree = KDTree(sub_xyz, leaf_size=LEAF_SZ)
    with open(join(SUB_DIR, f'{room}_KDTree.pkl'), 'wb') as f:
        pickle.dump(tree, f)
    # 每个原始点投影到最近的下采样点
    _, proj_idx = tree.query(xyz, k=1)
    proj_idx = proj_idx.squeeze().astype(np.int32)
    with open(join(SUB_DIR, f'{room}_proj.pkl'), 'wb') as f:
        pickle.dump([proj_idx, labels], f)
    print(f'> [{room}] KDTree & proj 保存')

    # —— 4. 加权 KNN —— #
    knn_idx, knn_w = build_weighted_knn(sub_xyz)
    with open(join(KNN_DIR, f'{room}_knn.pkl'), 'wb') as f:
        pickle.dump({'idx':knn_idx, 'w':knn_w}, f)
    print(f'> [{room}] 加权 KNN 保存')

    # —— 5. 离线小波 + Hilbert 顺序 & 多尺度小波 —— #
    
    feats_all = np.concatenate([sub_xyz, sub_colors * 255.0], axis=1)
    p = 10
    hilbert = HilbertCurve(p, 3)
    coords = feats_all[:, :3]  # 取 x,y,z
    mins, maxs = coords.min(0), coords.max(0)
    norm = ((coords - mins) / (maxs - mins + 1e-6) * (2**p - 1)).astype(int)
    distances = np.array(hilbert.distances_from_points(norm.tolist()))
    order_h = np.argsort(distances).astype(np.int32)
    order_t = order_h[::-1].copy().astype(np.int32)
    with open(join(SUB_DIR, f'{room}_order.pkl'), 'wb') as f:
        pickle.dump({'order_h': order_h, 'order_t': order_t}, f)
    print(f'> [{room}] Hilbert 顺序保存: N={len(order_h)}')

    
    seq = feats_all[order_h]                       # 排序后的序列 [N,6]
    coeffs = pywt.wavedec(seq.T, 'haar', level=2)  # [cA2, cD2, cD1]
    
    low_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    rec = pywt.waverec(low_coeffs, 'haar')         # 平滑重构 [6, L]
   
    seq_len = seq.shape[0]
    if   rec.shape[1] > seq_len:
        rec = rec[:, :seq_len]
    elif rec.shape[1] < seq_len:
        pad = np.zeros((rec.shape[0], seq_len - rec.shape[1]), dtype=rec.dtype)
        rec = np.concatenate([rec, pad], axis=1)
    
    residual = seq - rec.T                         # [N,6] residual
    mu = residual.mean(0, keepdims=True)
    sigma = residual.std(0, keepdims=True) + 1e-6
    residual = (residual - mu) / sigma           # z-score
    wave_hilbert = np.zeros_like(residual)
    wave_hilbert[order_h] = residual               # 反排序回原始次序
    with open(join(WAVE_DIR, f'{room}_wavelet_hilbert.pkl'), 'wb') as f:
        pickle.dump(wave_hilbert, f)
    print(f'> [{room}] Hilbert 小波特征保存: {wave_hilbert.shape[0]}×6')

if __name__ == '__main__':

    scan_files = glob.glob(join(DATASET_ROOT, '*.ply'))
    print(f'发现 {len(scan_files)} 个扫描文件，开始处理 ...')
    for p in scan_files:
        convert_pc2ply(p)

