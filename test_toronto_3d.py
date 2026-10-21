import warnings
import argparse, os, time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from helper_tool import ConfigToronto_3D as cfg, DataProcessing as DP
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from toronto3d_dataset import Toronto_3D, Toronto_3DSampler
from helper_ply import write_ply

# 禁用 cudnn 加速以避免大矩阵反向传播报错
torch.backends.cudnn.enabled = False

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default= 'output/checkpoint.tar',
                    help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='test_output',
                    help='Dump dir to save logs and preds [default: test_output]')
parser.add_argument('--max_vote', type=int, default=100,
                    help='Number of vote rounds [default: 100]')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu to use [default: 0], -1 for cpu')
parser.add_argument('--L002', type=int, default=2,
                    help='Which area to use for test, option: 1-6 [default: 2]')
FLAGS = parser.parse_args()

# 日志目录
LOG_DIR = os.path.join(FLAGS.log_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
os.makedirs(os.path.join(LOG_DIR, 'val_preds'), exist_ok=True)
area_str = f"L{FLAGS.L002:03d}"
LOG_FOUT = open(os.path.join(LOG_DIR, f"log_test_{area_str}.txt"), 'a')

def log_string(msg):
    LOG_FOUT.write(msg + '\n'); LOG_FOUT.flush(); print(msg)

# 加载数据集与验证加载器
dataset = Toronto_3D(FLAGS.L002)
cfg.num_classes = dataset.num_classes
val_sampler = Toronto_3DSampler(dataset, split='validation')
val_loader = DataLoader(val_sampler, batch_size=cfg.val_batch_size,
                        shuffle=True, collate_fn=val_sampler.collate_fn,
                        num_workers=4, pin_memory=True, drop_last=False)

# 设备设置
if FLAGS.gpu >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{FLAGS.gpu}')
else:
    if FLAGS.gpu >= 0: warnings.warn('CUDA unavailable, using CPU')
    device = torch.device('cpu')

# 网络与优化器
# 网络与优化器（与训练脚本完全相同的分组策略）
net = Network(cfg).to(device)

# 构造两组参数：bias/BN → no decay，其它 → weight_decay
decay_params = []
no_decay_params = []
for name, param in net.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith('.bias') or 'bn' in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW(
    [
        {'params': decay_params,    'weight_decay': cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.},
    ],
    lr=cfg.learning_rate
)

# 加载 checkpoint
assert FLAGS.checkpoint_path and os.path.isfile(FLAGS.checkpoint_path), \
       f"Checkpoint not found: {FLAGS.checkpoint_path}"
ckpt = torch.load(FLAGS.checkpoint_path, map_location=device)
# 恢复模型状态
try:
    net.load_state_dict(ckpt['model_state_dict'])
except RuntimeError:
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    net.load_state_dict(state)
# 恢复 optimizer 状态（可选）
if 'optimizer_state_dict' in ckpt:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    log_string(f"Loaded model and optimizer from {FLAGS.checkpoint_path}")
else:
    log_string(f"Loaded model (no optimizer state) from {FLAGS.checkpoint_path}")



class ModelTester:
    def __init__(self, dataset):
        # 初始化子云投票概率矩阵
        self.test_probs = [np.zeros((labels.shape[0], dataset.num_classes), dtype=np.float32)
                           for labels in dataset.input_labels['validation']]
        # 缓存标签映射为 numpy，避免 list 索引问题
        self.label_values = np.array(dataset.label_values, dtype=np.int32)
        self.dataset = dataset

    def test(self, num_vote=100):
        test_smooth = 0.95
        for round_id in range(num_vote):
            net.eval()
            for batch_data in val_loader:
                # 移动到设备
                for k, v in batch_data.items():
                    if isinstance(v, list):
                        batch_data[k] = [x.to(device) for x in v]
                    else:
                        batch_data[k] = v.to(device)
                with torch.no_grad():
                    end_points = net(batch_data)
                    _, end_points = compute_loss(end_points, cfg, device)

                probs = end_points['valid_logits']
                idxs  = end_points['input_inds'].cpu().numpy()
                c_inds= end_points['cloud_inds'].cpu().numpy()
                B = c_inds.shape[0]
                N = idxs.shape[1]
                probs = F.softmax(probs.reshape(B, N, dataset.num_classes), dim=2).cpu().numpy()
                for b in range(B):
                    cloud_id = int(c_inds[b][0])
                    p_idx = idxs[b]
                    self.test_probs[cloud_id][p_idx] = (
                        test_smooth * self.test_probs[cloud_id][p_idx] +
                        (1 - test_smooth) * probs[b]
                    )

            # 子云评估
            log_string(f"--- Voting round {round_id+1}/{num_vote} ---")
            confusion_list = []
            for i_c, labels in enumerate(self.dataset.input_labels['validation']):
                preds = self.label_values[np.argmax(self.test_probs[i_c], axis=1)]
                    # 先把 confusion matrix 存到 conf 里
                conf = confusion_matrix(labels, preds, labels=self.dataset.label_values)
                confusion_list.append(conf)
                   # 再用 conf 来算每类 IoU
                ious = DP.IoU_from_confusions(conf)
                mean_iou = np.mean(ious) * 100
                log_string(f"Sub-cloud {i_c}: mIoU = {mean_iou:.2f}%")
                # 输出每个类别的 IoU
                for cls_id, cls_iou in zip(self.dataset.label_values, ious):
                    log_string(f"    Class {cls_id} IoU = {cls_iou*100:.2f}%")

        # 全云投影并保存
        os.makedirs(os.path.join(LOG_DIR, 'val_preds'), exist_ok=True)
        full_conf = []
        for i_c in range(len(self.dataset.val_proj)):
            proj_idx = self.dataset.val_proj[i_c]
            probs_full = self.test_probs[i_c][proj_idx]
            preds_full = self.label_values[np.argmax(probs_full, axis=1)]
            labels_full = self.dataset.val_labels[i_c]
            write_ply(
                os.path.join(LOG_DIR, 'val_preds',
                             f"{self.dataset.input_names['validation'][i_c]}.ply"),
                [preds_full, labels_full], ['pred', 'label']
            )
            full_conf.append(
                confusion_matrix(
                    labels_full,
                    preds_full,
                    labels=self.dataset.label_values
                )
            )
        C = np.sum(np.stack(full_conf), axis=0)
        IoUs = DP.IoU_from_confusions(C)
        mean_full = np.mean(IoUs) * 100
        log_string(f"Full-cloud mIoU = {mean_full:.2f}%")
        # 输出每个类别的全云 IoU
        for cls_id, cls_iou in zip(self.dataset.label_values, IoUs):
            log_string(f"    Class {cls_id} IoU = {cls_iou*100:.2f}%")
        # 最后标记测试完成
        log_string('Testing completed.')

if __name__ == '__main__':
    tester = ModelTester(dataset)
    tester.test(num_vote=FLAGS.max_vote)


