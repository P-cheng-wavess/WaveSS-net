import warnings
from helper_tool import ConfigToronto_3D as cfg
from WavessNet import Network, compute_loss, compute_acc, IoUCalculator
from toronto3d_dataset import Toronto_3D, Toronto_3DSampler
import numpy as np
import os, argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time

torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='train_output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=64, help='Epoch to run [default: 64]')
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 0], -1 for cpu')
parser.add_argument('--L002', type=int, default=2, help='Which area to use for test [default: L002]')
FLAGS = parser.parse_args()
dataset = Toronto_3D(FLAGS.L002)

cfg.num_classes        = dataset.num_classes
training_dataset   = Toronto_3DSampler(dataset, 'training')
training_dataloader = DataLoader(
    training_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=training_dataset.collate_fn,
    drop_last=True
)


LOG_DIR = FLAGS.log_dir
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
os.makedirs(LOG_DIR, exist_ok=True)
area_str = f"L{FLAGS.L002:03d}"
log_file_name = f"log_train_{area_str}.log"
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


training_dataset = Toronto_3DSampler(dataset, 'training')
validation_dataset = Toronto_3DSampler(dataset, 'validation')
training_dataloader= DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=training_dataset.collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=validation_dataset.collate_fn)

print(len(training_dataloader), len(validation_dataloader))


if FLAGS.gpu >= 0:
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
else:
    FLAGS.gpu = torch.device('cpu')

device = FLAGS.gpu

net = Network(cfg)
net.to(device)


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

it = -1

start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(training_dataloader):
        t_start = time.time()
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)


        optimizer.zero_grad()
        end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg, device)
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)


        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 50
        if (batch_idx + 1) % batch_interval == 0:
            t_end = time.time()


            log_string('Step %03d Loss %.3f Acc %.2f lr %.5f --- %.2f ms/batch' % (batch_idx + 1, stat_dict['loss'] / batch_interval, stat_dict['acc'] / batch_interval, optimizer.param_groups[0]['lr'], 1000 * (t_end - t_start)))
            stat_dict['loss'], stat_dict['acc'] = 0, 0
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def evaluate_one_epoch():
    """
    Evaluate the model for one epoch: switch to eval mode, compute metrics, then switch back to train mode.
    """
    stat_dict = {}
    net.eval()
    iou_calc = IoUCalculator(cfg)

    for batch_idx, batch_data in enumerate(validation_dataloader):

        for key, val in batch_data.items():
            if isinstance(val, list):
                batch_data[key] = [v.to(device) for v in val]
            else:
                batch_data[key] = val.to(device)


        with torch.no_grad():
            end_points = net(batch_data)
            loss, end_points = compute_loss(end_points, cfg, device)
            acc, end_points = compute_acc(end_points)


        iou_calc.add_data(end_points)


        for key, value in end_points.items():
            if 'loss' in key or 'acc' in key:
                stat_dict.setdefault(key, 0)
                stat_dict[key] += value.item()


    for key in sorted(stat_dict.keys()):
        log_string(f'eval mean {key}: {stat_dict[key]/(batch_idx+1):.6f}')


    mean_iou, iou_list = iou_calc.compute_iou()
    log_string(f'mean IoU:{mean_iou * 100:.1f}%')
    log_string('-' * 86)

    s = f'{mean_iou*100:.1f} | ' + ' '.join(f'{100*i:.2f}' for i in iou_list)
    log_string(s)
    log_string('-' * 86)

    net.train()
    return mean_iou


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    now_miou = 0
    max_miou = 0
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        training_dataset.current_epoch = epoch
        log_string('**** EPOCH %03d ****' % (epoch))

        log_string(str(datetime.now()))

        np.random.seed()
        train_one_epoch()


        log_string('**** EVAL EPOCH %03d START****' % (epoch))
        now_miou = evaluate_one_epoch()


        if(now_miou>max_miou):
            save_dict = {'epoch': epoch+1,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }
            try:
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
            max_miou = now_miou

        log_string('Best mIoU = {:2.2f}%'.format(max_miou*100))
        log_string('**** EVAL EPOCH %03d END****' % (epoch))
        log_string('')


if __name__ == '__main__':

    train(start_epoch)

