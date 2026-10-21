# WaveSS-net
Official implementation of WaveSS-Net: Wavelet-Guided Structural Prior and State-Space Learning for Large-Scale MLS Point Cloud Semantic Segmentation.
## Requirements

The research environment uses Python 3.9, PyTorch 1.13.1 and CUDA 11.8. For installation, choose mutually compatible PyTorch, CUDA and `mamba-ssm` builds for your GPU.

```bash
pip install -r requirements.txt
pip install mamba-ssm
```

## Toronto-3D dataset

Download Toronto-3D from its dataset provider and prepare the raw point clouds. The current code supports `.ply` files with XYZ, RGB and semantic label fields, and text files containing XYZ, RGB and a final label column.

The experiment uses the L002 split:

- **Training:** L001, L003 and L004
- **Testing:** L002
- **Classes:** 8 semantic classes; raw label 0 (Unclassified) is ignored.

### Preprocessing

```bash
python preprocess.py --raw_dir /path/to/Toronto_3D --out_dir data/processed
```

The preprocessing script produces downsampled point clouds, point-level Hilbert wavelet priors and the original-cloud projection indices.

## Training

```bash
python train_toronto3d.py --data_dir data/processed --output runs/toronto
```

Training options and network hyperparameters are defined in `wavessnet/config.py`. Additional command-line options are available with `--help`.

## Evaluation

```bash
python test_toronto3d.py --data_dir data/processed --checkpoint runs/toronto/last.pth --votes 100
```

The evaluation script uses label-free spatial coverage, probability voting and projection back to the original point cloud. It reports overall accuracy (OA) and mean intersection-over-union (mIoU) across eight valid classes.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements

We thank the authors and contributors of the following research projects:

- **RandLA-Net** — Qingyong Hu et al., *RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds*, CVPR 2020. [Project](https://github.com/QingyongHu/RandLA-Net)
- **PointNet++** — Charles R. Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*, NeurIPS 2017. [Project](https://github.com/charlesq34/pointnet2)
- **Mamba** — Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2024. [Project](https://github.com/state-spaces/mamba)

We also acknowledge the creators of the Toronto-3D dataset and the open-source libraries used in this implementation. Please cite the original works where appropriate.

## Citation

Paper citation details will be added when available.


