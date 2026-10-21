# WaveSS-net
Official implementation of WaveSS-Net: Wavelet-Guided Structural Prior and State-Space Learning for Large-Scale MLS Point Cloud Semantic Segmentation.
## Environment

The experimental setup reported in the paper uses Python 3.9, PyTorch 1.13.1, and CUDA 11.8. The implementation also uses NumPy, pandas, scikit-learn, PyWavelets (`pywt`), `hilbertcurve`, and `mamba_ssm`. The C++ grid-subsampling extension imported by `helper_tool.py` must be available in the Python environment. Install compatible versions of the dependencies for your CUDA and PyTorch setup.

## Toronto-3D Data Preparation

Run preprocessing

```bash
python convert_toronto3d.py
```

The preprocessing script produces `original_ply/` and `input_0.060/`, including subsampled PLY files, KDTree and projection caches, `knn/` neighbor caches, `wavelet_features/` residual caches, and Hilbert order files. The dataset loader reads these files during training and evaluation.

## Training

Toronto-3D uses L002 for validation/testing and the remaining sections for training. The default configuration uses AdamW, batch size `4`, and `64` epochs.

```bash
python main_toronto3d.py 
```

The best validation checkpoint is saved as `train_output/<timestamp>/checkpoint.tar` by default. Use `--checkpoint_path` to resume training from a saved checkpoint.



The evaluation script accumulates point-wise probabilities through voting, projects predictions to the original point cloud, reports overall accuracy (OA), mean intersection over union (mIoU), and per-class IoU, and saves prediction PLY files under `test_output/<timestamp>/val_preds/`.

## Reported Toronto-3D Result

| Test split | OA | mIoU | Ground | Road markings | Natural | Building | Utility line | Pole | Car | Fence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L002 | 97.1% | 86.4% | 95.61% | 86.48% | 95.37% | 87.86% | 87.25% | 84.84% | 89.10% | 64.86% |

These are the Toronto-3D results reported in the WaveSS-Net manuscript. Class-wise values are IoU scores.

## Acknowledgments

We thank the authors and contributors of the following open-source projects for their work:

- [Mamba](https://github.com/state-spaces/mamba)
- [RandLA-Net](https://github.com/QingyongHu/RandLA-Net)
- [PointNet++](https://github.com/charlesq34/pointnet2)
- [RandLA-Net-Pytorch-New](https://github.com/liuxuexun/RandLA-Net-Pytorch-New)

These projects provided valuable foundations and references for the development of this implementation.


