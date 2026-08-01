# MPF_VI-ReID

**Modality Purification Framework for Visible-Infrared Person Re-Identification**

---

## Overview

The repository supports training and evaluation on the following benchmark datasets:

- SYSU-MM01
- RegDB
- LLCM

---

## Features

- End-to-end PyTorch implementation
- Mixed precision (FP16) training
- TensorBoard logging
- Automatic checkpoint saving
- Resume training from checkpoints
- Adaptive Query Expansion (AQE)
- Re-ranking support
- Multi-dataset evaluation
- Reproducible experiments

---

## Repository Structure

```
MPF_VI-ReID/
│
├── apex/
├── assets/
├── configs/
│   ├── SYSU.yml
│   ├── RegDB.yml
│   └── LLCM.yml
├── data/
├── engine/
├── losses/
├── models/
├── utils/
├── main.py
└── README.md
```

---

## Requirements

- Python 3.9+
- PyTorch
- CUDA 11+
- torchvision
- numpy
- scipy
- pytorch-ignite
- tensorboard
- opencv-python
- pyyaml
- tqdm

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the datasets from their official sources.

Supported datasets:

- SYSU-MM01
- RegDB
- LLCM

Organize them as

```
data/

├── SYSU-MM01/
├── RegDB/
└── LLCM/
```

Update the dataset root path inside the configuration file if necessary.

---

## Training

Example (SYSU-MM01)

```bash
python main.py --cfg configs/SYSU.yml --gpu 0
```

Example (RegDB)

```bash
python main.py --cfg configs/RegDB.yml --gpu 0
```

Example (LLCM)

```bash
python main.py --cfg configs/LLCM.yml --gpu 0
```

---

## Testing

Testing is automatically performed after training.

To evaluate a pretrained model, specify the checkpoint path in the configuration file or enable

```
resume: path/to/checkpoint.pth
test_only: True
```

Then run

```bash
python main.py --cfg configs/SYSU.yml --gpu 0
```

---

Configuration files are located in

```
configs/
```

---


## Citation

If you find this repository useful, please cite our paper.

```bibtex
@article{MPF2026,
  title={MPF: Multi-Scale Progressive Feature Learning for Visible-Infrared Person Re-Identification},
  author={Vipin Gautam and Co-authors},
  journal={IEEE},
  year={2026}
}
```

Please replace the citation with the final published version after acceptance.

---

## Acknowledgements

This work builds upon several excellent open-source person re-identification frameworks. We thank the authors for making their implementations publicly available.

---

## Contact

**Vipin Gautam**

For questions regarding the implementation, please open an Issue on GitHub.
