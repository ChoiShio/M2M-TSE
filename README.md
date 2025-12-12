# M2M-TSE

[![PAPER](https://img.shields.io/badge/ICASSP_2025-paper-green)](https://ieeexplore.ieee.org/abstract/document/10890145)
[![WEBPAGE](https://img.shields.io/badge/Demo-webpage-blue)](https://choishio.github.io/demo_M2M-TSE/)

This repository provides the codes for the __multichannel-to-multichannel target sound extraction (M2M-TSE) using direction and timestamp clues__, presented at ICASSP 2025.

*We propose a multichannel-to-multichannel target sound extraction (M2M-TSE) framework for separating multichannel target signals from a multichannel mixture of sound sources. Target sound extraction (TSE) isolates a specific target signal using user-provided clues, typically focusing on single-channel extraction with class labels or temporal activation maps. However, to preserve and utilize spatial information in multichannel audio signals, it is essential to extract multichannel signals of a target sound source. Moreover, the clue for extraction can also include spatial or temporal cues like direction-of-arrival (DoA) or timestamps of source activation. To address these challenges, we present an M2M framework that extracts a multichannel sound signal based on spatio-temporal clues. We demonstrate that our transformer-based architecture can successively accomplish the M2M-TSE task for multichannel signals synthesized from audio signals of diverse classes in different room environments. Furthermore, we show that the multichannel extraction task introduces sufficient inductive bias in the DNN, allowing it to directly handle DoA clues without utilizing hand-crafted spatial features.*

## Model architecture

Model architecture for M2M-TSE based on direction and timestamp clues.

![Model architecture](./assets/model.jpg)

## Dataset

(To be added)

## Training & Evaluation

Go to the `M2M-TSE` directory:

    cd M2M-TSE

### Training

    python3 -W ignore -m src.training.train experiments/{experiment directory with config.json} --use_cuda --gpu_ids {list of GPU ids used for training, e.g., 0 1 2 3}

### Evaluation

    python3 -W ignore -m src.training.eval experiments/{experiment directory with config.json} --use_cuda --gpu_ids {list of GPU ids used for evaluation, e.g., 0 1 2 3}

## Note

Some components of this repository are based on and modified from:
- Dataset & dataloader: [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) / [Scaper](https://github.com/justinsalamon/scaper) / [Pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- Overall framework of training & evaluation: [Waveformer](https://github.com/vb000/Waveformer)
- Model architecture: [DeFTAN-II](https://github.com/donghoney0416/DeFTAN-II)

Since [Scaper](https://github.com/justinsalamon/scaper) library is licensed under the BSD-3-Clause License, please make sure to include the original license text in your distribution. For details, see the `THIRD_PARTY_LICENSES` file. For other repositories, you can see the `LICENSE` file.

## Citation

    @inproceedings{choi2025multichannel,
        title={Multichannel-to-Multichannel Target Sound Extraction Using Direction and Timestamp Clues},
        author={Choi, Dayun and Choi, Jung-Woo},
        booktitle={Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
        pages={1--5},
        year={2025},
        organization={IEEE},
        address="Hyderabad, India"
    }
