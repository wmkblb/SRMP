# Support-Resampled Meta Prompt Learning (SRMP)

This repository contains the implementation of **Support-Resampled Meta Prompt Learning (SRMP)** for vision-language prompt learning.

SRMP is a **plug-and-play training strategy** designed to improve few-shot prompt generalization by reducing the dependence of prompt updates on specific support-set compositions.

Rather than introducing a new prompt architecture, SRMP can be directly integrated into existing prompt learning methods. Currently, we provide implementations based on:

- **MaPLe + SRMP**
- **MMRL + SRMP**

## Method

Under the 16-shot setting, SRMP randomly divides the samples of each class into two complementary 8-shot support subsets. It performs one-step virtual adaptation on one subset and evaluates the resulting prompt update on the complementary subset.

The two subsets exchange their roles for bidirectional cross-support optimization, and the support partition is resampled at each training epoch.

SRMP:

- introduces no additional learnable parameters;
- modifies only the training procedure;
- requires no additional inference cost;
- can be attached to different prompt learning methods.

## Evaluation

We evaluate SRMP under three settings:

- Base-to-Novel Generalization
- Cross-Dataset Transfer
- Domain Generalization

Experiments are conducted with CLIP ViT-B/16 under the 16-shot setting.

## Code Structure

```text
trainers/
├── maple.py
├── maple_SRMP.py
├── mmrl.py
└── mmrl_SRMP.py
