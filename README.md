# Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models
This repository is an official implementation of the paper "Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models"

## TODO
- [x] Release [Arxiv paper](https://arxiv.org/abs/2404.10335)
- [ ] Release core code
- [ ] Release adversarial example generation code
- [ ] Release test code
## Introduction
![image](https://github.com/user-attachments/assets/3c24c832-0cb6-4295-ab77-0c8cfe47efe1)
we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation
(AEGE) to modify the score during the diffusion model’s reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask Generation (GCMG) to disperse adversarial semantics throughout the image rather than concentrating them in a single area.

## Citation
```
@article{guo2024efficiently,
  title={Efficiently Adversarial Examples Generation for Visual-Language Models under Targeted Transfer Scenarios using Diffusion Models},
  author={Guo, Qi and Pang, Shanmin and Jia, Xiaojun and Guo, Qing},
  journal={arXiv preprint arXiv:2404.10335},
  year={2024}
}
```
