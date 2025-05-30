# Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models
This repository is an official implementation of the paper "Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models". 

## TODO
- [x] Release [Arxiv paper](https://arxiv.org/abs/2404.10335)
- [x] Release core code
- [x] Release adversarial example generation code
- [ ] Release test code
## Introduction
![image](https://github.com/user-attachments/assets/3c24c832-0cb6-4295-ab77-0c8cfe47efe1)
we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation
(AEGE) to modify the score during the diffusion model’s reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask Generation (GCMG) to disperse adversarial semantics throughout the image rather than concentrating them in a single area.

## Quick Start
The target data and evaluation methods can be found in AttackVLM. 

The weights of the adversarial sample generation model are in https://github.com/CompVis/latent-diffusion.

The code of GradCAM is in https://github.com/ramprs/grad-cam/. 

The hyperparameters in ldm/models/diffusion/ddim_main can be adjusted to obtain a trade-off between attack capability and image quality.

python demo.py

python main.py

## Citation
```
@article{guo2024efficient,
  title={Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models},
  author={Guo, Qi and Pang, Shanmin and Jia, Xiaojun and Liu, Yang and Guo, Qing},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```
## Related work
[AdvDiffuser](https://github.com/lafeat/advdiffuser)
[AttackVLM](https://github.com/yunqing-me/AttackVLM)
[Attack-Bard](https://github.com/thu-ml/Attack-Bard)
[SIA](https://github.com/xiaosen-wang/SIT)
