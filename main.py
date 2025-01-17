import sys
import os 
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from taming.models import vqgan 

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim_mask import DDIMSampler

from torchvision.utils import save_image
from torch.backends import cudnn
import numpy as np


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(): 
    config = OmegaConf.load("ldm.yaml")  # configs/latent-diffusion/cin256-v2.yaml
    model = load_model_from_config(config, "model.ckpt") # https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
    return model

seed = 0
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

seedEverything()
#------------------------------------------------------------------ #  

model = get_model()  

import clip 
import torchvision
device = model.device

clip_rn_50,_ = clip.load('RN50', device=device)
clip_rn_101,_ = clip.load('RN101', device=device)
clip_vit_b_16,_ = clip.load('ViT-B/16', device=device)
clip_vit_b_32,_ = clip.load('ViT-B/32', device=device)
clip_vit_l_14,_ = clip.load('ViT-L/14', device=device)
models = [clip_rn_50, clip_rn_101, clip_vit_b_16, clip_vit_b_32]
# models = [clip_vit_b_32]
clip_preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(clip_vit_b_32.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        # torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        torchvision.transforms.CenterCrop(clip_vit_b_32.visual.input_resolution),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
    ]
)
final_preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(clip_vit_b_32.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        # torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        torchvision.transforms.CenterCrop(clip_vit_b_32.visual.input_resolution),
    ]
)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)
input_res = 224
cle_data_path = '/cle_data'  # original image, imagenet dataset
res = sorted(os.listdir(cle_data_path))
name_key = {}
for i,n in enumerate(res):
    key = '%05d' % i
    value = n.split('.')[0]
    name_key[key] = value

cle_data_path = '/cle_data' # original image, imagenet dataset
tgt_data_path = '/tgt_data' # target image, generated by stable diffusion
batch_size = 1
num_samples = 1000
transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        ]
    )
transform_fn_org = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(), # [0, 1]
        torchvision.transforms.Lambda(lambda img: (img * 2 - 1)),
        # torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        # torchvision.transforms.Lambda(lambda img: to_tensor(img)),
    ]
)

clean_data    = ImageFolderWithPaths(cle_data_path, transform=transform_fn_org)
target_data   = ImageFolderWithPaths(tgt_data_path, transform=transform_fn)

data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False)
data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False)


sampler = DDIMSampler(model, models=models, preprocess=clip_preprocess)  


import numpy as np 
from PIL import Image
from einops import rearrange 
from torchvision.utils import make_grid
import math
import cv2

n_samples_per_class = 1
import pandas as pd
data = pd.read_csv('images.csv')  # label, imagenet dataset label, /data/images.csv
labeles = {}
for i in range(1000):
    labeles[data['ImageId'][i]] = data['TrueLabel'][i]


ddim_steps = 200 # 200
ddim_eta = 0.0
scale = 5.0   # for unconditional guidance

img_transformed_list = []
cam_root = '/mask' # mask root, generated by GradCAM

for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):

    image_org = image_org.to(device)
    image_tgt = image_tgt.to(device)
    # # get tgt featutres
    with torch.no_grad():
        tgt_image_features_list=[]
        image_tgt = clip_preprocess(image_tgt)
        for clip_model in models:
            tgt_image_features = clip_model.encode_image(image_tgt)  # [bs, 512]
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
            tgt_image_features_list.append(tgt_image_features)
            
    # with torch.no_grad():
    #     org_image_features_list=[]
    #     org = image_org.clone()
    #     org = clip_preprocess(org)
    #     for clip_model in models:
    #         org_image_features = clip_model.encode_image(org)  # [bs, 512]
    #         org_image_features = org_image_features / org_image_features.norm(dim=1, keepdim=True)
    #         org_image_features_list.append(org_image_features)
    
    with torch.no_grad():
        with model.ema_scope():  
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                ) 
            label_id = path[0].split('/')[-1].split('.')[0]
            label_id = name_key[label_id] 
            class_label = labeles[label_id]
            all_samples = list()
            all_labels = list() 
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            encoder_posterior = model.encode_first_stage(image_org)
            z = model.get_first_stage_encoding(encoder_posterior).detach()
            cam = cv2.imread(cam_root+label_id+'.png', 0) / 255.
            cam = cv2.resize(cam, (64, 64))
            cam = torch.tensor(cam).float()
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            x_T=z,
                                            batch_size=n_samples_per_class,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=ddim_eta,
                                            label=xc.to(model.device),
                                            tgt_image_features_list=tgt_image_features_list,
                                            org_image_features_list=None,
                                            cam=cam,
                                            K=1,s=35,a=5)
            
            
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                        min=0.0, max=1.0)
        img_transformed = clip_preprocess(x_samples_ddim).to(device) # image transformation to model input
        img_transformed_list.append(x_samples_ddim)
        adv_image_feature_list = []
        for clip_model in models:
            adv_image_features = clip_model.encode_image(img_transformed)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
            adv_image_feature_list.append(adv_image_features)

        path = "{}.png".format(i)
        torchvision.utils.save_image(final_preprocess(x_samples_ddim), path)
        

