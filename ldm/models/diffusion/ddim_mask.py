"""SAMPLING ONLY."""

import os
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torchvision.transforms as T

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import torch.nn.functional as F
from torch.backends import cudnn
import random

from torchvision.utils import save_image, make_grid


DEFAULT_RANDOM_SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def sample_coordinates(prob_matrix):

    flattened_probs = prob_matrix.flatten()

    sampled_index = np.random.choice(flattened_probs.size, p=flattened_probs)

    sampled_coordinates = np.unravel_index(sampled_index, prob_matrix.shape)
    
    return sampled_coordinates

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

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)

# def linear_beta_schedule(timesteps):
#     scale = 1000 / timesteps
#     beta_start = scale * 0.0001
#     beta_end = scale * 0.02
#     return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
# l = linear_beta_schedule(1000).to(device)

def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def get_target_label(logits, label, device): # seond-like label for attack
    
    rates, indices = logits.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  
    
    tar_label = torch.zeros_like(label).to(device)
    
    for i in range(label.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]
    
    return tar_label


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", models=None, preprocess=None, **kwargs):
        
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        
        self.models = models
        self.preprocess = preprocess        


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr) 

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,  
               batch_size, 
               shape, 
               conditioning=None, 
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               cam=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1., 
               unconditional_conditioning=None, 
               tgt_image_features_list=None,
               org_image_features_list=None,
               label=None,K=10,s=2,a=1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        seedEverything()
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    cam=cam, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,label=label,
                                                    tgt_image_features_list=tgt_image_features_list,
                                                    org_image_features_list=org_image_features_list,
                                                    K=K,s=s,a=a
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      cam=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,label=None,
                      tgt_image_features_list=None,
                      org_image_features_list=None,
                      K=10,s=0.75,a=0.5):
        device = self.model.betas.device
        # print(self.model.betas.shape)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
            # print(img.shape, torch.max(img), torch.min(img))
        else:
            z = x_T
            # encoder_posterior = self.model.encode_first_stage(x_T)
            # z = self.model.get_first_stage_encoding(encoder_posterior).detach()
            # z = x_T
            t = torch.full((1,), 251, device=device, dtype=torch.long) 
            img = self.model.q_sample(z, t, noise=torch.randn_like(z.float()))
            # print(img.shape, torch.max(img), torch.min(img))
            # img = torch.randn(shape, device=device)
            

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]


        pri_img = img.detach().requires_grad_(True)
        
        for k in range(K):
            
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
            
            img = pri_img.detach().requires_grad_(True)
            
            print(f"Running Adversarial Sampling at {k} step")
            
            print(f"Running DDIM Sampling with {total_steps} timesteps")
        
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            
            costs = torch.zeros([72, 5])
            weights = torch.zeros([72, 5])
            Temp = 2
            N_models = 4
            idx_time = 149
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                if(index > total_steps * 0.25):
                    continue
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                mask = cam.clone().to(device)
                mask = torch.clamp(mask, 0.0, 1)
                cam_new = torch.clamp(cam, 0.3, 0.7)
                # mask = cam_new.to(device)
                prob_matrix = cam_new.numpy()
                prob_matrix /= prob_matrix.sum()  
                (x, y) = sample_coordinates(prob_matrix)
                x_left, x_right = max(0, x-4), min(64, x+4)
                y_left, y_right = max(0, y-4), min(64, y+4)
                mask[x_left:x_right, y_left:y_right] = 0
                # mask = None 
               
                if mask is not None:
                    x0=z 
                    assert x0 is not None
                    # print("using mask", mask.shape, img.shape)
                    img_orig = self.model.q_sample(x0, ts)  
                    img = img_orig * mask + (1. - mask) * img
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
                img, pred_x0 = outs
                
                    
                '''
                if index % 20 == 0:
                    x_samples_ddim = self.model.decode_first_stage(img)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                 min=0.0, max=1.0)
                    save_image(x_samples_ddim, f"img/Diff_{index}.png", nrow=1, normalize=True)
                '''
                    
                if(index > total_steps * 0 and index <= total_steps * 0.25):
                    
                    if i == idx_time or i == idx_time+1:
                        weights[i-idx_time,:] = 1.0
                    else:
                        w1 = costs[i-idx_time-1, 0] / costs[i-idx_time-2, 0]
                        w2 = costs[i-idx_time-1, 1] / costs[i-idx_time-2, 1]
                        w3 = costs[i-idx_time-1, 2] / costs[i-idx_time-2, 2]
                        w4 = costs[i-idx_time-1, 3] / costs[i-idx_time-2, 3]
                        # w5 = costs[i-idx_time-1, 4] / costs[i-idx_time-2, 4]
                        sum_w = torch.exp(w1/Temp) + torch.exp(w2/Temp) + torch.exp(w3/Temp) + torch.exp(w4/Temp) 
                        # weights[i-idx_time, 0] = N_models * torch.exp(w1/Temp) / sum_w
                        # weights[i-idx_time, 1] = N_models * torch.exp(w2/Temp) / sum_w
                        # weights[i-idx_time, 2] = N_models * torch.exp(w3/Temp) / sum_w
                        # weights[i-idx_time, 3] = N_models * torch.exp(w4/Temp) / sum_w
                        # weights[i-idx_time, 4] = N_models * torch.exp(w5/Temp) / sum_w
                        weights[i-idx_time, 0] = sum_w / N_models * torch.exp(w1/Temp)
                        weights[i-idx_time, 1] = sum_w / N_models * torch.exp(w2/Temp)
                        weights[i-idx_time, 2] = sum_w / N_models * torch.exp(w3/Temp)
                        weights[i-idx_time, 3] = sum_w / N_models * torch.exp(w4/Temp)
                    
                    

                    for _ in range(1): 
                        with torch.enable_grad():
                            img_n = img.detach().requires_grad_(True)
                            # img_n = img.clone().requires_grad_(True)
                            img_transformed = self.model.differentiable_decode_first_stage(img_n) # image transformation from latent code
                            img_transformed = torch.clamp((img_transformed+1.0)/2.0, 
                            min=0.0, max=1.0)  
                            img_transformed = self.preprocess(img_transformed)
                            adv_image_feature_list = []
                            for model in self.models:
                                adv_image_features = model.encode_image(img_transformed)
                                adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
                                adv_image_feature_list.append(adv_image_features)
                            loss = torch.zeros(1).to(device)
                            crit_list = []
                            for model_i, (pred_i, target_i) in enumerate(zip(adv_image_feature_list, tgt_image_features_list)):
                                crit1 =  torch.mean(torch.sum(pred_i * target_i, dim=1))  
                                # crit2 = 1 - torch.mean(torch.sum(pred_i * org_i, dim=1)) 
                                costs[i-idx_time, model_i] = crit1.data
                                loss.add_(crit1, alpha=weights[i-idx_time, model_i])
                                crit_list.append(crit1.data.detach().cpu().numpy())
                                # loss.add_(crit2, alpha=0.3)    
                            # print("loss:", crit_list)
                            gradient = torch.autograd.grad(loss, img_n)[0]
                        gradient = torch.clamp(gradient, min=-0.0025, max=0.0025)  
                        img = img + s * gradient
                        # if mask is not None:
                        #     x0=z # z
                        #     assert x0 is not None
                        #     # print("using mask", mask.shape, img.shape)
                        #     img_orig = self.model.q_sample(x0, ts) 
                        #     img = img_orig * mask + (1. - mask) * img
                        # img = img + s * gradient.sign()
                        # img = torch.clamp(img, img_n-self.model.betas[step], img_n+self.model.betas[step])
                    # img = img * mask + (1. - mask) * img_t
                    
                    
                    
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)    
            x_samples_ddim = self.model.decode_first_stage(img)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) # classifier-free guidance

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
