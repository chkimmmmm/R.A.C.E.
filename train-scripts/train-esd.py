from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
import json

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers


def pgd_attack(unet, latent_model_input, t, text_embeddings, gaussian_noise, adv_loss, txt_min_max, num_iter=10, alpha=2/255, epsilon = 8/255, from_generation_code = True): 
    #REF:https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD
    epsilon = epsilon
    random_start = True
    
    conditional_emb = text_embeddings.clone().detach().to(text_embeddings.device)
    adv_conditional_emb = conditional_emb.clone().detach()
    
    max_val = txt_min_max['max']
    min_val = txt_min_max['min']
    
    if random_start:
        adv_conditional_emb = adv_conditional_emb + torch.empty_like(adv_conditional_emb).uniform_(
                -epsilon, epsilon)
        adv_conditional_emb = torch.clamp(adv_conditional_emb, min=min_val, max=max_val).detach()
    
    for _ in range(num_iter):
        adv_conditional_emb.requires_grad = True
        
        if from_generation_code:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=adv_conditional_emb).sample
        else:
            noise_pred = unet.apply_model(latent_model_input, t, adv_conditional_emb)
        
        # - delta is necessary because we are minimizing the loss
        loss = - adv_loss(gaussian_noise, noise_pred) 
        grad = torch.autograd.grad(loss, adv_conditional_emb, retain_graph=False, create_graph=False)[0]
        
        adv_conditional_emb = adv_conditional_emb.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_conditional_emb - conditional_emb, min=-epsilon, max=epsilon)
        adv_conditional_emb = torch.clamp(conditional_emb + delta, min=min_val, max=max_val).detach()
                
    
    delta = adv_conditional_emb - conditional_emb
    return adv_conditional_emb, delta, loss

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def train_esd(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, saving_path, seperator=None, image_size=512, ddim_steps=50):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')
    if prompt == 'allartist':
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
    print(words)
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)
    
    if args.adv_train: #If it is adv training, load pretrained ESD model's weight.
        try:
            print("Adv Training is activated")
            print("load Unet part from pretrained esd")
            model_weight = torch.load(args.pretrained_esd, map_location=devices[0])
            model.load_state_dict(model_weight)
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')
            exit() #This loading is important. So, if it fails, we need to exit.

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    if args.lasso:
        print("Lasso is activated")
        lasso_loss_fn = torch.nn.L1Loss()
        lasso_lambda = 0.1
        original_parameters = [param.clone().detach() for param in parameters]


    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    if args.adv_train:
        name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}-adv-{args.atk_method}-eps-{args.epsilon}-exp-{args.exp_name}'
    else:
        name = f'compvis-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}'
    
    
    folder_path = f'{saving_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    args_dict = vars(args)
    args_filename = os.path.join(folder_path, 'args_debug.json')
    with open(args_filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    
    #RACE
    if args.adv_train:
        if args.adv_loss == "l1":
            adv_loss = torch.nn.L1Loss()
        elif args.adv_loss == "l2": #Default
            adv_loss = torch.nn.MSELoss()
        else:
            raise ValueError('Invalid adv_loss')
    
    
    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words,1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = model.get_learned_conditioning(['']) 
        emb_p = model.get_learned_conditioning([word]) 
        emb_n = model.get_learned_conditioning([f'{word}']) 
        
        txt_emb_min_max = {'min': emb_n.min(), 'max': emb_n.max()}

        opt.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc)) # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            e_p = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_p.to(devices[1]))
        
        if args.adv_train:
            attacked_emb_n, delta, loss = pgd_attack(model, z, t_enc_ddpm, emb_n.to(devices[0]), start_code, adv_loss, txt_emb_min_max, alpha=args.epsilon/4., num_iter=10, epsilon = args.epsilon, from_generation_code=False)
        else:
            attacked_emb_n = emb_n
            
        # breakpoint()
        # get conditional score from ESD model
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), attacked_emb_n.to(devices[0]))
        e_0.requires_grad = False
        e_p.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(e_n.to(devices[0]), e_0.to(devices[0]) - (negative_guidance*(e_p.to(devices[0]) - e_0.to(devices[0])))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        if args.lasso:
            lasso_loss = sum(lasso_loss_fn(param, original_param) for param, original_param in zip(parameters, original_parameters))
            loss += lasso_lambda * lasso_loss
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
                
        if loss > 500.:
            raise ValueError('Loss is too high, training is not working')
        
        # save checkpoint and loss curve
        if (i+1) % 1000 == 0 and i+1 != iterations and i+1>= 500:
            #save_model(model, name, i-1, saving_path, save_compvis=True, save_diffusers=False)
            save_model(model, name, i, saving_path = saving_path, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

        # if i % 10 == 0:
        #     save_history(losses, name, word_print, saving_path)

    model.eval()
    save_model(model, name, args.iterations, saving_path = saving_path, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print, saving_path)

def save_model(model, name, num, saving_path, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = os.path.join(saving_path, name)
    os.makedirs(folder_path, exist_ok=True)
    name = f'{name}-epoch_{num}'
    if num is not None:
        path = f'{folder_path}/{name}.pt'
    else:
        path = f'{folder_path}/{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, saving_path=folder_path, device=device)

def save_history(losses, name, word_print, saving_path):
    folder_path = f'{saving_path}/{name}'
    #folder_path = os.path.join(saving_path, name)
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--saving_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--adv_train', help="adv training", action='store_true')
    parser.add_argument('--atk_method', help="atk method", type=str, default="pgd")
    parser.add_argument('--epsilon', help="adv attack epsilon", type=float, default=0.1)
    parser.add_argument('--pgd_num_step', help="numstep for PGD", type=int, default=10)
    parser.add_argument('--pretrained_esd', help="use pretrained esd", type=str)
    parser.add_argument('--adv_loss', help="l2, l1", type=str, default="l2")
    parser.add_argument('--lasso', help="Lasso", action = 'store_true')
    parser.add_argument('--exp_name', help="experiment_name", type=str, default="")
    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    saving_path = args.saving_path

    train_esd(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, saving_path=saving_path)
