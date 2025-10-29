# coding=utf-8
import os
import json
import time
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchprofile import profile_macs
from fvcore.nn import FlopCountAnalysis

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import clip
from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes, cifar10_classes
from data.datautils import AugMixAugmenter, build_dataset, AugMixAugmenter_Asif
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from params import get_args_rl, get_args_kd
from clip_reward import get_reward_model
from utils import kd_distill_loss_v2, dkd_distill_loss, atkd_distill_loss, KDClipLoss
from utils.metrics import ECELoss, ClasswiseECELoss, Calculator, ECE_Loss
# ece_metric, cece_metric = ECELoss(), ClasswiseECELoss()
from utils import plot_img, plot_pil_img, plot_features, CLIPInterpreter

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

CLASSNAMES = []
targeted_class_dict = {i: value for i, value in enumerate(CLASSNAMES)}
clip_features = []
clip_labels = []

def confidence_filter(logits: torch.Tensor, probs: torch.Tensor, top:float, return_idx=False):
    batch_entropy = -(probs * probs.log()).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    if not return_idx:
        return logits[idx]
    return logits[idx], idx

def break_sample_tie(ties, logit, device):
    ties = torch.tensor(ties, dtype=torch.int, device=device)
    logit[~ties] = -torch.inf
    scalar_pred = torch.argmax(logit, dim=-1)
    return scalar_pred

def greedy_break(ties, logits, device):
    ties_tensor = torch.tensor(ties, dtype=torch.int, device=device)
    preds = torch.argmax(logits, dim=1)
    for pred in preds:
        if pred in ties_tensor:
            return pred
    return break_sample_tie(ties, logit=logits[0], device=device)

@torch.jit.script
def calc_energy(x: torch.Tensor, temp_factor: float = 1.0) -> torch.Tensor:
    return temp_factor * torch.logsumexp(x / temp_factor, dim=1)

class EnergyLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(EnergyLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        e = calc_energy(x, self.temp_factor)
        # energy = 1.0 / torch.linalg.vector_norm(6.0 - energy, 2)
        e = 1.0 / e.mean()
        return e


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)      # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def entropy_of_opinion(x: torch.Tensor): #key component of COME
    K = 1000
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True) * torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    brief = torch.exp(x)/(torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    uncertainty = K / (torch.sum(torch.exp(x), dim=1, keepdim=True) + 1000)
    probability = torch.cat([brief, uncertainty], dim=1) + 1e-7
    entropy = -(probability * torch.log(probability)).sum(1)
    return entropy.mean(0)
    
def compute_inter_view_cross_entropy(output):
    """
    Computes the mean inter-view cross-entropy loss of the predicted distributions
    :param preds: Tensor of shape (batch_size, num_classes), predicted probabilities for each view
    :return: Mean inter-view cross-entropy loss
    """
    preds = output.softmax(1)
    # Get the number of views in the batch
    num_views = preds.size(0)
    # Initialize total cross-entropy loss
    total_ce_loss = 0.0
    count = 0
    # Compute cross-entropy loss for each pair of views
    for i in range(num_views):
        for j in range(num_views):
            if i==j : continue
            # Cross-entropy loss between predicted distribution of view i and view j
            ce_loss_ij = F.cross_entropy(preds[i], preds[j])
            total_ce_loss += ce_loss_ij
            count += 1
    # Return the mean cross-entropy loss over all pairs
    if count > 0:
        return total_ce_loss / count
    else:
        return 0.0
    
# num_images = 7500
# tta_steps = 5
# mean_dist_matrix = torch.zeros((num_images, tta_steps))  # Shape: (7500, 5)

# Raza modified this function to test other KD losses
def test_time_tuning(model, images, optimizer, scaler, args, set_id=None, image_idx=None):
    selected_idx = None
    # output = None
    # output2 = None
    # single_output = None

    shape = images.shape[1:]
    eps = 1/255.0 #hyperparameter (Raza checked that 1/255 is the best value)
    noise_lr = 0.001 #hyperparameter (ToDo)
    # noise = torch.zeros(shape).to(images.device)  
    # noise = torch.rand(shape).to(images.device) * 2 * eps - eps
    noise = torch.randn(shape).to(images.device).clamp(-eps, eps)
    noise = noise.unsqueeze(0).detach() # [1, 3, 224, 224]
    
    # noise_0 = noise.clone()
    # folder = f"/home/raza.imam/Documents/RLCF/RLCF/plots/attentions/dataset_{set_id}/{CLASSNAMES[TARGET.item()]}/{image_idx}/"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # clip_interpreter = CLIPInterpreter(device=images.device, start_layer=-1, start_layer_text=-1)
    # # attn for clean image
    # texts = "a photo of a" + CLASSNAMES[TARGET.item()]
    # plot_pil_img(noise_0.squeeze(0), save_path=folder+f"clean_noise"+".png")
    # plot_pil_img(images[0].squeeze(0), save_path=folder+f"clean_img"+".png")
    # clean_attn = clip_interpreter.forward(texts, images[0].unsqueeze(0), save_dir=folder, file_name=f"clean_img_attn.png")
    
    # lower param tnt --------------------------------
    # shape = (224, 9)
    # noise = torch.randn(shape, requires_grad=True, device=images.device).clamp(-eps, eps)
    # padded_noise = F.pad(noise, (0, 224 - 9))
    # noise = padded_noise.unsqueeze(0).unsqueeze(0)
    # noise.retain_grad()
    # lower param tnt --------------------------------
    
    for j in range(args.tta_steps):
        
        noise.requires_grad = True
        inputs = images + noise
        # inputs = images.clone() ### lower param tnt
        # inputs[:, 0, :, :] += noise.squeeze(0).squeeze(0) ### lower param tnt
        inputs = normalize(torch.clamp(inputs, 0, 1))
        
        with torch.cuda.amp.autocast():
            # here only play confident selection once
            if selected_idx is not None:
                output, image_feats, text_feats = model(inputs)
                output, image_feats, text_feats = output[selected_idx], image_feats[selected_idx], text_feats
            else:
                output, image_feats, text_feats = model(inputs)
                # entropy = avg_entropy(output)  # Compute entropy
                # gamma = 0.3*(1.0 / (1 + entropy))  # Lower gamma for higher entropy
                # output, selected_idx = select_confident_samples(output, gamma)
                output, selected_idx = select_confident_samples(output, args.selection_p)
                image_feats = image_feats[selected_idx]
                
                # reward_model.set_image_features(inputs[selected_idx])
                # For KD
                # r_logits_per_image, r_logits_per_text = reward_model.calulate_similarity()
                # r_logits_per_image, r_logits_per_text = None, None


            # bs = output.shape[0]

            # # top-k sample results
            # value, index = torch.topk(output, sample_k, dim=-1)
            # flatten_index = index.flatten()
            # # reward calculation
            # clip_score = reward_model.CLIPScore(class_index=flatten_index, pairwise=False)     
            # rewards = reward_model.rewards_post_process(clip_score if reward_model.process_batch else clip_score.reshape(bs, -1))

            # rep_output = torch.repeat_interleave(output, sample_k, dim=0)
            # all_loss = F.cross_entropy(rep_output, flatten_index, reduction='none')
            # rlcf_loss = torch.mean(rewards * all_loss)

            # SaLs # Robust Calibration of Large Vision-Language Adapters
            # with torch.no_grad():
            #     if j == 0:
            #         zs_pred = output[0,:]
            #         min_logits_zs, max_logits_zs = torch.min(zs_pred), torch.max(zs_pred)
        
            #     min_logits_tpt, max_logits_tpt = torch.min(output,1)[0].unsqueeze(1), torch.max(output,1)[0].unsqueeze(1)
            
            # output_norm = (output - min_logits_tpt)/ (max_logits_tpt - min_logits_tpt)
            # output_norm = output_norm * (max_logits_zs - min_logits_zs) + min_logits_zs
            # output = output_norm
            
            # diff = image_feats - model(images[selected_idx])[1]
            # l2_loss = torch.norm(diff, dim=-1).mean()
            # clip_loss = torch.diag(output).mean()
            
            dists = torch.cdist(image_feats, image_feats, p=2)
            upper_triangle_values = dists.triu(diagonal=1)
            # Filter out the zeros (as the non-upper-triangular part will be zeros)
            mean_dist = upper_triangle_values[upper_triangle_values > 0].mean()
            
            # mean_dist_matrix[image_idx, j] = mean_dist.item()

            if args.min_entropy_reg:
                alpha = args.min_entropy_w #hyperparameter (ToDo)
                beta = 0.1
                loss = alpha * avg_entropy(output) + mean_dist #beta * compute_inter_view_cross_entropy(output)
            else:
                loss = 0
            # loss = rlcf_loss + loss
            
            # if args.kd_loss == 'DKD':
            #     kd_loss = dkd_distill_loss(output, r_logits_per_image, r_logits_per_image.argmax(-1))
            # elif args.kd_loss == 'ATKD':
            #     kd_loss = atkd_distill_loss(output, r_logits_per_image)
            # elif args.kd_loss == 'KD':
            #     kd_loss = kd_distill_loss_v2(output, r_logits_per_image)
            # elif args.kd_loss[-4:] == 'CLIP':
            #     kd_loss = kd_clip_loss(reward_model.image_features,
            #                          reward_model.class_features,
            #                          image_feats,
            #                          text_feats,
            #                          args.kd_loss) # weight = 2000 (as per CLIP-KD paper)
            # else:
            #     raise NotImplementedError
            
            # loss = kd_loss + loss

            # if True:# 'ctpt' in args.run_type:
            #     if output == None and output2 == None:
            #         single_output = model(inputs[0]).squeeze(0)

            #     lambda_ = 20 if set_id in ['A', 'V', 'R', 'K'] else 50
            #     loss += (-lambda_* model.l2_norm_mean_training)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        noise_grad = noise.grad.data 
        noise = noise - torch.sign(noise_grad)*noise_lr
        # noise = noise - noise_grad * noise_lr
        noise = noise.clamp(-eps, eps).detach()
        
        # # -------------------------------- ATTN --------------------------------
        # noisy_attn = clip_interpreter.forward(texts, images[0]+noise, save_dir=folder, file_name=f"{j}_attn_on_noisy_img")
        # plot_pil_img((images[0]+noise).squeeze(0), save_path=folder+f"{j}_noisy_img"+".png")
        # plot_pil_img(noise.squeeze(0), save_path=folder+f"{j}_noise"+".png")
        # # -------------------------------- ATTN --------------------------------
    
        # # -------------------------------- ATTN --------------------------------        
        # # diff between clean and adapted
        # # for noise diff
        # diff_noise = np.abs(noise_0.cpu() - noise.cpu()); diff_noise = np.where(diff_noise > 0.003, diff_noise, 255)  # Thresholding and setting small differences to white
        # Image.fromarray(diff_noise.squeeze(0).reshape(224,224,3).astype(np.uint8)).save(folder+f"diff_noise_{j}.png")  
        # # for image diff
        # diff_img = np.abs(images[0].cpu() - (images[0]+noise).cpu()); diff_img = np.where(diff_img > 0.0035, diff_img, 255) 
        # Image.fromarray(diff_img.squeeze(0).reshape(224,224,3).astype(np.uint8)).save(folder+f"diff_img_{j}.png") 
        # # for attn diff 
        # diff_attn = np.abs(clean_attn - noisy_attn); diff_attn = np.where(diff_attn > 5, diff_attn, 255)  
        # Image.fromarray(diff_attn.astype(np.uint8)).save(folder+f"diff_attn_{j}.png") 
        # # -------------------------------- ATTN --------------------------------
    
    return noise.detach()

def test_time_zero_tuning(model, inputs, optimizer, scaler, args, reward_model=None, transform=None):
    selected_idx = None
    temp = 1/model.logit_scale.exp()
    zero_temp = torch.finfo(torch.float16).eps

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            if selected_idx is not None:
                _, image_feats, text_feats = model(inputs)
                image_feats, text_feats = image_feats[selected_idx], text_feats
                l_filt = image_feats @ text_feats.t() #unscaled logits
            else:
                _, image_feats, text_feats = model(inputs)
                l = image_feats @ text_feats.t() #unscaled logits
                p = (l / temp).softmax(1) # probabilities
                l_filt, selected_idx = confidence_filter(l, p, top=args.selection_p, return_idx=True) # retain most confident views


            # zero-out the temperature, marginalize and predict
            p_bar = (l_filt / zero_temp).softmax(1) # marginalize

            if args.min_entropy_reg:
                loss = 0 + args.min_entropy_w * avg_entropy(p_bar)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
    
@torch.no_grad()
def forward_zero(inputs, updated_model, reward_model=None, noise=None):
    # here, we assume that the 'inputs' are a batch of augmented images,
    # where the first image is the original image and the rest are augmented\

    model = updated_model
    if noise is not None:
        views = inputs + noise
        views = normalize(torch.clamp(views, 0, 1))
    else:
        views = normalize(inputs)

    _, image_feats, text_feats = model(views) # forward pass

    z_txt = text_feats
    z_img = image_feats    

    temp = 1/model.logit_scale.exp()

    # compute probabilities and confidence filter 
    l = z_img @ z_txt.t() # unscaled logits
    p = (l / temp).softmax(1) # probabilities

    gamma = 0.1

    strategy = None
    if strategy == 'entropy':
        # 1. Entropy-Based Gamma Adjustment
        entropy = avg_entropy(l / temp)   # Compute entropy
        gamma = 0.3*(1.0 / (1 + entropy))  # Lower gamma for higher entropy
    elif strategy == 'confidence':
        # 2. Batch-Wise Prediction Confidence
        avg_confidence = p.max(1)[0].mean()  # Average confidence score across batch
        gamma = avg_confidence  # Gamma proportional to the confidence
    elif strategy == 'consistency':
        # 3. Self-Consistency Across Augmentations
        consistency = torch.var(p, dim=0).mean()  # Measure prediction variance across augmentations
        gamma = 1.0 - consistency  # Lower consistency -> lower gamma
    elif strategy == 'confidence_range':
        # 4. Adaptive Based on Model Confidence Range
        confidence_range = p.max(1)[0] - p.min(1)[0]  # Difference between max and min confidence scores
        gamma = confidence_range.mean()  # Adaptive gamma based on confidence spread
    elif strategy == 'diversity':
        # 5. Diversity-Aware Gamma
        diversity_score = torch.mean(1 - torch.cosine_similarity(views[1:], views[0:1]))  # Similarity to the original view
        gamma = diversity_score  # Higher diversity -> higher gamma
    elif strategy == 'uncertainty':
        # 6. Uncertainty-Based View Selection
        uncertainty = torch.var(p, dim=1).mean()  # Variance in predictions for each sample
        gamma = 1.0 / (1 + uncertainty)  # Inverse relationship: higher uncertainty -> lower gamma
    else:
        pass

    l_filt, idx = confidence_filter(l, p, top=gamma, return_idx=True) # retain most confident views
    
    # --------------------------------- T-SNE ---------------------------------
    clip_labels.append(TARGET.item())
    clip_features.append(z_img[idx[0]])
    # --------------------------------- T-SNE ---------------------------------
    
    # zero-out the temperature, marginalize and predict
    zero_temp = torch.finfo(l_filt.dtype).eps
    base_zero_temp = torch.finfo(l_filt.dtype).eps  # base zero_temp (smallest possible value)

    adaptive = True
    entropy_based = False
    variance_based = False
    similarity_based = False
    if adaptive:  # Adaptive zero_temp based on batch uncertainty (entropy of probabilities)
        entropy = avg_entropy(l / temp)  # average batch entropy
        scaling_factor = 1.0 + entropy.item()  # increase temp based on entropy
        zero_temp = base_zero_temp * scaling_factor
    if entropy_based:
        entropy = avg_entropy(l / temp)  # mean entropy over batch
        zero_temp = temp / (1.0 + entropy)  # reducing temperature with increasing entropy
        zero_temp = max(zero_temp, 0.0)  # Ensure zero_temp is not negative
    elif variance_based:
        variance = torch.var(p, dim=0).mean()  # variance of probabilities over batch
        zero_temp = temp / (1.0 + variance)  # Decrease temperature with increasing variance
        zero_temp = max(zero_temp, 0.0)  # Ensure zero_temp is not negative
    elif similarity_based:
        avg_similarity = l.mean()  # average similarity score over batch
        zero_temp = temp * torch.exp(-avg_similarity)  # Decrease temperature as similarity increases
        zero_temp = max(zero_temp, 0.0)  # Ensure zero_temp is not negative
    else:
        pass

    p_bar = (l_filt / zero_temp).softmax(1).sum(0) # marginalize
    
    # check if we have to break ties in some way
    max_counts, scalar_pred = torch.max(p_bar, dim=-1)
    ties = [scalar_pred]
    for i in range(len(p_bar)):
        if i == scalar_pred: continue
        if p_bar[i] == max_counts: ties.append(i)

    # if so, break ties greedily
    if len(ties) > 1:
        k = int(views.size(0) * gamma)
        scalar_pred = greedy_break(ties, l[k:], device=l.device)
        p_bar[scalar_pred]+=1
    
    # need to unsqueeze for compatibility with the 'accuracy' function
    return p_bar.unsqueeze(0)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    # use CoOp model
    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    if args.load is not None:
        print("Use pre-trained soft prompt (CoOp) as initialization {}".format(args.load))
        pretrained_ctx = torch.load(args.load, map_location=f'cuda:{args.gpu}')['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        device = torch.device('cpu')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))

    # define optimizer
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr, weight_decay=args.weight_decay)
    optim_state = deepcopy(optimizer.state_dict())

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')
    torch.backends.cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    all_start = time.time()
    for set_id in datasets:
        data_start_time = time.time()

        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                # normalize
                ])
            data_transform = AugMixAugmenter_Asif(base_transform, preprocess, n_views=args.batch_size-1, 
                                                augmix=len(set_id)>1, hard_aug=args.hard_aug)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
        # reset the model, Reset classnames of custom CLIP model
        if set_id == "cifar10c":
            classnames = cifar10_classes
        elif len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I', 'C']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        model.reset_classnames(classnames, args.arch)
        with torch.cuda.amp.autocast():
            reward_model.set_class_features(tokenized_classes=model.prompt_learner.tokenized_prompts)

        global CLASSNAMES
        CLASSNAMES = classnames
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True)

        results[set_id] = test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args, set_id=set_id, device=device)
        del val_dataset, val_loader

        data_time = time.time() - data_start_time        
        time_log = f'The running time for dataset {set_id} is {data_time // 3600:.1f} Hour {data_time % 3600 / 60:.1f} Minute\n'
        mem_log = f'The maximum GPU memory occupied by this program is {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB\n'
        acc_log = f"\n=> Acc. on testset [{set_id}]: @1 {results[set_id][0]:.2f} / @5 {results[set_id][1]:.2f}"
        ece_log = f"\n=> ECE on testset [{set_id}]: {results[set_id][2]:.4f}\n"

        with open(os.path.join(args.output, "log.txt"), 'a+') as fp:
            fp.write(time_log)
            fp.write(mem_log)
            fp.write(acc_log)
        print(acc_log, ece_log, time_log, mem_log)

    # Outer loop
    # Save output
    # with open(os.path.join(args.output, "results.json"), 'a+') as fp:
    #     json.dump(results, fp, indent=4)
    
    print("======== Result Summary ========\n")
    all_time = time.time() - all_start
    time_log = f'The total running time of the program is {all_time // 3600:.1f} Hour {all_time % 3600 / 60:.1f} Minute\n'
    mem_log = f'The maximum GPU memory occupied by this program is {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB\n'

    with open(os.path.join(args.output, "log.txt"), 'a+') as fp:
        fp.write(time_log)
        fp.write(mem_log)
    print(time_log, mem_log)

    print(f"params: nstep\tlr\tbs")
    print(f"params: {args.tta_steps}\t{args.lr}\t{args.batch_size}")
    print("{:<15} {:<10} {:<10} {:<10}".format("set_id", "Top-1 acc.", "Top-5 acc.", "ECE"))
    for id in results.keys():
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.4f}".format(
            id, results[id][0], results[id][1], results[id][2]
        ))


def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args, set_id, device):
    
    result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # ecemeter = AverageMeter('ECE', ':3.2f', Summary.AVERAGE)
    # cecemeter = AverageMeter('CECE', ':3.2f', Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        model.reset()
    
    # if True: # 'ctpt' in args.run_type:
    #     model.l2_norm_cal = True
    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)

        if args.tpt: images = torch.cat(images, dim=0)

        global TARGET
        TARGET = target
        
        # reset the tunable prompt to its initial state
        if args.tta_steps > 0:
            with torch.no_grad():
                model.reset()
        # reset the optimizer state
        optimizer.load_state_dict(optim_state)

        # Test Time Adaptation
        noise = test_time_tuning(model, images, optimizer, scaler, args, set_id, i)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                    # output = model(normalize(image))[0]
                    # output = model(normalize(torch.clamp(image+noise,0,1)))[0]
                    output = forward_zero(images, model, reward_model=None, noise=noise) # <<zero>>-based inference

        #maximum confidence of the softmax_output and its index
        max_confidence, max_index = torch.max(output.softmax(1), 1)

        #save the max confidence, prediction, and label to the result_dict
        result_dict['max_confidence'].append(max_confidence.item())
        result_dict['prediction'].append(max_index.item())
        result_dict['label'].append(target.item())
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # ece = ece_metric(output, target).item()
        # cece = cece_metric(output, target).item()

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        # ecemeter.update(ece)
        # cecemeter.update(cece)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)
            
        # samples = 2000  
        # if i >= samples:
        #     progress.display_summary()
        #     acc, ece = Calculator(result_dict)
        #     return [round(x, 3) for x in [top1.avg.item(), top5.avg.item(), ece]]

    # torch.save(mean_dist_matrix, "mean_dist_matrix_imagenet_a.pt")
    
    # -------------------------------- T-SNE --------------------------------
        # if CLASSNAMES[TARGET.item()] in targeted_class_dict.values():
            # feat = model.image_encoder.vision_model.encoder.layers[-1].visual_feat[0] # for image
            # feat = model.image_encoder.transformer.resblocks[-1].visual_feat[0] # for prompt
            # clip_labels.append(TARGET.item())
            # clip_features.append(feat)
        
        # if i==(len(val_loader)-1):
        # # if i==(100):
        #     targeted_class_dict = {i: value for i, value in enumerate(CLASSNAMES)}
        #     all_clip_features = torch.stack(clip_features, dim=0).to('cpu').numpy()
        #     all_clip_labels = torch.tensor(clip_labels).to('cpu').numpy()
        #     plot_umap_features(all_clip_features, all_clip_labels, len(targeted_class_dict.keys()), targeted_class_dict, f"/home/raza.imam/Documents/RLCF/RLCF/plots/tnt_umap_{set_id}.png")
            # plot_features(all_clip_features, all_clip_labels, len(targeted_class_dict.keys()), targeted_class_dict, f"/home/raza.imam/Documents/RLCF/RLCF/plots/tnt_features_{set_id}_confident_augsx3.png")
        # -------------------------------- T-SNE --------------------------------
        

    progress.display_summary()
    acc, ece = Calculator(result_dict)
    return [round(x, 3) for x in [top1.avg.item(), top5.avg.item(), ece]]
    # return [top1.avg.item(), top5.avg.item(), ecemeter.avg, cecemeter.avg]

if __name__ == '__main__':
    args = get_args_rl()

    set_random_seed(args.seed)
    main_worker(args.gpu, args)
    os.system('export ghost="cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook"')