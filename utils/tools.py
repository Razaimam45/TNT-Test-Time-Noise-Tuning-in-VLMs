import os
import time
import random
# from umap import UMAP

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt 
import math

plt.rcParams.update({'font.size': 20})
plt.style.use('ggplot')

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device is not 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg

def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            # bin_dict[binn][BIN_ACC] = 0
            # bin_dict[binn][BIN_CONF] = 0
            ## for getting a 45d line.
            bin_dict[binn][BIN_ACC] = binn / num_bins
            bin_dict[binn][BIN_CONF] = binn / num_bins

        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def reliability_curve_save(confs, preds, labels, metric, savepath, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.plot(bns, bns, color='pink', label='Expected',linewidth=5)
    plt.plot(bns, y, color='blue', label='Actual',linewidth=5)
    plt.text(0.9, 0.1, metric, size=50, ha="right", va="bottom",bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    plt.ylabel('Accuracy',fontsize=32)
    plt.xlabel('Confidence',fontsize=32)
    plt.legend(fontsize=32,loc='upper left')
    plt.savefig(savepath,bbox_inches='tight')
    # plt.show()
    
def countParams(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: ", total_params)
    print("Trainable Params: ", trainable_params)
    print("Trainable %: ", (trainable_params/total_params)*100)

import matplotlib.pyplot as plt
def plot_img(image, save_path='saved_plot.png', target=None, predicted=None):
    if type(image) == torch.Tensor:
        image_array = image.to('cpu').squeeze().permute(1, 2, 0).detach().numpy()
    else:
        image_array = image
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    plt.figure(figsize=(3, 3), tight_layout=True)
    plt.imshow(image_array)
    # title = f'Target: {target}, Pred: {predicted}'
    plt.axis('off')
    # plt.title(title, fontsize=10)
    plt.savefig(save_path)
    plt.close()

from torchvision.transforms import ToPILImage
to_pil =  ToPILImage()
def plot_pil_img(image, save_path='saved_plot.png'):
    img_noi  = to_pil(image)
    img_noi.save(save_path)
    
# T_SNE plot
from sklearn.manifold import TSNE
def plot_features(features, labels, num_classes, targeted_class_dict, save_path="plots/t_sne"):
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=5.0)
    features = tsne.fit_transform(features)
    
    # colors = ['C{}'.format(i) for i in range(num_classes)]
    # plt.figure(figsize=(3, 3), dpi=350)
    # for idx, label_idx in enumerate(list(targeted_class_dict.keys())):
    #     plt.scatter(
    #         features[labels.flatten()==label_idx, 0],
    #         features[labels.flatten()==label_idx, 1],
    #         c=colors[idx],
    #         s=15,
    #     )

    # plt.grid(True, which='major', color='gray', linestyle='-', alpha=0.4)
    # plt.grid(True, which='minor', color='gray', linestyle='--', alpha=0.1)
    # plt.minorticks_on()
    # plt.xticks([])
    # plt.yticks([])
    # class_names = list(targeted_class_dict.values())
    # # plt.legend(ncol=2, labelspacing=0.1, prop={'size': 10, 'weight': 'bold'}, bbox_to_anchor=(1.05, 1), loc='upper left')
    # # plt.title(f"CLIP Vision Feature Space", weight='bold', size=16)
    # plt.tight_layout()
    # plt.show()
    # # dirname = os.path.join(save_dir)
    # # if not os.path.exists(dirname):
    # #     os.makedirs(dirname)
    # plt.savefig(f'{save_path}.pdf', dpi=350, facecolor='auto', edgecolor='auto',
    #         orientation='portrait', format="pdf",
    #         transparent=False, bbox_inches='tight', pad_inches=0.05)
    # plt.close()
    
# Set up a colormap and transparency
    colormap = plt.cm.get_cmap('Spectral', num_classes)  # 'Spectral' for a smooth gradient
    alpha = 0.7  # Transparency level
    
    # Set up the plot with a dark background
    plt.figure(figsize=(5, 5), dpi=350)
    plt.style.use('dark_background')  # Dark background for contrast
    
    for idx, label_idx in enumerate(list(targeted_class_dict.keys())):
        plt.scatter(
            features[labels.flatten() == label_idx, 0],
            features[labels.flatten() == label_idx, 1],
            c=[colormap(idx)],  # Unique color for each class
            s=20,
            alpha=alpha,  # Set transparency
        )

    # Aesthetic grid and ticks
    plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)  # Subtle grid lines
    plt.xticks([])  # Remove x-ticks for minimalism
    plt.yticks([])  # Remove y-ticks for minimalism
    
    # Add title with bold and increased font size
    plt.title("CLIP Vision Feature Space", fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Tight layout and display
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig(f'{save_path}.pdf', dpi=350, format="pdf",
                transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    

# def plot_umap_features(features, labels, num_classes, targeted_class_dict, save_path="plots/umap"):
    
#     # Initialize UMAP with 2 components for 2D visualization
#     umap = UMAP(n_components=2, random_state=0, n_neighbors=5, min_dist=0.3)
#     features = umap.fit_transform(features)
    
#     # Set up a colormap and transparency
#     colormap = plt.cm.get_cmap('Spectral', num_classes)  # 'Spectral' for a smooth gradient
#     alpha = 0.7  # Transparency level
    
#     # Set up the plot with a dark background
#     plt.figure(figsize=(5, 5), dpi=350)
#     plt.style.use('dark_background')  # Dark background for contrast
    
#     for idx, label_idx in enumerate(list(targeted_class_dict.keys())):
#         plt.scatter(
#             features[labels.flatten() == label_idx, 0],
#             features[labels.flatten() == label_idx, 1],
#             c=[colormap(idx)],  # Unique color for each class
#             s=20,
#             alpha=alpha,  # Set transparency
#         )

#     # Aesthetic grid and ticks
#     plt.grid(True, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)  # Subtle grid lines
#     plt.xticks([])  # Remove x-ticks for minimalism
#     plt.yticks([])  # Remove y-ticks for minimalism
    
#     # Add title with bold and increased font size
#     plt.title("CLIP Vision Feature Space (UMAP)", fontsize=18, fontweight='bold', color='white', pad=20)
    
#     # Tight layout and display
#     plt.tight_layout()
#     plt.show()
    
#     # Save the plot
#     plt.savefig(f'{save_path}.pdf', dpi=350, format="pdf",
#                 transparent=True, bbox_inches='tight', pad_inches=0.05)
#     plt.close()



# class color:
#    PURPLE = '\033[95m'
#    CYAN = '\033[96m'
#    DARKCYAN = '\033[36m'
#    BLUE = '\033[94m'
#    GREEN = '\033[92m'
#    YELLOW = '\033[93m'
#    RED = '\033[91m'
#    BOLD = '\033[1m'
#    UNDERLINE = '\033[4m'
#    END = '\033[0m'


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.insert(0, '/home/raza.imam/Documents/RLCF/RLCF/TPT/Transformer-MM-Explainability')
import CLIP.clip as clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from captum.attr import visualization
import os

class CLIPInterpreter:
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu", start_layer=-1, start_layer_text=-1, cls_idx=0):
        self.device = device
        self.start_layer = start_layer
        self.start_layer_text = start_layer_text
        self._tokenizer = _Tokenizer()
        # self.cls = cls_idx
        model, preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        self.model = model.to(self.device)
        self.preprocess = preprocess

    def interpret(self, image, texts, model, start_layer=-1, start_layer_text=-1):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, _ = model(images, texts) # modified for my model
        # logits_per_image = logits_per_image[:,self.cls].unsqueeze(0) # For selecting actual class
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if start_layer == -1: 
            start_layer = len(image_attn_blocks) - 1
        
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]
        
        # text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
        # if start_layer_text == -1: 
        #     start_layer_text = len(text_attn_blocks) - 1

        # num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        # R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(self.device)
        # R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        # for i, blk in enumerate(text_attn_blocks):
        #     if i < start_layer_text:
        #         continue
        #     grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        #     cam = blk.attn_probs.detach()
        #     cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        #     grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        #     cam = grad * cam
        #     cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        #     cam = cam.clamp(min=0).mean(dim=1)
        #     R_text = R_text + torch.bmm(cam, R_text)
        # text_relevance = R_text
    
        return None, image_relevance

    def show_image_relevance(self, image_relevance, image, save_path):
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        # Prepare the relevance map
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).to(self.device).data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        
        # Normalize the image and overlay heatmap
        image = image[0].permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        # Save the `vis` plot without any boundary or axis
        vis_image = Image.fromarray(vis)
        vis_image.save(save_path)
        return vis


    def show_heatmap_on_text(self, text, text_encoding, R_text):
        CLS_idx = text_encoding.argmax(dim=-1)
        R_text = R_text[CLS_idx, 1:CLS_idx]
        text_scores = R_text / R_text.sum()
        text_scores = text_scores.flatten()
        print(text_scores)
        text_tokens = self._tokenizer.encode(text)
        text_tokens_decoded = [self._tokenizer.decode([a]) for a in text_tokens]
        vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
        visualization.visualize_text(vis_data_records)

    def forward(self, texts, image_tensor, save_dir, file_name='attn'):
        os.makedirs(save_dir, exist_ok=True)

        text_tensor = clip.tokenize(texts).to(self.device)
        R_text, R_image = self.interpret(image=image_tensor, texts=text_tensor, model=self.model)

        # logits_per_image, logits_per_text = model(image_tensor, text_tensor)

        batch_size = text_tensor.shape[0]
        for i in range(batch_size):
            # text_save_path = os.path.join(save_dir, f"text_heatmap.png")
            image_save_path = os.path.join(save_dir, f"{file_name}.png")
            
            # self.show_heatmap_on_text(texts[i], text_tensor[i], R_text[i])
            return self.show_image_relevance(R_image[i], image_tensor, save_path=image_save_path)


# __main__
# import torchvision
# device = "cuda:2" if torch.cuda.is_available() else "cpu"

# interpreter = CLIPInterpreter(device=device, start_layer=-1, start_layer_text=-1)
# img_path = "/home/raza.imam/Documents/RLCF/RLCF/TPT/Transformer-MM-Explainability/CLIP/0_flower_noise.png"
# img = interpreter.preprocess(Image.open(img_path)).unsqueeze(0).to(device)
# texts = "a sunflower"
# interpreter.forward(texts, img, save_dir="/home/raza.imam/Documents/RLCF/RLCF/plots/attentions/class1/")