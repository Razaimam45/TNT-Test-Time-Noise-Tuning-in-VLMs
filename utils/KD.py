# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def kd_distill_loss(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean").mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd

def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student, only the gradient related part
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    # kl_div = -p log q
    loss_kd = - torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd

def dkl_loss(logits_student, logits_teacher, weight, alpha, beta):
    num_classes = logits_student.size(1)
    delta_n = logits_student.view(-1, num_classes, 1) - logits_student.view(-1, 1, num_classes)
    delta_a = logits_teacher.view(-1, num_classes, 1) - logits_teacher.view(-1, 1, num_classes)
    
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_student.size(0)
    loss_sce = -(F.softmax(logits_student, dim=1).detach() * F.log_softmax(logits_teacher, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce 


# Raza coded all the below Classes and Functions

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss

class KDClipLoss(nn.Module):
    def __init__(self, reward_model,
                 model):
        super(KDClipLoss, self).__init__()

        self.t_embed_dim = reward_model.embed_dim
        self.s_embed_dim = model.embed_dim

        if self.t_embed_dim != self.s_embed_dim:
            self.visual_proj = nn.Linear(self.t_embed_dim, self.s_embed_dim).to(reward_model.device)
            self.text_proj = nn.Linear(self.t_embed_dim, self.s_embed_dim).to(reward_model.device)

        self.kl_loss = DistillKL(T=1.5)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale = reward_model.clip_model.logit_scale.exp()
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features_teacher, text_features_teacher,
            image_features_student, text_features_student, kd_type="FD_CLIP"):
        """
        Feature distillation, L2 distance between teacher and student
        """
        # visual_proj = nn.Linear(768, 512).to(image_features_student.device, dtype=image_features_student.dtype)
        # text_proj = nn.Linear(768, 512).to(image_features_student.device, dtype=image_features_student.dtype)
        std_dtype = image_features_student.dtype
        logit_scale = self.logit_scale.exp()
        
        with torch.no_grad():
            image_features_teacher = self.visual_proj(image_features_teacher).to(dtype=std_dtype)
            text_features_teacher = self.text_proj(text_features_teacher).to(dtype=std_dtype)
        image_features_student = F.normalize(image_features_student, dim=1)
        text_features_student = F.normalize(text_features_student, dim=1)
        image_features_teacher = F.normalize(image_features_teacher, dim=1)
        text_features_teacher = F.normalize(text_features_teacher, dim=1)
        
        if kd_type == "FD_CLIP":
            fd_loss = F.mse_loss(image_features_student, image_features_teacher) +\
                F.mse_loss(text_features_student, text_features_teacher)
            return fd_loss
        
        elif kd_type == "CKD_CLIP":
            logits_per_image = logit_scale * image_features_student @ text_features_student.T
            logits_per_text = logit_scale * text_features_student @ image_features_student.T
            t_logits_per_image = logit_scale * image_features_teacher @ text_features_teacher.T
            t_logits_per_text = t_logits_per_image.T
            ckd_loss = (self.kl_loss(logits_per_image, t_logits_per_image.detach()) +\
                        self.kl_loss(logits_per_text, t_logits_per_text.detach())) / 2
            return ckd_loss
        
        elif kd_type == "crossKD_CLIP":
            logits_per_s_image_to_t_text = self.cross_logit_scale * image_features_student @ text_features_teacher.T
            logits_per_s_text_to_t_image = self.cross_logit_scale * text_features_student @ image_features_teacher.T
            t_logits_per_image = logit_scale * image_features_teacher @ text_features_teacher.T
            t_logits_per_text = t_logits_per_image.T
            cross_kd_loss = (self.kl_loss(logits_per_s_image_to_t_text, t_logits_per_image.detach()) +\
                             self.kl_loss(logits_per_s_text_to_t_image, t_logits_per_text.detach())) / 2
            return cross_kd_loss
        
        elif kd_type == "FM_CLIP":
        # def feature_matching_loss(image_features_teacher, text_features_teacher, image_features_student, text_features_student):
            # Compute L2 loss between teacher and student image features
            image_loss = F.mse_loss(image_features_student, image_features_teacher)
            # Compute L2 loss between teacher and student text features
            text_loss = F.mse_loss(text_features_student, text_features_teacher)
            # Total loss
            total_loss = image_loss + text_loss
            return total_loss
        
        elif kd_type == "KL_CLIP":
        # def kl_divergence_loss(image_features_teacher, text_features_teacher, image_features_student, text_features_student, temperature=1.0):
            # Apply softmax with temperature scaling to the features
            temperature = 1.0
            teacher_image_probs = F.softmax(image_features_teacher / temperature, dim=-1)
            student_image_probs = F.log_softmax(image_features_student / temperature, dim=-1)
            teacher_text_probs = F.softmax(text_features_teacher / temperature, dim=-1)
            student_text_probs = F.log_softmax(text_features_student / temperature, dim=-1)

            # Compute KL Divergence loss
            image_loss = F.kl_div(student_image_probs, teacher_image_probs, reduction='batchmean') * (temperature ** 2)
            text_loss = F.kl_div(student_text_probs, teacher_text_probs, reduction='batchmean') * (temperature ** 2)
            # Total loss
            total_loss = image_loss + text_loss
            return total_loss
        
        elif kd_type == "AT_CLIP":
        # def attention_transfer_loss(image_features_teacher, text_features_teacher, image_features_student, text_features_student):
            # Compute attention maps as the sum of squares of the features along the channel dimension
            def compute_attention(features):
                return torch.sum(features ** 2, dim=1)

            # Compute attention maps
            attention_teacher_image = compute_attention(image_features_teacher)
            attention_student_image = compute_attention(image_features_student)
            attention_teacher_text = compute_attention(text_features_teacher)
            attention_student_text = compute_attention(text_features_student)

            # Compute L2 loss between teacher and student attention maps
            image_loss = F.mse_loss(attention_student_image, attention_teacher_image)
            text_loss = F.mse_loss(attention_student_text, attention_teacher_text)
            # Total loss
            total_loss = image_loss + text_loss
            return total_loss

        elif kd_type == "MI_CLIP":
        # def mutual_information_loss(image_features_teacher, text_features_teacher, image_features_student, text_features_student):
            # Normalize the features
            image_features_teacher = F.normalize(image_features_teacher, p=2, dim=-1)
            image_features_student = F.normalize(image_features_student, p=2, dim=-1)
            text_features_teacher = F.normalize(text_features_teacher, p=2, dim=-1)
            text_features_student = F.normalize(text_features_student, p=2, dim=-1)
            
            # Compute cosine similarity
            def compute_similarity(f1, f2):
                return torch.sum(f1 * f2, dim=-1)

            # Compute mutual information as negative similarity
            image_loss = -torch.mean(compute_similarity(image_features_teacher, image_features_student))
            text_loss = -torch.mean(compute_similarity(text_features_teacher, text_features_student))
            # Total loss
            total_loss = image_loss + text_loss
            return total_loss
        
        elif kd_type == "CON_CLIP":
        # def contrastive_loss(image_features_teacher, text_features_teacher, image_features_student, text_features_student, margin=0.2):
            margin = 0.2
            # Normalize features
            image_features_teacher = F.normalize(image_features_teacher, p=2, dim=-1)
            image_features_student = F.normalize(image_features_student, p=2, dim=-1)
            text_features_teacher = F.normalize(text_features_teacher, p=2, dim=-1)
            text_features_student = F.normalize(text_features_student, p=2, dim=-1)

            # Calculate all pairwise cosine similarities between image and text features
            pos_sim = torch.matmul(image_features_teacher, text_features_teacher.T)  # Shape: (6, 200)
            neg_sim_image = torch.matmul(image_features_student, text_features_teacher.T)  # Shape: (6, 200)
            neg_sim_text = torch.matmul(image_features_teacher, text_features_student.T)  # Shape: (6, 200)
            
            # Get the maximum similarity for positive pairs
            pos_sim_max, _ = pos_sim.max(dim=1)  # Shape: (6,)
            
            # Contrastive loss computation
            image_loss = F.relu(margin - pos_sim_max.unsqueeze(1) + neg_sim_image).mean()
            text_loss = F.relu(margin - pos_sim_max.unsqueeze(1) + neg_sim_text).mean()
            
            # Total loss
            total_loss = image_loss + text_loss
            return total_loss

