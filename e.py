import os
import copy
import math
import random
import threading
import queue
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageTk, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================================
# CONFIGURATION
# ============================================================================
OBJ_DIR = "object_ids"
IMG_SIZE = 256
MAX_OBJECTS_FINAL = 12
BATCH_SIZE = 16
NUM_EPOCHS = 200
LR = 3e-4
BACKBONE_LR_MULT = 0.1     # Backbone fine-tuning multiplier
WARMUP_STEPS = 300
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "scene_ai_checkpoint.pth"
BEST_CHECKPOINT_FILE = "scene_ai_best.pth"
HISTORY_FILE = "training_history.json"
NUM_WORKERS = min(4, os.cpu_count() or 2)
USE_AMP = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Curriculum Configuration
STREAK_NEEDED = 10
VALIDATION_INTERVAL = 1
SUCCESS_MSE_THRESHOLD = 0.01
VISUAL_UPDATE_INTERVAL = 1

STALL_RELAX_ATTEMPTS = 150
STALL_RELAX_FACTOR = 1.15
STALL_RELAX_MAX = 0.05

# Decoder & Training Knobs
NUM_DECODER_LAYERS = 4
EMA_DECAY = 0.999
VALIDATION_SAMPLES = 5
CURRICULUM_BAND = 4
LABEL_SMOOTHING = 0.05

# Denoising Training (DN-DETR)
USE_DENOISING_TRAINING = True
DN_POS_NOISE_SCALE = 0.15
DN_CLASS_NOISE_PROB = 0.2
W_DN = 1.0

# Loss Knobs
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

W_ID = 3.0
W_ATTR = 1.5
W_MATCH_PRES = 20.0
W_NO_DETECT_PENALTY = 15.0
W_CARDINALITY = 20.0

# Attribute L1 weights: [x, y, sin, cos, scale, r, g, b, layer]
W_ATTR_POS = 2.0
W_ATTR_ROT = 1.0
W_ATTR_SCALE = 1.0
W_ATTR_COLOR = 2.5
W_ATTR_LAYER = 1.0


def _attr_weight_vector(device, dtype):
    return torch.tensor(
        [W_ATTR_POS, W_ATTR_POS, W_ATTR_ROT, W_ATTR_ROT, W_ATTR_SCALE,
         W_ATTR_COLOR, W_ATTR_COLOR, W_ATTR_COLOR, W_ATTR_LAYER],
        device=device, dtype=dtype
    )


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """Sigmoid focal loss (Lin et al., RetinaNet) for query presence detection."""
    p = torch.sigmoid(inputs).clamp(1e-6, 1.0 - 1e-6)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def sine_embed_from_anchor(anchor, d_model, temperature=10000.0):
    """Generates dynamic 2D sinusoidal positional embeddings from (x, y) anchors."""
    scale = 2 * math.pi
    dim_half = d_model // 2
    dim_t = torch.arange(dim_half, dtype=torch.float32, device=anchor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / dim_half)

    x_embed = anchor[..., 0] * scale
    y_embed = anchor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([pos_y, pos_x], dim=-1).to(anchor.dtype)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen non-trainable statistics for stable fine-tuning."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def freeze_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.weight.data.copy_(child.weight.data)
            frozen.bias.data.copy_(child.bias.data)
            frozen.running_mean.data.copy_(child.running_mean.data)
            frozen.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)

# ============================================================================
# CURRICULUM MANAGER
# ============================================================================
class CurriculumManager:
    def __init__(self, max_objects_target, streak_needed=10, mse_threshold=0.015):
        self.current_max = 1
        self.max_target = max_objects_target
        self.streak_needed = streak_needed
        self.base_mse_threshold = mse_threshold
        self.effective_mse_threshold = self._level_mse_threshold(self.current_max)
        self.streak = 0
        self.total_validations = 0
        self.attempts_at_level = 0
        self.promotions = []
        self.level_stats = {}
        self.recent_mse = []

    def _level_mse_threshold(self, level):
        return self.base_mse_threshold * (1.0 + 0.05 * (level - 1))

    def _count_tolerance(self, level):
        if level <= 3:
            return 0
        return max(1, round(0.10 * level))

    @property
    def level_name(self):
        return f"1–{self.current_max} obj"

    @property
    def is_complete(self):
        return self.current_max >= self.max_target

    def record_result(self, mse_val, detected_count, gt_count):
        self.total_validations += 1
        self.attempts_at_level += 1
        self.recent_mse.append(mse_val)
        if len(self.recent_mse) > 50:
            self.recent_mse.pop(0)

        count_ok = abs(detected_count - gt_count) <= self._count_tolerance(self.current_max)
        mse_ok = mse_val < self.effective_mse_threshold
        success = count_ok and mse_ok

        if self.current_max not in self.level_stats:
            self.level_stats[self.current_max] = {"attempts": 0, "successes": 0}
        self.level_stats[self.current_max]["attempts"] += 1

        relaxed = False
        if success:
            self.streak += 1
            self.level_stats[self.current_max]["successes"] += 1
            if self.streak >= self.streak_needed and not self.is_complete:
                self._promote()
                return True, True, False
        else:
            self.streak = max(0, self.streak - 2)
            if self.attempts_at_level >= STALL_RELAX_ATTEMPTS and self.effective_mse_threshold < STALL_RELAX_MAX:
                self.effective_mse_threshold = min(STALL_RELAX_MAX, self.effective_mse_threshold * STALL_RELAX_FACTOR)
                self.attempts_at_level = 0
                relaxed = True

        return False, success, relaxed

    def _promote(self):
        old = self.current_max
        self.current_max = min(self.current_max + 1, self.max_target)
        self.streak = 0
        self.attempts_at_level = 0
        self.effective_mse_threshold = self._level_mse_threshold(self.current_max)
        self.promotions.append({"from": old, "to": self.current_max, "step": self.total_validations})

    def get_summary(self):
        return {
            "level": self.current_max,
            "level_name": self.level_name,
            "streak": self.streak,
            "streak_needed": self.streak_needed,
            "is_complete": self.is_complete,
            "promotions": len(self.promotions),
            "avg_mse": float(np.mean(self.recent_mse)) if self.recent_mse else 0.0,
            "effective_mse_threshold": self.effective_mse_threshold,
            "attempts_at_level": self.attempts_at_level,
        }

# ============================================================================
# DATASET
# ============================================================================
class SpriteSceneDataset(Dataset):
    def __init__(self, obj_dir, img_size=256, max_objects=12, num_samples=10000):
        self.img_size = img_size
        self.max_objects = max_objects
        self.num_samples = num_samples
        self.current_difficulty = 1
        self.sprites = []
        self.sprite_names = []
        for f in os.listdir(obj_dir):
            if os.path.splitext(f)[1].lower() == ".png":
                img = Image.open(os.path.join(obj_dir, f)).convert("RGBA")
                self.sprites.append(img)
                self.sprite_names.append(os.path.splitext(f)[0])
        self.num_sprites = len(self.sprites)

    def set_difficulty(self, max_obj):
        self.current_difficulty = min(max_obj, self.max_objects)

    def __len__(self):
        return self.num_samples

    def render_sprite(self, canvas, sprite, x, y, scale, angle, color_shift):
        """Optimized CPU sprite renderer."""
        w, h = sprite.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        spr = sprite.resize((new_w, new_h), Image.BILINEAR) if (new_w != w or new_h != h) else sprite
        if abs(angle) > 1e-2:
            spr = spr.rotate(angle, resample=Image.BICUBIC, expand=True)

        if np.any(np.abs(color_shift) > 0.5):
            arr = np.array(spr, dtype=np.int16)
            arr[:, :, :3] += color_shift.astype(np.int16)
            np.clip(arr, 0, 255, out=arr)
            spr = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

        rw, rh = spr.size
        paste_x = int(x - rw / 2)
        paste_y = int(y - rh / 2)
        canvas.paste(spr, (paste_x, paste_y), spr)

    def _build_scene(self, n):
        canvas = Image.new("RGBA", (self.img_size, self.img_size), (128, 128, 128, 255))
        indices = random.choices(range(self.num_sprites), k=n)
        objects = []
        for i in range(n):
            obj_id = indices[i]
            x = random.uniform(0.0, 1.0) * self.img_size
            y = random.uniform(0.0, 1.0) * self.img_size
            scale = random.uniform(0.3, 2.5)
            angle = random.uniform(0, 360)
            color_shift = np.random.uniform(-30, 30, size=3)
            layer = random.uniform(0, 1)
            objects.append({
                "id": obj_id, "x": x, "y": y, "scale": scale,
                "angle": angle, "color_shift": color_shift, "layer": layer
            })
        objects.sort(key=lambda o: o["layer"])
        for obj in objects:
            self.render_sprite(canvas, self.sprites[obj["id"]], obj["x"], obj["y"],
                               obj["scale"], obj["angle"], obj["color_shift"])

        padded_targets = []
        for i in range(self.max_objects):
            if i < n:
                o = objects[i]
                rad = math.radians(o["angle"])
                padded_targets.append([
                    1.0, o["id"],
                    o["x"] / self.img_size, o["y"] / self.img_size,
                    math.sin(rad), math.cos(rad),
                    o["scale"] / 3.0,
                    o["color_shift"][0] / 30.0,
                    o["color_shift"][1] / 30.0,
                    o["color_shift"][2] / 30.0,
                    o["layer"]
                ])
            else:
                padded_targets.append([0.0, 0, 0.5, 0.5, 0, 1, 0.5, 0, 0, 0, 0.5])
        target_tensor = torch.tensor(padded_targets, dtype=torch.float32)
        img_rgb = canvas.convert("RGB")
        img_tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0
        return img_tensor, target_tensor

    def __getitem__(self, idx):
        if random.random() < 0.80:
            n = self.current_difficulty
        else:
            low = max(1, self.current_difficulty - CURRICULUM_BAND)
            n = random.randint(low, self.current_difficulty)
        return self._build_scene(n)

    def generate_fixed_n(self, n=None):
        if n is None:
            n = self.current_difficulty
        n = max(1, min(n, self.max_objects))
        return self._build_scene(n)

# ============================================================================
# ARCHITECTURE MODULES
# ============================================================================
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, h=16, w=16):
        super().__init__()
        pe = torch.zeros(d_model, h, w)
        d_model_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half)
        )
        pos_x = torch.arange(w).float()
        enc_x = pos_x.unsqueeze(0) * div_term.unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(enc_x).unsqueeze(1).expand(-1, h, -1)
        pe[1:d_model_half:2, :, :] = torch.cos(enc_x).unsqueeze(1).expand(-1, h, -1)

        pos_y = torch.arange(h).float()
        enc_y = pos_y.unsqueeze(0) * div_term.unsqueeze(1)
        pe[d_model_half::2, :, :] = torch.sin(enc_y).unsqueeze(2).expand(-1, -1, w)
        pe[d_model_half + 1::2, :, :] = torch.cos(enc_y).unsqueeze(2).expand(-1, -1, w)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class FeaturePyramidFusion(nn.Module):
    """Multi-scale Feature Aggregation across backbone stages into 16x16 grid."""
    def __init__(self, d_model=512):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        self.conv_l3 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        self.conv_l4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(256 * 3, d_model, 3, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_stem, feat_l3, feat_l4):
        p_stem = F.adaptive_avg_pool2d(self.conv_stem(feat_stem), (16, 16))
        p_l3 = self.conv_l3(feat_l3)
        p_l4 = F.interpolate(self.conv_l4(feat_l4), size=(16, 16), mode='bilinear', align_corners=False)
        fused = torch.cat([p_stem, p_l3, p_l4], dim=1)
        return self.out_conv(fused)


class DABDecoderLayer(nn.Module):
    """Decoupled Content + Positional Transformer Decoder Layer (DAB-DETR Style)."""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos, memory_pos=None, attn_mask=None):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        q = tgt + query_pos
        k = memory + memory_pos if memory_pos is not None else memory
        tgt2 = self.cross_attn(query=q, key=k, value=memory)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt

# ============================================================================
# MODEL
# ============================================================================
class ScenePredictor(nn.Module):
    def __init__(self, num_sprites, max_objects):
        super().__init__()
        self.max_objects = max_objects
        self.num_sprites = num_sprites
        self.d_model = 512

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        freeze_batchnorm(resnet)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fuse = FeaturePyramidFusion(self.d_model)
        self.feat_grid = 16
        self.pos_enc = PositionalEncoding2D(self.d_model, self.feat_grid, self.feat_grid)

        self.matching_content = nn.Embedding(max_objects, self.d_model)
        self.matching_anchor = nn.Embedding(max_objects, 2)

        self.decoder_layers = nn.ModuleList([
            DABDecoderLayer(self.d_model, nhead=8, dim_feedforward=1024, dropout=0.1)
            for _ in range(NUM_DECODER_LAYERS)
        ])

        anchor_refine_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU(),
            nn.Linear(self.d_model, 2)
        )
        self.anchor_refine_heads = nn.ModuleList([
            copy.deepcopy(anchor_refine_head) for _ in range(NUM_DECODER_LAYERS)
        ])

        self.dn_id_embed = nn.Embedding(num_sprites + 1, self.d_model)

        self.pres_head = nn.Linear(self.d_model, 1)
        self.id_head = nn.Linear(self.d_model, num_sprites)
        self.attr_head = nn.Linear(self.d_model, 7)

        self._dn_mask_cache = {}
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.pres_head.bias, -1.0)
        nn.init.zeros_(self.attr_head.weight)
        nn.init.constant_(self.attr_head.bias, 0.0)
        with torch.no_grad():
            self.attr_head.bias[1] = 2.0
        nn.init.uniform_(self.matching_anchor.weight, -2.0, 2.0)
        for head in self.anchor_refine_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _heads(self, o, anchor_xy):
        pres = self.pres_head(o)
        ids = self.id_head(o)
        attrs = self.attr_head(o)

        attr_sin_cos = torch.tanh(attrs[:, :, 0:2])
        attr_sin_cos = attr_sin_cos / attr_sin_cos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        attr_scale = torch.sigmoid(attrs[:, :, 2:3])
        attr_cs = torch.tanh(attrs[:, :, 3:6])
        attr_layer = torch.sigmoid(attrs[:, :, 6:7])
        attrs_norm = torch.cat([anchor_xy, attr_sin_cos, attr_scale, attr_cs, attr_layer], dim=-1)

        return torch.cat([pres, ids, attrs_norm], dim=-1)

    def _dn_self_attn_mask(self, M, device):
        key = (M, device)
        if key not in self._dn_mask_cache:
            total = 2 * M
            mask = torch.zeros(total, total, device=device)
            mask[:M, M:] = float('-inf')
            mask[M:, :M] = float('-inf')
            self._dn_mask_cache[key] = mask
        return self._dn_mask_cache[key]

    def _build_dn_queries(self, targets):
        t_pres = targets[:, :, 0]
        t_id = targets[:, :, 1].long()
        t_xy = targets[:, :, 2:4]
        is_valid = t_pres > 0.5

        jitter = (torch.rand_like(t_xy) * 2 - 1) * DN_POS_NOISE_SCALE
        dn_anchor = (t_xy + jitter).clamp(0.0, 1.0)

        class_noise = torch.rand(t_id.shape, device=t_id.device) < DN_CLASS_NOISE_PROB
        rand_ids = torch.randint(0, self.num_sprites, t_id.shape, device=t_id.device)
        dn_ids = torch.where(class_noise, rand_ids, t_id)
        pad_id = self.num_sprites
        dn_ids = torch.where(is_valid, dn_ids, torch.full_like(dn_ids, pad_id))
        dn_content = self.dn_id_embed(dn_ids)

        return dn_content, dn_anchor

    def forward(self, x, targets=None, return_all_layers=False):
        B = x.size(0)
        feat_mid = self.stem(x)
        feat_l3 = self.layer3(feat_mid)
        feat_l4 = self.layer4(feat_l3)

        feat_fused = self.fuse(feat_mid, feat_l3, feat_l4)
        memory = self.pos_enc(feat_fused).flatten(2).permute(0, 2, 1)

        content = self.matching_content.weight.unsqueeze(0).expand(B, -1, -1)
        anchor = torch.sigmoid(self.matching_anchor.weight).unsqueeze(0).expand(B, -1, -1)

        use_dn = self.training and targets is not None
        attn_mask = None
        if use_dn:
            dn_content, dn_anchor = self._build_dn_queries(targets)
            state = torch.cat([content, dn_content], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            attn_mask = self._dn_self_attn_mask(self.max_objects, x.device)
        else:
            state = content

        layer_outputs, anchors_per_layer = [], []
        for layer, refine_head in zip(self.decoder_layers, self.anchor_refine_heads):
            query_pos = sine_embed_from_anchor(anchor, self.d_model)
            state = layer(state, memory, query_pos=query_pos, attn_mask=attn_mask)
            delta = refine_head(state)
            anchor = torch.sigmoid(inverse_sigmoid(anchor) + delta)
            layer_outputs.append(state)
            anchors_per_layer.append(anchor)

        if return_all_layers:
            preds_all = [self._heads(o, a) for o, a in zip(layer_outputs, anchors_per_layer)]
        else:
            preds_all = self._heads(layer_outputs[-1], anchors_per_layer[-1])

        if use_dn:
            M = self.max_objects
            if return_all_layers:
                matching = [p[:, :M] for p in preds_all]
                dn = [p[:, M:] for p in preds_all]
            else:
                matching, dn = preds_all[:, :M], preds_all[:, M:]
            return matching, dn
        return preds_all

# ============================================================================
# EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

# ============================================================================
# LOSS
# ============================================================================
class HungarianLoss(nn.Module):
    def __init__(self, num_sprites):
        super().__init__()
        self.num_sprites = num_sprites

    def forward(self, pred, tgt):
        B, Q, _ = pred.shape
        device = pred.device
        eps = 1e-6

        p_pres = pred[:, :, 0:1]
        t_pres = tgt[:, :, 0:1]
        p_id = pred[:, :, 1:1 + self.num_sprites]
        t_id = tgt[:, :, 1].long()
        p_attrs = pred[:, :, 1 + self.num_sprites:]
        t_attrs = tgt[:, :, 2:]

        pres_logits = p_pres.squeeze(-1)
        pres_targets_raw = t_pres.squeeze(-1)

        num_gt_objects = pres_targets_raw.sum(dim=1)

        # --- 1. Fully Differentiable Soft Count ---
        # Smooth sum of probabilities across all queries (no boolean cutoffs, 100% differentiable)
        soft_count = torch.sigmoid(pres_logits).sum(dim=1)

        # --- 2. Smooth Cardinality Loss ---
        # Smoothly penalizes any discrepancy between soft_count and num_gt_objects
        loss_cardinality = F.smooth_l1_loss(soft_count, num_gt_objects, beta=0.5)

        # --- 3. Differentiable Asymmetric Under-Count Penalty ---
        # Activates ONLY when soft_count < num_gt_objects.
        # When soft_count >= num_gt_objects, this penalty evaluates to EXACTLY 0.0 (no false punishment!)
        under_count_diff = F.relu(num_gt_objects - soft_count)
        under_detect_penalty = (under_count_diff ** 2).mean() * W_NO_DETECT_PENALTY

        # Pairwise matching cost matrices for Hungarian assignment
        p_pres_exp = p_pres.expand(-1, -1, Q)
        t_pres_exp = t_pres.transpose(1, 2).expand(-1, Q, -1)
        c_pres = sigmoid_focal_loss(p_pres_exp, t_pres_exp, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='none')

        log_probs = F.log_softmax(p_id, dim=-1)
        idx = t_id.unsqueeze(1).unsqueeze(-1).expand(-1, Q, -1, 1)
        log_probs_exp = log_probs.unsqueeze(2).expand(-1, -1, Q, -1)
        c_id = -torch.gather(log_probs_exp, 3, idx).squeeze(-1)

        attr_weights = _attr_weight_vector(device, p_attrs.dtype)
        diff = torch.abs(p_attrs.unsqueeze(2) - t_attrs.unsqueeze(1))
        c_attrs = (diff * attr_weights).sum(dim=-1)

        valid_mask = (t_pres.squeeze(-1) == 1.0)
        invalid_cols = (~valid_mask).unsqueeze(1).expand(-1, Q, -1)
        c_id = c_id.masked_fill(invalid_cols, 0.0)
        c_attrs = c_attrs.masked_fill(invalid_cols, 0.0)

        c_total = W_MATCH_PRES * c_pres + W_ID * c_id + W_ATTR * c_attrs

        cost_np = c_total.detach().float().cpu().numpy()
        col_ind_batch = np.empty((B, Q), dtype=np.int64)
        for b in range(B):
            _, col_ind = linear_sum_assignment(cost_np[b])
            col_ind_batch[b] = col_ind
        col_ind_t = torch.from_numpy(col_ind_batch).to(device)

        matched_pres_targets = torch.gather(pres_targets_raw, 1, col_ind_t)
        loss_pres = sigmoid_focal_loss(pres_logits, matched_pres_targets, alpha=FOCAL_ALPHA,
                                        gamma=FOCAL_GAMMA, reduction='mean')

        matched_valid_mask = torch.gather(valid_mask.float(), 1, col_ind_t)
        valid_count = matched_valid_mask.sum(dim=1)

        matched_t_id = torch.gather(t_id, 1, col_ind_t)
        loss_id_all = F.cross_entropy(
            p_id.reshape(B * Q, self.num_sprites), matched_t_id.reshape(B * Q),
            reduction='none', label_smoothing=LABEL_SMOOTHING
        ).reshape(B, Q)
        loss_id = ((loss_id_all * matched_valid_mask).sum(dim=1) / (valid_count + eps)).mean()

        idx_attr = col_ind_t.unsqueeze(-1).expand(-1, -1, t_attrs.size(-1))
        matched_t_attrs = torch.gather(t_attrs, 1, idx_attr)
        per_dim_l1 = F.l1_loss(p_attrs, matched_t_attrs, reduction='none')
        loss_attrs_all = (per_dim_l1 * attr_weights).sum(dim=-1)
        loss_attrs = ((loss_attrs_all * matched_valid_mask).sum(dim=1) / (valid_count + eps)).mean()

        return (
            loss_pres
            + W_ID * loss_id
            + W_ATTR * loss_attrs
            + under_detect_penalty
            + W_CARDINALITY * loss_cardinality
        )

    def forward_dn(self, pred, tgt):
        """Denoising Branch Loss"""
        B, M, _ = pred.shape
        eps = 1e-6
        p_pres = pred[:, :, 0]
        t_pres = tgt[:, :, 0]
        p_id = pred[:, :, 1:1 + self.num_sprites]
        t_id = tgt[:, :, 1].long()
        p_attrs = pred[:, :, 1 + self.num_sprites:]
        t_attrs = tgt[:, :, 2:]

        loss_pres = sigmoid_focal_loss(p_pres, t_pres, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean')

        valid_mask = (t_pres > 0.5).float()
        valid_count = valid_mask.sum(dim=1)

        loss_id_all = F.cross_entropy(
            p_id.reshape(B * M, self.num_sprites), t_id.reshape(B * M),
            reduction='none', label_smoothing=LABEL_SMOOTHING
        ).reshape(B * M)
        loss_id = ((loss_id_all * valid_mask.reshape(-1)).sum() / (valid_count.sum() + eps))

        attr_weights = _attr_weight_vector(pred.device, p_attrs.dtype)
        per_dim_l1 = F.l1_loss(p_attrs, t_attrs, reduction='none')
        loss_attrs_all = (per_dim_l1 * attr_weights).sum(dim=-1)
        loss_attrs = ((loss_attrs_all * valid_mask).sum(dim=1) / (valid_count + eps)).mean()

        return loss_pres + W_ID * loss_id + W_ATTR * loss_attrs

# ============================================================================
# EVALUATION & RECONSTRUCTION
# ============================================================================
def compute_reconstruction_mse(model, dataset, device):
    model.eval()
    with torch.no_grad():
        img_tensor, tgt_tensor = dataset.generate_fixed_n()
        img_input = img_tensor.unsqueeze(0).to(device)
        pred = model(img_input)[0]

    gt_count = int(tgt_tensor[:, 0].sum().item())
    probs = torch.sigmoid(pred[:, 0])

    presence = probs > 0.5

    detected_count = int(presence.sum().item())

    predictions = []
    for i in range(dataset.max_objects):
        if presence[i]:
            obj_id = torch.argmax(pred[i, 1:1 + dataset.num_sprites]).item()
            attrs = pred[i, 1 + dataset.num_sprites:]
            layer_depth = attrs[8].item()
            predictions.append((layer_depth, obj_id, attrs))

    predictions.sort(key=lambda p: p[0])

    ai_canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (128, 128, 128, 255))
    for layer_depth, obj_id, attrs in predictions:
        x = attrs[0].item() * IMG_SIZE
        y = attrs[1].item() * IMG_SIZE
        sin_a, cos_a = attrs[2].item(), attrs[3].item()
        angle = math.degrees(math.atan2(sin_a, cos_a))
        scale = attrs[4].item() * 3.0
        cs = attrs[5:8].cpu().numpy() * 30.0
        dataset.render_sprite(ai_canvas, dataset.sprites[obj_id], x, y, scale, angle, cs)

    gt_arr = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)
    ai_arr = np.array(ai_canvas.convert("RGB")).astype(np.float32) / 255.0
    mse = float(np.mean((gt_arr - ai_arr) ** 2))

    return mse, detected_count, gt_count


def generate_visual_comparison(model, dataset, device):
    model.eval()
    with torch.no_grad():
        img_tensor, tgt_tensor = dataset.generate_fixed_n()
        img_input = img_tensor.unsqueeze(0).to(device)
        pred = model(img_input)[0]

    gt_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    num_gt = int(tgt_tensor[:, 0].sum().item())

    probs = torch.sigmoid(pred[:, 0])

    presence = probs > 0.5

    num_detected = int(presence.sum().item())

    predictions = []
    for i in range(dataset.max_objects):
        if presence[i]:
            obj_id = torch.argmax(pred[i, 1:1 + dataset.num_sprites]).item()
            attrs = pred[i, 1 + dataset.num_sprites:]
            layer_depth = attrs[8].item()
            predictions.append((layer_depth, obj_id, attrs))

    predictions.sort(key=lambda p: p[0])

    ai_canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (128, 128, 128, 255))
    for layer_depth, obj_id, attrs in predictions:
        x = attrs[0].item() * IMG_SIZE
        y = attrs[1].item() * IMG_SIZE
        angle = math.degrees(math.atan2(attrs[2].item(), attrs[3].item()))
        scale = attrs[4].item() * 3.0
        cs = attrs[5:8].cpu().numpy() * 30.0
        dataset.render_sprite(ai_canvas, dataset.sprites[obj_id], x, y, scale, angle, cs)

    ai_img = ai_canvas.convert("RGB")
    label_h = 28
    gap = 4
    comp = Image.new("RGB", (IMG_SIZE * 2 + gap, IMG_SIZE + label_h), (30, 30, 46))
    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((IMG_SIZE // 2 - 45, 5), "Ground Truth", fill=(166, 227, 161), font=font)
    draw.text((IMG_SIZE + gap + IMG_SIZE // 2 - 45, 5), "AI Prediction", fill=(137, 180, 250), font=font)
    comp.paste(gt_img, (0, label_h))
    comp.paste(ai_img, (IMG_SIZE + gap, label_h))
    return comp, {"gt_objects": num_gt, "detected_objects": num_detected}

# ============================================================================
# LR SCHEDULER
# ============================================================================
def get_lr_at_step(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# ============================================================================
# TRAINING ENGINE
# ============================================================================
class TrainingEngine:
    def __init__(self, msg_queue):
        self.msg_queue = msg_queue
        self.stop_flag = False
        self.pause_flag = False
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None
        self.dataloader = None
        self.curriculum = None
        self.start_epoch = 0
        self.global_step = 0
        self.current_epoch = 0
        self.history = {
            "epoch_losses": [],
            "batch_losses": [],
            "val_accuracy": [],      # Full history (non-resetting)
            "val_steps": [],         # Step corresponding to each val
            "lr_history": [],
            "level_changes": []
        }
        self.num_sprites = 0
        self.scaler = None
        self.ema = None
        self.best_mse = float("inf")

    def log(self, msg, level="INFO"):
        self.msg_queue.put(("log", f"[{level}] {msg}"))

    def update_status(self, epoch, batch, total_batches, loss, lr):
        self.msg_queue.put(("status", epoch, batch, total_batches, loss, lr))

    def update_loss_plot(self, batch_losses):
        self.msg_queue.put(("loss_plot", batch_losses))

    def update_acc_plot(self, accuracies, steps):
        self.msg_queue.put(("acc_plot", accuracies, steps))

    def update_visual(self, img, metrics):
        self.msg_queue.put(("visual", img, metrics))

    def update_curriculum(self, summary):
        self.msg_queue.put(("curriculum", summary))

    def setup(self):
        self.log(f"Device: {DEVICE}")
        self.dataset = SpriteSceneDataset(OBJ_DIR, IMG_SIZE, MAX_OBJECTS_FINAL, num_samples=20000)
        self.num_sprites = self.dataset.num_sprites
        self.log(f"Loaded {self.num_sprites} sprites")

        self.curriculum = CurriculumManager(MAX_OBJECTS_FINAL, STREAK_NEEDED, SUCCESS_MSE_THRESHOLD)
        self.dataset.set_difficulty(1)
        self.log(f"🎓 Curriculum: Start with 1 object (Target: {MAX_OBJECTS_FINAL})")

        self.dataloader = DataLoader(
            self.dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
        )
        self.log(f"DataLoader workers: {NUM_WORKERS}")
        self.model = ScenePredictor(self.num_sprites, MAX_OBJECTS_FINAL).to(DEVICE)
        self.criterion = HungarianLoss(self.num_sprites)
        self.scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
        self.ema = ModelEMA(self.model, decay=EMA_DECAY)

        backbone_params = (list(self.model.stem.parameters()) +
                           list(self.model.layer3.parameters()) +
                           list(self.model.layer4.parameters()))
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT, "name": "backbone"},
                {"params": other_params, "lr": LR, "name": "head"},
            ],
            weight_decay=WEIGHT_DECAY,
        )

        param_count = sum(p.numel() for p in self.model.parameters())
        self.log(f"Model params: {param_count:,}")

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_FILE):
            self.log("Loading checkpoint...")
            ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)

            missing, unexpected = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if missing or unexpected:
                self.log(f"⚠️ Architecture partial load ({len(missing)} missing, {len(unexpected)} unused)", "WARN")
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                self.log("⚠️ Optimizer state re-initialized.", "WARN")

            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.history = ckpt.get('history', self.history)
            if "val_accuracy" not in self.history:
                self.history["val_accuracy"] = []
            if "val_steps" not in self.history:
                self.history["val_steps"] = []

            self.best_mse = ckpt.get('best_mse', float('inf'))

            if 'curriculum_level' in ckpt:
                self.curriculum.current_max = ckpt['curriculum_level']
                self.curriculum.streak = ckpt.get('curriculum_streak', 0)
                self.curriculum.effective_mse_threshold = ckpt.get(
                    'curriculum_effective_mse_threshold', self.curriculum.base_mse_threshold
                )
                self.curriculum.attempts_at_level = ckpt.get('curriculum_attempts_at_level', 0)
                self.dataset.set_difficulty(self.curriculum.current_max)

            if 'ema_state_dict' in ckpt:
                self.ema.shadow.load_state_dict(ckpt['ema_state_dict'], strict=False)
                self.ema.updates = ckpt.get('ema_updates', 0)
            else:
                self.ema.shadow.load_state_dict(self.model.state_dict())

            # Push loaded accuracy history to plot
            if self.history["val_accuracy"]:
                self.update_acc_plot(self.history["val_accuracy"], self.history["val_steps"])

            self.log(f"Resumed: epoch={self.start_epoch}, level={self.curriculum.level_name}, best_mse={self.best_mse:.4f}")
            return True
        self.log("No checkpoint found. Starting fresh.")
        return False

    def save_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_state_dict': self.ema.shadow.state_dict(),
            'ema_updates': self.ema.updates,
            'history': self.history,
            'num_sprites': self.num_sprites,
            'curriculum_level': self.curriculum.current_max,
            'curriculum_streak': self.curriculum.streak,
            'curriculum_effective_mse_threshold': self.curriculum.effective_mse_threshold,
            'curriculum_attempts_at_level': self.curriculum.attempts_at_level,
            'best_mse': self.best_mse,
        }
        torch.save(ckpt, CHECKPOINT_FILE)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f)

    def save_best_checkpoint(self, epoch):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.ema.shadow.state_dict(),
            'best_mse': self.best_mse,
            'curriculum_level': self.curriculum.current_max,
            'num_sprites': self.num_sprites,
        }
        torch.save(ckpt, BEST_CHECKPOINT_FILE)

    def run_validation(self):
        mses, dets, gts = [], [], []
        for _ in range(VALIDATION_SAMPLES):
            mse, detected, gt_count = compute_reconstruction_mse(self.ema.shadow, self.dataset, DEVICE)
            mses.append(mse)
            dets.append(detected)
            gts.append(gt_count)
        mse_avg = float(np.mean(mses))
        detected_avg = int(round(np.mean(dets)))
        gt_avg = int(round(np.mean(gts)))

        # Image Reconstruction Accuracy % (higher is better, 100% = 0 MSE)
        accuracy_pct = max(0.0, (1.0 - mse_avg)) * 100.0
        self.history["val_accuracy"].append(accuracy_pct)
        self.history["val_steps"].append(self.global_step)

        # Send full non-resetting accuracy history
        self.update_acc_plot(list(self.history["val_accuracy"]), list(self.history["val_steps"]))

        promoted, success, relaxed = self.curriculum.record_result(mse_avg, detected_avg, gt_avg)

        if relaxed:
            self.log(
                f"⚠️ Easing MSE bar at level {self.curriculum.current_max} → "
                f"{self.curriculum.effective_mse_threshold:.4f}", "WARN"
            )

        if mse_avg < self.best_mse:
            self.best_mse = mse_avg
            self.save_best_checkpoint(self.current_epoch)
            self.log(f"⭐ New best MSE: {self.best_mse:.4f} (Accuracy: {accuracy_pct:.2f}%)")

        if promoted:
            self.dataset.set_difficulty(self.curriculum.current_max)
            self.log(f"🎉 PROMOTED → {self.curriculum.level_name}")
            self.history["level_changes"].append({"step": self.global_step, "level": self.curriculum.current_max})

        self.update_curriculum(self.curriculum.get_summary())
        return promoted

    def train(self):
        self.setup()
        self.load_checkpoint()
        total_batches = len(self.dataloader)
        total_steps = NUM_EPOCHS * total_batches
        self.log(f"Training: {NUM_EPOCHS} epochs × {total_batches} batches")
        self.log("─" * 50)

        try:
            comp, met = generate_visual_comparison(self.ema.shadow, self.dataset, DEVICE)
            self.update_visual(comp, met)
        except Exception:
            pass
        self.update_curriculum(self.curriculum.get_summary())

        for epoch in range(self.start_epoch, NUM_EPOCHS):
            self.current_epoch = epoch
            if self.stop_flag:
                break
            self.model.train()
            epoch_loss = 0.0

            for i, (imgs, tgts) in enumerate(self.dataloader):
                if self.stop_flag:
                    break
                while self.pause_flag:
                    time.sleep(0.1)
                    if self.stop_flag:
                        break
                if self.stop_flag:
                    break

                current_lr = get_lr_at_step(self.global_step, LR, WARMUP_STEPS, total_steps)
                for pg in self.optimizer.param_groups:
                    mult = BACKBONE_LR_MULT if pg.get("name") == "backbone" else 1.0
                    pg['lr'] = current_lr * mult

                imgs = imgs.to(DEVICE, non_blocking=True)
                tgts = tgts.to(DEVICE, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    dn_targets = tgts if USE_DENOISING_TRAINING else None
                    model_out = self.model(imgs, targets=dn_targets, return_all_layers=True)

                if USE_DENOISING_TRAINING:
                    preds_all, dn_all = model_out
                else:
                    preds_all, dn_all = model_out, None

                layer_losses = [self.criterion(p.float(), tgts) for p in preds_all]
                loss = sum(layer_losses) / len(layer_losses)

                if dn_all is not None:
                    dn_layer_losses = [self.criterion.forward_dn(p.float(), tgts) for p in dn_all]
                    loss = loss + W_DN * (sum(dn_layer_losses) / len(dn_layer_losses))

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.model)

                loss_val = float(loss.item())
                epoch_loss += loss_val
                self.global_step += 1
                self.history["batch_losses"].append(loss_val)

                if i % 10 == 0:
                    self.log(
                        f"E{epoch} B{i}/{total_batches} │ Loss:{loss_val:.3f} │ "
                        f"LR:{current_lr:.1e} │ Lvl:{self.curriculum.current_max}"
                    )
                    self.update_loss_plot(self.history["batch_losses"][-300:])

                if self.global_step % VALIDATION_INTERVAL == 0:
                    self.run_validation()

                if self.global_step % VISUAL_UPDATE_INTERVAL == 0:
                    try:
                        comp, met = generate_visual_comparison(self.ema.shadow, self.dataset, DEVICE)
                        self.update_visual(comp, met)
                    except Exception:
                        pass

                self.update_status(epoch, i, total_batches, loss_val, current_lr)

            if not self.stop_flag:
                avg_loss = epoch_loss / max(1, total_batches)
                self.history["epoch_losses"].append(avg_loss)
                self.log(f"═══ Epoch {epoch} │ Loss: {avg_loss:.4f} │ Level: {self.curriculum.level_name} ═══")
                self.save_checkpoint(epoch)

        self.log("Training complete.")
        self.msg_queue.put(("done",))

    def stop(self):
        self.stop_flag = True

    def pause(self):
        self.pause_flag = True

    def resume(self):
        self.pause_flag = False

# ============================================================================
# GUI
# ============================================================================
class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scene AI — DAB-DETR Curriculum Trainer")
        self.root.geometry("1300x850")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(1100, 750)
        self.engine = None
        self.train_thread = None
        self.msg_queue = queue.Queue()
        self.visual_photo = None
        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Consolas", 10))
        style.configure("Title.TLabel", font=("Consolas", 13, "bold"), foreground="#89b4fa")
        style.configure("Status.TLabel", font=("Consolas", 10), foreground="#a6e3a1")
        style.configure("Panel.TLabel", font=("Consolas", 11, "bold"), foreground="#cba6f7")
        style.configure("Stop.TButton", foreground="#f38ba8")
        style.configure("Side.TFrame", background="#181825")
        style.configure("Side.TLabel", background="#181825", foreground="#cdd6f4")
        style.configure("Metric.TLabel", background="#181825", foreground="#f9e2af", font=("Consolas", 10))
        style.configure("Level.TLabel", background="#181825", foreground="#89dceb", font=("Consolas", 12, "bold"))

        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(top, text="🎨 Scene AI — DAB-DETR Curriculum Trainer", style="Title.TLabel").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        btn = ttk.Frame(main)
        btn.pack(fill=tk.X, pady=(0, 4))
        self.btn_start = ttk.Button(btn, text="▶ Start", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self.btn_pause = ttk.Button(btn, text="⏸ Pause", command=self.pause_training, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_resume = ttk.Button(btn, text="⏵ Resume", command=self.resume_training, state=tk.DISABLED)
        self.btn_resume.pack(side=tk.LEFT, padx=2)
        self.btn_save_stop = ttk.Button(btn, text="💾 Save & Stop", command=self.save_and_stop,
                                        style="Stop.TButton", state=tk.DISABLED)
        self.btn_save_stop.pack(side=tk.LEFT, padx=2)
        ttk.Button(btn, text="🗑 Clear", command=self.clear_log).pack(side=tk.RIGHT, padx=2)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_training = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_training, text="Training Dashboard")

        self.tab_inference = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_inference, text="Interactive Visualization")

        pane = tk.PanedWindow(self.tab_training, orient=tk.HORIZONTAL, bg="#313244", sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(pane)
        pane.add(left, width=620, minsize=400)

        # Dual Plots: Top = Loss (Last 300), Bottom = Accuracy % (Non-Resetting)
        self.fig = Figure(figsize=(6, 3.8), dpi=100, facecolor='#1e1e2e')
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_acc = self.fig.add_subplot(212)

        for ax in (self.ax_loss, self.ax_acc):
            ax.set_facecolor('#313244')
            ax.tick_params(colors="#a6adc8", labelsize=7)
            for s in ax.spines.values():
                s.set_color('#585b70')

        self.ax_loss.set_title("Training Loss (Last 300 Batches)", color="#cdd6f4", fontsize=8, pad=3)
        self.ax_acc.set_title("Image Reconstruction Accuracy % (Full History - Non-Resetting)", color="#cdd6f4", fontsize=8, pad=3)
        self.fig.tight_layout()

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_widget.get_tk_widget().pack(fill=tk.X)

        ttk.Label(left, text="Log", style="Panel.TLabel").pack(anchor=tk.W, pady=(4, 0))
        self.log_text = scrolledtext.ScrolledText(left, height=10, bg="#181825", fg="#cdd6f4",
                                                  font=("Consolas", 9), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

        right = ttk.Frame(pane, style="Side.TFrame")
        pane.add(right, width=600, minsize=400)

        curr_frame = ttk.Frame(right, style="Side.TFrame")
        curr_frame.pack(fill=tk.X, pady=(5, 2), padx=5)
        ttk.Label(curr_frame, text="📊 Curriculum", style="Panel.TLabel").pack(anchor=tk.W)
        self.level_var = tk.StringVar(value="Level: 1 object")
        ttk.Label(curr_frame, textvariable=self.level_var, style="Level.TLabel").pack(anchor=tk.W, pady=2)

        streak_frame = ttk.Frame(curr_frame, style="Side.TFrame")
        streak_frame.pack(fill=tk.X, pady=2)
        ttk.Label(streak_frame, text="Streak:", style="Side.TLabel").pack(side=tk.LEFT)
        self.streak_bar = ttk.Progressbar(streak_frame, length=200, mode='determinate', maximum=STREAK_NEEDED)
        self.streak_bar.pack(side=tk.LEFT, padx=5)
        self.streak_var = tk.StringVar(value="0/10")
        ttk.Label(streak_frame, textvariable=self.streak_var, style="Side.TLabel").pack(side=tk.LEFT)

        self.metrics_var = tk.StringVar(value="")
        ttk.Label(curr_frame, textvariable=self.metrics_var, style="Metric.TLabel").pack(anchor=tk.W, pady=2)

        ttk.Label(right, text="👁 Visual Alignment", style="Panel.TLabel").pack(pady=(8, 2))
        self.visual_label = ttk.Label(right, style="Side.TLabel")
        self.visual_label.pack(pady=2, expand=True)
        self.visual_placeholder = tk.StringVar(value="Waiting for first validation...")
        ttk.Label(right, textvariable=self.visual_placeholder, style="Side.TLabel").pack()

        # --- Interactive Visualization Tab ---
        self._build_inference_ui()

    def _build_inference_ui(self):
        inf_frame = ttk.Frame(self.tab_inference, padding=10)
        inf_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(inf_frame)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Select Sprite:").pack(side=tk.LEFT, padx=5)

        # We will populate this when engine starts
        self.sprite_combo = ttk.Combobox(controls, state="readonly", width=30)
        self.sprite_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls, text="Predict", command=self.run_interactive_prediction).pack(side=tk.LEFT, padx=20)
        ttk.Button(controls, text="Clear Canvas", command=self.clear_interactive_canvas).pack(side=tk.LEFT, padx=5)

        canvas_frame = ttk.Frame(inf_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        left_cv_frame = ttk.Frame(canvas_frame)
        left_cv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(left_cv_frame, text="Click to place sprites:", style="Panel.TLabel").pack(pady=5)

        # Interactive Canvas
        self.user_canvas = tk.Canvas(left_cv_frame, width=IMG_SIZE, height=IMG_SIZE, bg="gray")
        self.user_canvas.pack()
        self.user_canvas.bind("<Button-1>", self.on_canvas_click)

        right_cv_frame = ttk.Frame(canvas_frame)
        right_cv_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(right_cv_frame, text="AI Prediction:", style="Panel.TLabel").pack(pady=5)

        self.ai_canvas = tk.Canvas(right_cv_frame, width=IMG_SIZE, height=IMG_SIZE, bg="gray")
        self.ai_canvas.pack()

        self.placed_objects = []

    def clear_interactive_canvas(self):
        self.placed_objects = []
        self.user_canvas.delete("all")
        self.ai_canvas.delete("all")

    def on_canvas_click(self, event):
        if not self.engine or not self.engine.dataset:
            return

        sprite_idx = self.sprite_combo.current()
        if sprite_idx < 0:
            return

        x, y = event.x, event.y
        scale = random.uniform(0.5, 1.5)
        angle = random.uniform(0, 360)
        color_shift = np.random.uniform(-30, 30, size=3)
        layer = random.uniform(0, 1)

        self.placed_objects.append({
            "id": sprite_idx, "x": x, "y": y, "scale": scale,
            "angle": angle, "color_shift": color_shift, "layer": layer
        })

        # Draw on user canvas using PIL for complex rendering, then convert to ImageTk
        self._update_user_canvas()

    def _update_user_canvas(self):
        canvas_img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (128, 128, 128, 255))
        self.placed_objects.sort(key=lambda o: o["layer"])

        for obj in self.placed_objects:
            self.engine.dataset.render_sprite(canvas_img, self.engine.dataset.sprites[obj["id"]],
                                              obj["x"], obj["y"], obj["scale"], obj["angle"], obj["color_shift"])

        self.user_photo = ImageTk.PhotoImage(canvas_img)
        self.user_canvas.create_image(0, 0, anchor=tk.NW, image=self.user_photo)

    def run_interactive_prediction(self):
        if not self.engine or not self.engine.ema or not self.placed_objects:
            return

        # Build tensor from placed objects
        canvas_img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (128, 128, 128, 255))
        for obj in self.placed_objects:
            self.engine.dataset.render_sprite(canvas_img, self.engine.dataset.sprites[obj["id"]],
                                              obj["x"], obj["y"], obj["scale"], obj["angle"], obj["color_shift"])

        img_rgb = canvas_img.convert("RGB")
        img_tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0

        self.engine.ema.shadow.eval()
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(DEVICE)
            pred = self.engine.ema.shadow(img_input)[0]

        probs = torch.sigmoid(pred[:, 0])
        presence = probs > 0.5

        predictions = []
        for i in range(self.engine.dataset.max_objects):
            if presence[i]:
                obj_id = torch.argmax(pred[i, 1:1 + self.engine.dataset.num_sprites]).item()
                attrs = pred[i, 1 + self.engine.dataset.num_sprites:]
                layer_depth = attrs[8].item()
                predictions.append((layer_depth, obj_id, attrs))

        predictions.sort(key=lambda p: p[0])

        ai_canvas_img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (128, 128, 128, 255))
        for layer_depth, obj_id, attrs in predictions:
            x = attrs[0].item() * IMG_SIZE
            y = attrs[1].item() * IMG_SIZE
            angle = math.degrees(math.atan2(attrs[2].item(), attrs[3].item()))
            scale = attrs[4].item() * 3.0
            cs = attrs[5:8].cpu().numpy() * 30.0
            self.engine.dataset.render_sprite(ai_canvas_img, self.engine.dataset.sprites[obj_id], x, y, scale, angle, cs)

        self.ai_photo = ImageTk.PhotoImage(ai_canvas_img)
        self.ai_canvas.create_image(0, 0, anchor=tk.NW, image=self.ai_photo)


    def append_log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def update_loss_plot(self, losses):
        if not losses:
            return
        self.ax_loss.clear()
        self.ax_loss.set_facecolor('#313244')
        self.ax_loss.set_title("Training Loss (Last 300 Batches)", color="#cdd6f4", fontsize=8, pad=3)
        self.ax_loss.tick_params(colors="#a6adc8", labelsize=7)
        for s in self.ax_loss.spines.values():
            s.set_color('#585b70')

        self.ax_loss.plot(losses, color="#f38ba8", alpha=0.35, linewidth=1.0, label="Raw")
        if len(losses) > 5:
            ema = []
            alpha = 0.1
            v = losses[0]
            for x in losses:
                v = alpha * x + (1 - alpha) * v
                ema.append(v)
            self.ax_loss.plot(ema, color="#89b4fa", linewidth=1.2, label="EMA")

        self.fig.tight_layout()
        self.canvas_widget.draw_idle()

    def update_acc_plot(self, accuracies, steps):
        if not accuracies:
            return
        self.ax_acc.clear()
        self.ax_acc.set_facecolor('#313244')
        self.ax_acc.set_title("Image Reconstruction Accuracy % (Full History - Non-Resetting)", color="#cdd6f4", fontsize=8, pad=3)
        self.ax_acc.tick_params(colors="#a6adc8", labelsize=7)
        for s in self.ax_acc.spines.values():
            s.set_color('#585b70')

        x_axis = steps if (steps and len(steps) == len(accuracies)) else list(range(1, len(accuracies) + 1))
        self.ax_acc.plot(x_axis, accuracies, color="#a6e3a1", linewidth=1.5, marker='o', markersize=2)
        self.ax_acc.set_ylabel("Accuracy %", color="#a6adc8", fontsize=7)

        min_acc = max(0.0, min(accuracies) - 2.0)
        max_acc = min(100.0, max(accuracies) + 2.0)
        if max_acc > min_acc:
            self.ax_acc.set_ylim(min_acc, max_acc)

        self.fig.tight_layout()
        self.canvas_widget.draw_idle()

    def update_visual(self, pil_img, metrics):
        max_w = 520
        w, h = pil_img.size
        if w > max_w:
            sc = max_w / w
            pil_img = pil_img.resize((max_w, int(h * sc)), Image.LANCZOS)
        self.visual_photo = ImageTk.PhotoImage(pil_img)
        self.visual_label.config(image=self.visual_photo)
        self.visual_placeholder.set("")
        gt = metrics.get("gt_objects", "?")
        det = metrics.get("detected_objects", "?")
        self.metrics_var.set(f"GT Objects: {gt} │ AI Detected: {det}")

    def update_curriculum(self, summary):
        thresh = summary.get("effective_mse_threshold")
        thresh_str = f"  (MSE Target < {thresh:.4f})" if thresh is not None else ""
        self.level_var.set(f"Level: {summary['level_name']}{thresh_str}")
        self.streak_bar['value'] = summary["streak"]
        self.streak_bar['maximum'] = summary["streak_needed"]
        self.streak_var.set(f"{summary['streak']}/{summary['streak_needed']}")

    def start_training(self):
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_save_stop.config(state=tk.NORMAL)
        self.engine = TrainingEngine(self.msg_queue)

        # We need to setup dataset before starting training thread to populate UI
        self.engine.setup()
        self.sprite_combo['values'] = self.engine.dataset.sprite_names
        if self.engine.dataset.sprite_names:
            self.sprite_combo.current(0)

        # We start a modified training method since we already setup
        self.train_thread = threading.Thread(target=self._run_training, daemon=True)
        self.train_thread.start()

    def _run_training(self):
        # Continue the training setup process that was inside engine.train()
        self.engine.load_checkpoint()
        total_batches = len(self.engine.dataloader)
        total_steps = NUM_EPOCHS * total_batches
        self.engine.log(f"Training: {NUM_EPOCHS} epochs × {total_batches} batches")
        self.engine.log("─" * 50)

        try:
            comp, met = generate_visual_comparison(self.engine.ema.shadow, self.engine.dataset, DEVICE)
            self.engine.update_visual(comp, met)
        except Exception:
            pass
        self.engine.update_curriculum(self.engine.curriculum.get_summary())

        # Re-use the training loop from engine.train() by calling a slightly modified version
        self._engine_train_loop(total_batches, total_steps)

    def _engine_train_loop(self, total_batches, total_steps):
        for epoch in range(self.engine.start_epoch, NUM_EPOCHS):
            self.engine.current_epoch = epoch
            if self.engine.stop_flag:
                break
            self.engine.model.train()
            epoch_loss = 0.0

            for i, (imgs, tgts) in enumerate(self.engine.dataloader):
                if self.engine.stop_flag:
                    break
                while self.engine.pause_flag:
                    time.sleep(0.1)
                    if self.engine.stop_flag:
                        break
                if self.engine.stop_flag:
                    break

                current_lr = get_lr_at_step(self.engine.global_step, LR, WARMUP_STEPS, total_steps)
                for pg in self.engine.optimizer.param_groups:
                    mult = BACKBONE_LR_MULT if pg.get("name") == "backbone" else 1.0
                    pg['lr'] = current_lr * mult

                imgs = imgs.to(DEVICE, non_blocking=True)
                tgts = tgts.to(DEVICE, non_blocking=True)
                self.engine.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    dn_targets = tgts if USE_DENOISING_TRAINING else None
                    model_out = self.engine.model(imgs, targets=dn_targets, return_all_layers=True)

                if USE_DENOISING_TRAINING:
                    preds_all, dn_all = model_out
                else:
                    preds_all, dn_all = model_out, None

                layer_losses = [self.engine.criterion(p.float(), tgts) for p in preds_all]
                loss = sum(layer_losses) / len(layer_losses)

                if dn_all is not None:
                    dn_layer_losses = [self.engine.criterion.forward_dn(p.float(), tgts) for p in dn_all]
                    loss = loss + W_DN * (sum(dn_layer_losses) / len(dn_layer_losses))

                self.engine.scaler.scale(loss).backward()
                self.engine.scaler.unscale_(self.engine.optimizer)
                torch.nn.utils.clip_grad_norm_(self.engine.model.parameters(), GRAD_CLIP)
                self.engine.scaler.step(self.engine.optimizer)
                self.engine.scaler.update()
                self.engine.ema.update(self.engine.model)

                loss_val = float(loss.item())
                epoch_loss += loss_val
                self.engine.global_step += 1
                self.engine.history["batch_losses"].append(loss_val)

                if i % 10 == 0:
                    self.engine.log(
                        f"E{epoch} B{i}/{total_batches} │ Loss:{loss_val:.3f} │ "
                        f"LR:{current_lr:.1e} │ Lvl:{self.engine.curriculum.current_max}"
                    )
                    self.engine.update_loss_plot(self.engine.history["batch_losses"][-300:])

                if self.engine.global_step % VALIDATION_INTERVAL == 0:
                    self.engine.run_validation()

                if self.engine.global_step % VISUAL_UPDATE_INTERVAL == 0:
                    try:
                        comp, met = generate_visual_comparison(self.engine.ema.shadow, self.engine.dataset, DEVICE)
                        self.engine.update_visual(comp, met)
                    except Exception:
                        pass

                self.engine.update_status(epoch, i, total_batches, loss_val, current_lr)

            if not self.engine.stop_flag:
                avg_loss = epoch_loss / max(1, total_batches)
                self.engine.history["epoch_losses"].append(avg_loss)
                self.engine.log(f"═══ Epoch {epoch} │ Loss: {avg_loss:.4f} │ Level: {self.engine.curriculum.level_name} ═══")
                self.engine.save_checkpoint(epoch)

        self.engine.log("Training complete.")
        self.engine.msg_queue.put(("done",))


    def pause_training(self):
        if self.engine:
            self.engine.pause()
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.NORMAL)

    def resume_training(self):
        if self.engine:
            self.engine.resume()
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.DISABLED)

    def save_and_stop(self):
        if self.engine:
            self.append_log("[INFO] Saving state and stopping gracefully...")
            self.engine.stop()

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "log":
                    self.append_log(msg[1])
                elif msg[0] == "status":
                    _, ep, bat, tot, loss, lr = msg
                    self.status_var.set(f"E{ep+1}/{NUM_EPOCHS} B{bat+1}/{tot} │ Loss: {loss:.3f} │ LR: {lr:.1e}")
                elif msg[0] == "loss_plot":
                    self.update_loss_plot(msg[1])
                elif msg[0] == "acc_plot":
                    self.update_acc_plot(msg[1], msg[2])
                elif msg[0] == "visual":
                    self.update_visual(msg[1], msg[2])
                elif msg[0] == "curriculum":
                    self.update_curriculum(msg[1])
                elif msg[0] == "done":
                    self.status_var.set("✅ Training Complete")
                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_pause.config(state=tk.DISABLED)
                    self.btn_resume.config(state=tk.DISABLED)
                    self.btn_save_stop.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def on_close(self):
        if self.engine:
            self.engine.stop()
            time.sleep(0.3)
        self.root.destroy()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    def auto_start():
        app.start_training()

    root.after(500, auto_start)
    root.mainloop()