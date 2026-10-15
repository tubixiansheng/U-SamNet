import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = '#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = config.mode,
            factor = config.factor,
            patience = config.patience,
            threshold = config.threshold,
            threshold_mode = config.threshold_mode,
            cooldown = config.cooldown,
            min_lr = config.min_lr,
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        def lr_func(epoch):
            if epoch <= config.warm_up_epochs:
                return epoch / config.warm_up_epochs
            return config.gamma ** len([m for m in config.milestones if m <= epoch])

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        def lr_func(epoch):
            if epoch <= config.warm_up_epochs:
                return epoch / config.warm_up_epochs
            progress = (epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs)
            return 0.5 * (math.cos(progress * math.pi) + 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()


def tensor_to_numpy_image(image_tensor):
    image = image_tensor.detach().cpu().squeeze(0)
    if image.dim() == 3:
        image = image.permute(1, 2, 0).numpy()
    else:
        image = image.numpy()
    image = image.astype(np.float32)
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-6)
    return (image * 255).clip(0, 255).astype(np.uint8)


def mask_to_uint8(mask_tensor, threshold=0.5):
    mask = mask_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    mask = (mask >= threshold).astype(np.uint8) * 255
    return mask


def heatmap_to_uint8(heatmap_tensor):
    heatmap = heatmap_tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = plt.get_cmap('jet')(heatmap)[..., :3]
    return (colored * 255).astype(np.uint8)


def overlay_heatmap_on_image(image_uint8, heatmap_uint8, alpha=0.45):
    image = image_uint8.astype(np.float32)
    heatmap = heatmap_uint8.astype(np.float32)
    blended = image * (1.0 - alpha) + heatmap * alpha
    return blended.clip(0, 255).astype(np.uint8)


def overlay_mask_on_image(image_uint8, mask_uint8, color=(255, 80, 80), alpha=0.45):
    image = image_uint8.astype(np.float32)
    overlay = image.copy()
    mask = mask_uint8 > 0
    overlay[mask] = np.array(color, dtype=np.float32)
    blended = image * (1.0 - alpha) + overlay * alpha
    return blended.clip(0, 255).astype(np.uint8)


def save_visualization_bundle(image_tensor, prediction_tensor, visual_maps, output_root, sample_name, threshold=0.5):
    output_root = Path(output_root)
    original_dir = output_root / 'original'
    prediction_dir = output_root / 'prediction'
    overlay_dir = output_root / 'overlay'
    heatmap_root = output_root / 'heatmaps'
    for directory in [original_dir, prediction_dir, overlay_dir, heatmap_root]:
        directory.mkdir(parents=True, exist_ok=True)

    image_uint8 = tensor_to_numpy_image(image_tensor)
    prediction_uint8 = mask_to_uint8(prediction_tensor, threshold=threshold)
    prediction_overlay = overlay_mask_on_image(image_uint8, prediction_uint8)

    Image.fromarray(image_uint8).save(original_dir / f'{sample_name}.png')
    Image.fromarray(prediction_uint8).save(prediction_dir / f'{sample_name}.png')
    Image.fromarray(prediction_overlay).save(overlay_dir / f'{sample_name}.png')

    if visual_maps is None:
        return

    for module_name, heatmap_tensor in visual_maps.items():
        module_parts = module_name.split('/')
        module_dir = heatmap_root.joinpath(*module_parts[:-1]) if len(module_parts) > 1 else heatmap_root
        module_dir.mkdir(parents=True, exist_ok=True)
        heatmap_uint8 = heatmap_to_uint8(heatmap_tensor)
        heatmap_overlay = overlay_heatmap_on_image(image_uint8, heatmap_uint8)
        Image.fromarray(heatmap_uint8).save(module_dir / f'{sample_name}_{module_parts[-1]}_heatmap.png')
        Image.fromarray(heatmap_overlay).save(module_dir / f'{sample_name}_{module_parts[-1]}_overlay.png')



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


def build_boundary_target(target, dilation=2):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = (target > 0.5).float()
    kernel_size = dilation * 2 + 1
    dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=dilation)
    eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=kernel_size, stride=1, padding=dilation)
    return (dilated - eroded).clamp(0.0, 1.0)

class GT_BceDiceLoss(nn.Module):
    def __init__(
        self,
        wb=1,
        wd=1,
        gt_branch_weight=0.35,
        mask_branch_weight=0.25,
        boundary_weight=0.2,
        uncertainty_weight=0.1,
        confidence_weight=0.08,
        boundary_dilation=2,
    ):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.gt_branch_weight = gt_branch_weight
        self.mask_branch_weight = mask_branch_weight
        self.boundary_weight = boundary_weight
        self.uncertainty_weight = uncertainty_weight
        self.confidence_weight = confidence_weight
        self.boundary_dilation = boundary_dilation

    def _resize_target_like(self, target, prediction):
        if target.shape[-2:] == prediction.shape[-2:]:
            return target
        return F.interpolate(target, size=prediction.shape[-2:], mode='nearest')

    def _branch_loss(self, predictions, target, branch_weight):
        if predictions is None:
            return target.new_tensor(0.0)
        if isinstance(predictions, torch.Tensor):
            predictions = (predictions,)
        if len(predictions) == 0:
            return target.new_tensor(0.0)

        weights = torch.linspace(0.6, 1.0, steps=len(predictions), device=target.device)
        weights = weights / weights.sum()
        losses = [self.bcedice(prediction, target) for prediction in predictions]
        return branch_weight * sum(weight * loss for weight, loss in zip(weights, losses))

    def _probability_loss(self, prediction, target):
        return self.bce(prediction, target) + self.dice(prediction, target)

    def _uncertainty_loss(self, predictions, target, branch_weight=None):
        if predictions is None:
            return target.new_tensor(0.0)
        if isinstance(predictions, torch.Tensor):
            predictions = (predictions,)
        if len(predictions) == 0:
            return target.new_tensor(0.0)

        if branch_weight is None:
            branch_weight = self.uncertainty_weight
        weights = torch.linspace(0.7, 1.0, steps=len(predictions), device=target.device)
        weights = weights / weights.sum()
        losses = [self.bce(prediction, self._resize_target_like(target, prediction)) for prediction in predictions]
        return branch_weight * sum(weight * loss for weight, loss in zip(weights, losses))

    def _confidence_loss(self, predictions, target, branch_weight=None):
        if predictions is None:
            return target.new_tensor(0.0)
        if isinstance(predictions, torch.Tensor):
            predictions = (predictions,)
        if len(predictions) == 0:
            return target.new_tensor(0.0)

        if branch_weight is None:
            branch_weight = self.confidence_weight
        weights = torch.linspace(0.8, 1.0, steps=len(predictions), device=target.device)
        weights = weights / weights.sum()
        losses = [self.bce(prediction, self._resize_target_like(target, prediction)) for prediction in predictions]
        return branch_weight * sum(weight * loss for weight, loss in zip(weights, losses))

    def forward(self, gt_pre, masks, out, target, extras=None, dynamic_weights=None):
        if isinstance(out, tuple):
            out = out[-1]

        bcediceloss = self.bcedice(out, target)
        gt_weight = self.gt_branch_weight
        mask_weight = self.mask_branch_weight
        boundary_weight = self.boundary_weight
        uncertainty_weight = self.uncertainty_weight
        confidence_weight = self.confidence_weight

        if dynamic_weights is not None:
            gt_weight = dynamic_weights.get("gt_branch_weight", gt_weight)
            mask_weight = dynamic_weights.get("mask_branch_weight", mask_weight)
            boundary_weight = dynamic_weights.get("boundary_weight", boundary_weight)
            uncertainty_weight = dynamic_weights.get("uncertainty_weight", uncertainty_weight)
            confidence_weight = dynamic_weights.get("confidence_weight", confidence_weight)

        gt_loss = self._branch_loss(gt_pre, target, gt_weight)
        mask_loss = self._branch_loss(masks, target, mask_weight)
        extras = extras or {}
        boundary_target = build_boundary_target(target, dilation=self.boundary_dilation)
        boundary_prediction = extras.get('boundary')
        boundary_loss = target.new_tensor(0.0)
        if boundary_prediction is not None:
            boundary_loss = boundary_weight * self._probability_loss(boundary_prediction, boundary_target)
        uncertainty_loss = self._uncertainty_loss(extras.get('uncertainty'), boundary_target, branch_weight=uncertainty_weight)
        confidence_target = 1.0 - boundary_target
        fusion_confidence_loss = self._confidence_loss(extras.get('fusion_confidence'), confidence_target, branch_weight=confidence_weight)

        return bcediceloss + gt_loss + mask_loss + boundary_loss + uncertainty_loss + fusion_confidence_loss


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float(), torch.from_numpy(np.ascontiguousarray(mask)).permute(2,0,1).float()
class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return (
            TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR, antialias=True),
            TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST),
        )


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask
class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.degree = degree
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            return (
                TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR),
                TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST),
            )
        return image, mask
class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'mosmeddata':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'busi':
            if train:
                self.mean = 83.897738
                self.std = 56.973547
            else:
                self.mean = 85.346738
                self.std = 56.609226
        elif data_name == 'szu-bch-bus2093':
            if train:
                self.mean = 51.468704
                self.std = 47.121796
            else:
                self.mean = 50.91452
                self.std = 47.114117
        elif data_name == 'szu-bch-tus983':
            if train:
                self.mean = 49.013683
                self.std = 44.039753
            else:
                self.mean = 48.556217
                self.std = 43.780994
        elif data_name == 'szu-bch-bus2744':
            if train:
                self.mean = 97.445770
                self.std = 50.533020
            else:
                self.mean = 91.125404
                self.std = 50.246891
        else:
            raise ValueError(f'Unsupported dataset normalization config: {data_name}')

    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        value_range = np.max(img_normalized) - np.min(img_normalized)
        if value_range > 0:
            img_normalized = ((img_normalized - np.min(img_normalized)) / value_range) * 255.
        else:
            img_normalized = np.zeros_like(img_normalized)
        return img_normalized, msk


class myRandomScaleCrop:
    def __init__(self, size_h=256, size_w=256, p=0.4, scale_range=(0.85, 1.2)):
        self.size_h = size_h
        self.size_w = size_w
        self.p = p
        self.scale_range = scale_range

    def __call__(self, data):
        image, mask = data
        if random.random() >= self.p:
            return image, mask

        _, height, width = image.shape
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        scaled_h = max(16, int(round(height * scale)))
        scaled_w = max(16, int(round(width * scale)))

        image = TF.resize(image, [scaled_h, scaled_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [scaled_h, scaled_w], interpolation=InterpolationMode.NEAREST)

        pad_h = max(0, self.size_h - scaled_h)
        pad_w = max(0, self.size_w - scaled_w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            image = TF.pad(image, [left, top, right, bottom], fill=0)
            mask = TF.pad(mask, [left, top, right, bottom], fill=0)

        _, height, width = image.shape
        top = 0 if height == self.size_h else random.randint(0, height - self.size_h)
        left = 0 if width == self.size_w else random.randint(0, width - self.size_w)
        return TF.crop(image, top, left, self.size_h, self.size_w), TF.crop(mask, top, left, self.size_h, self.size_w)


class myRandomLesionFocus:
    def __init__(self, size_h=256, size_w=256, p=0.35, context_range=(1.2, 1.9)):
        self.size_h = size_h
        self.size_w = size_w
        self.p = p
        self.context_range = context_range

    def __call__(self, data):
        image, mask = data
        if random.random() >= self.p:
            return image, mask

        lesion = mask[0] > 0.5
        if lesion.sum() == 0:
            return image, mask

        ys, xs = torch.where(lesion)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        height = max(1, y_max - y_min + 1)
        width = max(1, x_max - x_min + 1)
        context = random.uniform(self.context_range[0], self.context_range[1])
        crop_h = min(image.shape[1], max(height + 8, int(round(height * context))))
        crop_w = min(image.shape[2], max(width + 8, int(round(width * context))))

        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        shift_y = int(round((random.random() - 0.5) * 0.2 * crop_h))
        shift_x = int(round((random.random() - 0.5) * 0.2 * crop_w))
        center_y = min(max(center_y + shift_y, crop_h // 2), image.shape[1] - (crop_h - crop_h // 2))
        center_x = min(max(center_x + shift_x, crop_w // 2), image.shape[2] - (crop_w - crop_w // 2))

        top = min(max(center_y - crop_h // 2, 0), image.shape[1] - crop_h)
        left = min(max(center_x - crop_w // 2, 0), image.shape[2] - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        return (
            TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR, antialias=True),
            TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST),
        )
