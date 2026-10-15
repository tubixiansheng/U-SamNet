from torchvision import transforms
from utils import (
    GT_BceDiceLoss,
    myNormalize,
    myRandomHorizontalFlip,
    myRandomLesionFocus,
    myRandomRotation,
    myRandomScaleCrop,
    myRandomVerticalFlip,
    myResize,
    myToTensor,
)

import copy
from datetime import datetime
from types import SimpleNamespace


DATASET_CONFIGS = {
    'mosmeddata': {
        'path': './data/MosMedData/',
        'image_dir_name': 'images',
        'mask_dir_name': 'masks',
        'normalize': {
            'train': {'mean': 159.922, 'std': 28.871},
            'val': {'mean': 148.429, 'std': 25.748},
        },
    },
    'szu-bch-bus2093': {
        'path': './data/SZU-BCH-BUS2093/',
        'image_dir_name': 'images',
        'mask_dir_name': 'masks',
        'normalize': {
            'train': {'mean': 51.468704, 'std': 47.121796},
            'val': {'mean': 50.91452, 'std': 47.114117},
        },
    },
    'szu-bch-tus983': {
        'path': './data/SZU-BCH-TUS983/',
        'image_dir_name': 'images',
        'mask_dir_name': 'masks',
        'normalize': {
            'train': {'mean': 49.013683, 'std': 44.039753},
            'val': {'mean': 48.556217, 'std': 43.780994},
        },
    },
    'busi': {
        'path': './data/BUSI/',
        'image_dir_name': 'images',
        'mask_dir_name': 'labels',
        'normalize': {
            'train': {'mean': 83.897738, 'std': 56.973547},
            'val': {'mean': 85.346738, 'std': 56.609226},
        },
    },
    'szu-bch-bus2744': {
        'path': './data/SZU-BCH-BUS2744/',
        'image_dir_name': 'images',
        'mask_dir_name': 'masks',
        'normalize': {
            'train': {'mean': 97.445770, 'std': 50.533020},
            'val': {'mean': 91.125404, 'std': 50.246891},
        },
    },
}


def canonicalize_dataset_name(dataset_name):
    dataset_name = dataset_name.strip()
    aliases = {
        'BUSI': 'busi',
        'busi': 'busi',
        'MosMedData': 'mosmeddata',
        'mosmeddata': 'mosmeddata',
        'SZU-BCH-BUS2093': 'szu-bch-bus2093',
        'szu-bch-bus2093': 'szu-bch-bus2093',
        'SZU-BCH-TUS983': 'szu-bch-tus983',
        'szu-bch-tus983': 'szu-bch-tus983',
        'SZU-BCH-BUS2744': 'szu-bch-bus2744',
        'szu-bch-bus2744': 'szu-bch-bus2744',
    }
    canonical_name = aliases.get(dataset_name, dataset_name.lower())
    if canonical_name not in DATASET_CONFIGS:
        raise ValueError(f'Unsupported dataset: {dataset_name}. Available: {", ".join(DATASET_CONFIGS.keys())}')
    return canonical_name


def get_dataset_config(dataset_name):
    return copy.deepcopy(DATASET_CONFIGS[canonicalize_dataset_name(dataset_name)])


class setting_config:
    """
    the config of training setting.
    """

    network = 'u_samnet'
    experiment_tag = 'usamnet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [16, 32, 48, 64, 96, 128],
        'bridge': True,
        'gt_ds': True,
        'img_size': 256,
        'sam_model': 'b',
        'sam_checkpoint': './models/sam_pretrain/sam_vit_b_01ec64.pth',
        'sam_freeze': True,
        'decoder4_3_fusion': 'drpf',
        'rghpa_mode': 'dynamic',
        'attention_module': 'rghpa',
        'rghpa_use_cab': None,
        'rghpa_use_boundary_gate': None,
        'rghpa_use_dynamic_routing': None,
        'rghpa_router_mode': 'full',
        'rghpa_router_temperature': 1.0,
        'rghpa_router_residual': 0.0,
        'rghpa_boundary_scale': 1.0,
        'rghpa_boundary_gate_strength': 1.0,
        'rghpa_boundary_gate_residual': 0.0,
        'drpf_discrepancy_penalty': 0.0,
        'enable_boundary_head': True,
        'enable_uncertainty_heads': True,
        'use_sam_adapter': True,
        'use_shared_sam_decoder': False,
        'shared_sam_decoder_dim': 64,
        'shared_sam_decoder_mode': 'full',
    }

    datasets = 'busi'
    _dataset_meta = get_dataset_config(datasets)
    data_path = _dataset_meta['path']
    image_dir_name = _dataset_meta['image_dir_name']
    mask_dir_name = _dataset_meta['mask_dir_name']

    criterion = GT_BceDiceLoss(
        wb=1,
        wd=1,
        gt_branch_weight=0.35,
        mask_branch_weight=0.25,
        boundary_weight=0.2,
        uncertainty_weight=0.1,
        confidence_weight=0.08,
        boundary_dilation=2,
    )

    pretrained_path = ''
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 1
    epochs = 140

    early_stopping = True
    early_stopping_patience = 18
    early_stopping_min_delta = 0.0005
    min_epochs_before_stop = 40

    run_name = network + '_' + experiment_tag + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    work_dir = 'results/' + run_name + '/'

    print_interval = 20
    val_interval = 2
    save_interval = 2
    threshold = 0.5
    train_visualize = True
    train_visualize_interval = 10
    train_visualize_samples = 1
    max_train_batches = None
    max_val_batches = None
    max_test_batches = None
    ablation_name = 'current_best'
    ablation_description = 'Released U-SamNet configuration with weak residual boundary gating.'

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myRandomScaleCrop(input_size_h, input_size_w, p=0.4, scale_range=(0.85, 1.2)),
        myRandomLesionFocus(input_size_h, input_size_w, p=0.35, context_range=(1.2, 1.9)),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01  # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9  # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6  # default: 1e-6 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adagrad':
        lr = 0.01  # default: 0.01 – learning rate
        lr_decay = 0  # default: 0 – learning rate decay
        eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001  # default: 1e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 1e-4
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-4
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3  # default: 2e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01  # default: 1e-2 – learning rate
        lambd = 1e-4  # default: 1e-4 – decay term
        alpha = 0.75  # default: 0.75 – power for eta update
        t0 = 1e6  # default: 1e6 – point at which to start averaging
        weight_decay = 0  # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        momentum = 0  # default: 0 – momentum factor
        alpha = 0.99  # default: 0.99 – smoothing constant
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        etas = (0.5,
                1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01  # – learning rate
        momentum = 0.9  # default: 0 – momentum factor
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        dampening = 0  # default: 0 – dampening for momentum
        nesterov = False  # default: False – enables Nesterov momentum

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5  # – Period of learning rate decay.
        gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
        gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99  # – Multiplicative factor of learning rate decay.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50  # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50  # – Number of iterations for the first restart.
        T_mult = 2  # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20


DEFAULT_LOSS_KWARGS = {
    'wb': 1,
    'wd': 1,
    'gt_branch_weight': 0.35,
    'mask_branch_weight': 0.25,
    'boundary_weight': 0.2,
    'uncertainty_weight': 0.1,
    'confidence_weight': 0.08,
    'boundary_dilation': 2,
}


def build_criterion(loss_overrides=None):
    criterion_kwargs = copy.deepcopy(DEFAULT_LOSS_KWARGS)
    if loss_overrides:
        criterion_kwargs.update(loss_overrides)
    return GT_BceDiceLoss(**criterion_kwargs)


def build_train_transformer(datasets, input_size_h, input_size_w, use_targeted_augmentation=True):
    transform_list = [
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
    ]
    if use_targeted_augmentation:
        transform_list.extend([
            myRandomScaleCrop(input_size_h, input_size_w, p=0.4, scale_range=(0.85, 1.2)),
            myRandomLesionFocus(input_size_h, input_size_w, p=0.35, context_range=(1.2, 1.9)),
        ])
    transform_list.append(myResize(input_size_h, input_size_w))
    return transforms.Compose(transform_list)


def build_test_transformer(datasets, input_size_h, input_size_w):
    return transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])


def _ensure_trailing_slash(path):
    return path if path.endswith('/') else path + '/'


def create_config(
    ablation_name='current_best',
    dataset_name=None,
    dataset_root=None,
    experiment_tag=None,
    run_name=None,
    work_dir=None,
    gpu_id=None,
    epochs=None,
    batch_size=None,
    threshold=None,
    train_visualize=None,
    max_train_batches=None,
    max_val_batches=None,
    max_test_batches=None,
    extra_overrides=None,
):
    from configs.ablation_registry import get_ablation_spec

    ablation_spec = get_ablation_spec(ablation_name)

    config = SimpleNamespace()
    for key, value in setting_config.__dict__.items():
        if key.startswith('__'):
            continue
        if callable(value):
            continue
        setattr(config, key, copy.deepcopy(value))

    config.ablation_name = ablation_name
    config.ablation_description = ablation_spec['description']

    canonical_dataset_name = canonicalize_dataset_name(dataset_name or setting_config.datasets)
    dataset_meta = get_dataset_config(canonical_dataset_name)
    config.datasets = canonical_dataset_name
    config.data_path = _ensure_trailing_slash(dataset_root or dataset_meta['path'])
    config.image_dir_name = dataset_meta['image_dir_name']
    config.mask_dir_name = dataset_meta['mask_dir_name']

    config.model_config = copy.deepcopy(setting_config.model_config)
    config.model_config.update(ablation_spec.get('model_overrides', {}))

    config.criterion = build_criterion(ablation_spec.get('loss_overrides', {}))

    use_targeted_augmentation = ablation_spec.get('augmentation_overrides', {}).get('use_targeted_augmentation', True)
    config.train_transformer = build_train_transformer(config.datasets, config.input_size_h, config.input_size_w, use_targeted_augmentation=use_targeted_augmentation)
    config.test_transformer = build_test_transformer(config.datasets, config.input_size_h, config.input_size_w)

    base_tag = experiment_tag or setting_config.experiment_tag
    tag_suffix = ablation_spec.get('tag_suffix')
    if ablation_name != 'full' and tag_suffix:
        config.experiment_tag = f'{base_tag}_{tag_suffix}'
    else:
        config.experiment_tag = base_tag

    if gpu_id is not None:
        config.gpu_id = gpu_id
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size
    if threshold is not None:
        config.threshold = threshold
    if train_visualize is not None:
        config.train_visualize = train_visualize
    if max_train_batches is not None:
        config.max_train_batches = max_train_batches
    if max_val_batches is not None:
        config.max_val_batches = max_val_batches
    if max_test_batches is not None:
        config.max_test_batches = max_test_batches

    if extra_overrides:
        for key, value in extra_overrides.items():
            setattr(config, key, value)

    timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    config.run_name = run_name or f'{config.network}_{config.experiment_tag}_{config.datasets}_{timestamp}'
    config.work_dir = _ensure_trailing_slash(work_dir or f'results/{config.run_name}')
    return config
