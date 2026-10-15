from collections import OrderedDict
import copy


CURRENT_BEST_ABLATION = 'current_best'


ABLATON_SPECS = OrderedDict({
    CURRENT_BEST_ABLATION: {
        'description': 'Released U-SamNet configuration with weak residual boundary gating.',
        'tag_suffix': 'weak_boundary_gate',
        'model_overrides': {
            'rghpa_use_boundary_gate': True,
            'rghpa_boundary_gate_strength': 0.35,
            'rghpa_boundary_gate_residual': 0.35,
            'rghpa_boundary_scale': 0.6,
            'rghpa_router_residual': 0.1,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'full': {
        'description': 'Full model with DRPF, boundary supervision, dynamic CDBHA, and targeted augmentation.',
        'tag_suffix': 'full',
        'model_overrides': {},
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_sam_adapter': {
        'description': 'Disable LRSA after the SAM encoder.',
        'tag_suffix': 'no_sam_adapter',
        'model_overrides': {
            'use_sam_adapter': False,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_confidence_fusion': {
        'description': 'Replace DRPF at decoder stages 4 and 3 with standard fusion.',
        'tag_suffix': 'no_conf_fusion',
        'model_overrides': {
            'decoder4_3_fusion': 'default',
        },
        'loss_overrides': {
            'confidence_weight': 0.0,
        },
        'augmentation_overrides': {},
    },
    'no_boundary_branch': {
        'description': 'Disable the explicit boundary branch and boundary loss.',
        'tag_suffix': 'no_boundary',
        'model_overrides': {
            'enable_boundary_head': False,
        },
        'loss_overrides': {
            'boundary_weight': 0.0,
        },
        'augmentation_overrides': {},
    },
    'no_uncertainty_supervision': {
        'description': 'Disable uncertainty heads and their auxiliary supervision.',
        'tag_suffix': 'no_uncertainty_supervision',
        'model_overrides': {
            'enable_uncertainty_heads': False,
        },
        'loss_overrides': {
            'uncertainty_weight': 0.0,
        },
        'augmentation_overrides': {},
    },
    'no_deep_supervision': {
        'description': 'Disable deep supervision at decoder stages 5, 4, and 3.',
        'tag_suffix': 'no_deep_supervision',
        'model_overrides': {},
        'loss_overrides': {
            'gt_branch_weight': 0.0,
        },
        'augmentation_overrides': {},
    },
    'no_mask_branch_supervision': {
        'description': 'Disable supervision of the prior-guided skip-mask branch.',
        'tag_suffix': 'no_mask_branch_supervision',
        'model_overrides': {},
        'loss_overrides': {
            'mask_branch_weight': 0.0,
        },
        'augmentation_overrides': {},
    },
    'static_rghpa': {
        'description': 'Replace dynamic CDBHA routing with static four-branch fusion.',
        'tag_suffix': 'static_rghpa',
        'model_overrides': {
            'rghpa_mode': 'static',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_rghpa_cab': {
        'description': 'Disable contrast-aware channel calibration in CDBHA.',
        'tag_suffix': 'no_rghpa_cab',
        'model_overrides': {
            'rghpa_use_cab': False,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_rghpa_boundary_gate': {
        'description': 'Disable the boundary gate and boundary-context path in CDBHA.',
        'tag_suffix': 'no_rghpa_boundary_gate',
        'model_overrides': {
            'rghpa_use_boundary_gate': False,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_rghpa_dynamic_routing': {
        'description': 'Use uniform branch weights while retaining channel calibration and boundary gating.',
        'tag_suffix': 'no_rghpa_no_router',
        'model_overrides': {
            'rghpa_use_dynamic_routing': False,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_rghpa_boundary_gate_no_cab': {
        'description': 'Disable both channel calibration and boundary gating in CDBHA.',
        'tag_suffix': 'no_rghpa_boundary_gate_no_cab',
        'model_overrides': {
            'rghpa_use_boundary_gate': False,
            'rghpa_use_cab': False,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_rghpa_boundary_gate_smoothed_routing': {
        'description': 'Disable boundary gating and use smoothed dynamic routing.',
        'tag_suffix': 'no_rghpa_boundary_gate_smooth_route',
        'model_overrides': {
            'rghpa_use_boundary_gate': False,
            'rghpa_router_temperature': 1.8,
            'rghpa_router_residual': 0.2,
            'rghpa_boundary_scale': 0.8,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'weak_boundary_gate': {
        'description': 'Use weak residual boundary gating in CDBHA.',
        'tag_suffix': 'weak_boundary_gate',
        'model_overrides': {
            'rghpa_use_boundary_gate': True,
            'rghpa_boundary_gate_strength': 0.35,
            'rghpa_boundary_gate_residual': 0.35,
            'rghpa_boundary_scale': 0.6,
            'rghpa_router_residual': 0.1,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'lite_routing': {
        'description': 'Use lightweight two-way routing without boundary gating.',
        'tag_suffix': 'lite_routing',
        'model_overrides': {
            'rghpa_use_boundary_gate': False,
            'rghpa_router_mode': 'lite',
            'rghpa_router_temperature': 1.25,
            'rghpa_router_residual': 0.1,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'weak_boundary_gate_lite_routing': {
        'description': 'Combine weak boundary gating with lightweight two-way routing.',
        'tag_suffix': 'weak_boundary_gate_lite_routing',
        'model_overrides': {
            'rghpa_use_boundary_gate': True,
            'rghpa_router_mode': 'lite',
            'rghpa_router_temperature': 1.25,
            'rghpa_router_residual': 0.1,
            'rghpa_boundary_gate_strength': 0.35,
            'rghpa_boundary_gate_residual': 0.35,
            'rghpa_boundary_scale': 0.6,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'rghpa_smoothed_routing': {
        'description': 'Smooth CDBHA routing with temperature scaling and uniform residual mixing.',
        'tag_suffix': 'rghpa_smooth_route',
        'model_overrides': {
            'rghpa_router_temperature': 1.5,
            'rghpa_router_residual': 0.15,
            'rghpa_boundary_scale': 0.85,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'busi_stable_rghpa': {
        'description': 'BUSI-oriented static routing configuration with reduced boundary injection.',
        'tag_suffix': 'busi_stable_rghpa',
        'model_overrides': {
            'rghpa_mode': 'static',
            'rghpa_use_dynamic_routing': False,
            'rghpa_boundary_scale': 0.8,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'busi_smoothed_rghpa': {
        'description': 'BUSI-oriented dynamic routing with temperature and residual smoothing.',
        'tag_suffix': 'busi_smoothed_rghpa',
        'model_overrides': {
            'rghpa_router_temperature': 1.8,
            'rghpa_router_residual': 0.2,
            'rghpa_boundary_scale': 0.8,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'drpf_discrepancy_aware_confidence': {
        'description': 'Attenuate DRPF confidence using discrepancy energy.',
        'tag_suffix': 'drpf_disc_conf',
        'model_overrides': {
            'drpf_discrepancy_penalty': 0.75,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'ultrasound_calibrated_full': {
        'description': 'Combine smoothed routing with discrepancy-aware DRPF confidence.',
        'tag_suffix': 'us_calibrated_full',
        'model_overrides': {
            'rghpa_router_temperature': 1.5,
            'rghpa_router_residual': 0.15,
            'rghpa_boundary_scale': 0.85,
            'drpf_discrepancy_penalty': 0.75,
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'shared_sam_decoder': {
        'description': 'Use a shared SAM-guided decoder at stages 5, 4, and 3.',
        'tag_suffix': 'shared_sam_decoder',
        'model_overrides': {
            'use_shared_sam_decoder': True,
            'shared_sam_decoder_dim': 64,
            'shared_sam_decoder_mode': 'full',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'residual_shared_sam_aux': {
        'description': 'Use a zero-initialized residual SAM auxiliary decoder at stages 5 and 4.',
        'tag_suffix': 'residual_shared_sam_aux',
        'model_overrides': {
            'use_shared_sam_decoder': True,
            'shared_sam_decoder_dim': 64,
            'shared_sam_decoder_mode': 'residual_aux',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'se_attention': {
        'description': 'Replace CDBHA with SE attention.',
        'tag_suffix': 'se_attention',
        'model_overrides': {
            'attention_module': 'se',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'cbam_attention': {
        'description': 'Replace CDBHA with CBAM attention.',
        'tag_suffix': 'cbam_attention',
        'model_overrides': {
            'attention_module': 'cbam',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'axial_attention': {
        'description': 'Replace CDBHA with axial attention.',
        'tag_suffix': 'axial_attention',
        'model_overrides': {
            'attention_module': 'axial',
        },
        'loss_overrides': {},
        'augmentation_overrides': {},
    },
    'no_targeted_aug': {
        'description': 'Disable lesion-focused and random scale-crop augmentation.',
        'tag_suffix': 'no_targeted_aug',
        'model_overrides': {},
        'loss_overrides': {},
        'augmentation_overrides': {
            'use_targeted_augmentation': False,
        },
    },
    'static_rghpa_no_targeted_aug': {
        'description': 'Use static CDBHA and disable targeted augmentation.',
        'tag_suffix': 'static_rghpa_no_targeted_aug',
        'model_overrides': {
            'rghpa_mode': 'static',
        },
        'loss_overrides': {},
        'augmentation_overrides': {
            'use_targeted_augmentation': False,
        },
    },
    'no_confidence_fusion_no_targeted_aug': {
        'description': 'Disable DRPF and targeted augmentation.',
        'tag_suffix': 'no_conf_fusion_no_targeted_aug',
        'model_overrides': {
            'decoder4_3_fusion': 'default',
        },
        'loss_overrides': {
            'confidence_weight': 0.0,
        },
        'augmentation_overrides': {
            'use_targeted_augmentation': False,
        },
    },
    'static_rghpa_no_confidence_fusion_no_targeted_aug': {
        'description': 'Use static CDBHA while disabling DRPF and targeted augmentation.',
        'tag_suffix': 'static_rghpa_no_conf_fusion_no_targeted_aug',
        'model_overrides': {
            'decoder4_3_fusion': 'default',
            'rghpa_mode': 'static',
        },
        'loss_overrides': {
            'confidence_weight': 0.0,
        },
        'augmentation_overrides': {
            'use_targeted_augmentation': False,
        },
    },
})


def list_ablation_names():
    return list(ABLATON_SPECS.keys())


def get_ablation_spec(name):
    if name not in ABLATON_SPECS:
        raise ValueError(f'Unknown ablation: {name}. Available: {", ".join(ABLATON_SPECS.keys())}')
    return copy.deepcopy(ABLATON_SPECS[name])
