import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from configs.config_setting import create_config, setting_config
from models.usamnet import USamNet
from utils import save_visualization_bundle


def build_model(config, device):
    model_cfg = config.model_config
    model = USamNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        c_list=model_cfg['c_list'],
        bridge=model_cfg['bridge'],
        gt_ds=model_cfg['gt_ds'],
        img_size=config.input_size_h,
        sam_model=model_cfg.get('sam_model', 'b'),
        sam_checkpoint=model_cfg.get('sam_checkpoint'),
        sam_freeze=model_cfg.get('sam_freeze', True),
        decoder4_3_fusion=model_cfg.get('decoder4_3_fusion', 'drpf'),
        cdbha_mode=model_cfg.get('cdbha_mode', 'dynamic'),
        attention_module=model_cfg.get('attention_module', 'cdbha'),
        cdbha_use_cab=model_cfg.get('cdbha_use_cab'),
        cdbha_use_boundary_gate=model_cfg.get('cdbha_use_boundary_gate'),
        cdbha_use_dynamic_routing=model_cfg.get('cdbha_use_dynamic_routing'),
        cdbha_router_mode=model_cfg.get('cdbha_router_mode', 'full'),
        cdbha_router_temperature=model_cfg.get('cdbha_router_temperature', 1.0),
        cdbha_router_residual=model_cfg.get('cdbha_router_residual', 0.0),
        cdbha_boundary_scale=model_cfg.get('cdbha_boundary_scale', 1.0),
        cdbha_boundary_gate_strength=model_cfg.get('cdbha_boundary_gate_strength', 1.0),
        cdbha_boundary_gate_residual=model_cfg.get('cdbha_boundary_gate_residual', 0.0),
        drpf_discrepancy_penalty=model_cfg.get('drpf_discrepancy_penalty', 0.0),
        enable_boundary_head=model_cfg.get('enable_boundary_head', True),
        enable_uncertainty_heads=model_cfg.get('enable_uncertainty_heads', True),
        use_sam_adapter=model_cfg.get('use_sam_adapter', True),
        use_shared_sam_decoder=model_cfg.get('use_shared_sam_decoder', False),
        shared_sam_decoder_dim=model_cfg.get('shared_sam_decoder_dim', 64),
        shared_sam_decoder_mode=model_cfg.get('shared_sam_decoder_mode', 'full'),
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_image(config, image_path):
    image = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)
    dummy_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
    image_tensor, _ = config.test_transformer((image, dummy_mask))
    return image_tensor.unsqueeze(0)


def collect_images(input_path, max_images=None):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    image_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_paths = sorted([path for path in input_path.iterdir() if path.suffix.lower() in image_suffixes])
    if max_images is not None:
        return image_paths[:max_images]
    return image_paths


def save_mask(mask_tensor, output_path, threshold):
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    binary_mask = (mask >= threshold).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(output_path)


def unwrap_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get('prediction'), outputs.get('visual_maps'), outputs.get('extras', {})
    if isinstance(outputs, tuple):
        return outputs[-1], None, {}
    return outputs, None, {}


def build_mask_map(mask_dir):
    mask_dir = Path(mask_dir)
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    mask_map = {}
    for path in mask_dir.iterdir():
        if path.suffix.lower() not in suffixes:
            continue
        stem = path.stem[:-5] if path.stem.endswith('_mask') else path.stem
        mask_map[stem] = path
    return mask_map


def infer_default_mask_dir(input_path):
    input_path = Path(input_path)
    parent = input_path.parent if input_path.is_file() else input_path
    if parent.name == 'images':
        for candidate_name in ['masks', 'labels']:
            candidate = parent.parent / candidate_name
            if candidate.exists():
                return candidate
    return None


def load_gt_mask(mask_path, target_shape):
    mask = Image.open(mask_path).convert('L')
    if mask.size != (target_shape[1], target_shape[0]):
        mask = mask.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return (np.array(mask).astype(np.float32) / 255.0) >= 0.5


def compute_metrics(probabilities, targets, threshold):
    prediction = probabilities >= threshold
    target = targets >= 0.5
    tp = np.logical_and(prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    precision = tp / (tp + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    dsc = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    fg_iou = tp / (tp + fp + fn + 1e-8)
    bg_iou = tn / (tn + fp + fn + 1e-8)
    binary_miou = 0.5 * (fg_iou + bg_iou)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        'threshold': float(threshold),
        'miou': float(fg_iou),
        'fg_iou': float(fg_iou),
        'bg_iou': float(bg_iou),
        'binary_miou': float(binary_miou),
        'dsc': float(dsc),
        'f1_or_dsc': float(dsc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def write_summary(summary_path, payload):
    if not summary_path:
        return
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open('w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Run U-SamNet inference and optional metric evaluation.')
    parser.add_argument('--ablation', default='current_best', help='Ablation name registered in configs/ablation_registry.py.')
    parser.add_argument('--dataset', help='Dataset name, e.g. BUSI or MosMedData.')
    parser.add_argument('--weights', required=True, help='Path to best.pth or latest.pth.')
    parser.add_argument('--input', required=True, help='Input image path or directory.')
    parser.add_argument('--output-dir', required=True, help='Directory to save predicted masks.')
    parser.add_argument('--threshold', type=float, default=setting_config.threshold, help='Binary mask threshold.')
    parser.add_argument('--gpu-id', help='GPU id string, e.g. 0 or 0,1.')
    parser.add_argument('--visualize', action='store_true', help='Export original image, prediction, module heatmaps and overlays.')
    parser.add_argument('--mask-dir', help='Optional ground-truth mask directory for threshold scan and metric evaluation.')
    parser.add_argument('--scan-thresholds', action='store_true', help='Scan thresholds on validation masks and save a JSON summary.')
    parser.add_argument('--scan-start', type=float, default=0.3, help='Threshold scan start.')
    parser.add_argument('--scan-end', type=float, default=0.7, help='Threshold scan end.')
    parser.add_argument('--scan-step', type=float, default=0.02, help='Threshold scan step.')
    parser.add_argument('--max-images', type=int, help='Only run the first N images, useful for smoke tests.')
    parser.add_argument('--summary-json', help='Optional JSON file to persist inference summary.')
    args = parser.parse_args()

    config = create_config(ablation_name=args.ablation, dataset_name=args.dataset, gpu_id=args.gpu_id, threshold=args.threshold)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(build_model(config, device), args.weights, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.input, max_images=args.max_images)
    if not image_paths:
        raise FileNotFoundError(f'No images found in {args.input}')

    resolved_mask_dir = Path(args.mask_dir) if args.mask_dir else infer_default_mask_dir(args.input)
    mask_map = build_mask_map(resolved_mask_dir) if resolved_mask_dir is not None else None
    probability_bank = []
    target_bank = []
    metrics_path = None
    scan_path = None
    metrics = None
    best_result = None

    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = preprocess_image(config, image_path).to(device)
            outputs = model(image_tensor, visualize=args.visualize)
            prediction, visual_maps, _ = unwrap_outputs(outputs)
            output_path = output_dir / f'{image_path.stem}_mask.png'
            save_mask(prediction, output_path, args.threshold)
            if args.visualize:
                sample_root = output_dir / image_path.stem
                save_visualization_bundle(image_tensor, prediction, visual_maps, sample_root, image_path.stem, threshold=args.threshold)
            if mask_map is not None and image_path.stem in mask_map:
                probability_bank.append(prediction.squeeze().detach().cpu().numpy())
                target_bank.append(load_gt_mask(mask_map[image_path.stem], prediction.shape[2:]))
            print(f'saved: {output_path}')

    if probability_bank and target_bank:
        probability_bank = np.stack(probability_bank, axis=0)
        target_bank = np.stack(target_bank, axis=0)
        metrics = compute_metrics(probability_bank, target_bank, args.threshold)
        metrics_path = output_dir / 'inference_metrics.json'
        with metrics_path.open('w', encoding='utf-8') as file_obj:
            json.dump({'mask_dir': str(resolved_mask_dir), 'metrics': metrics}, file_obj, ensure_ascii=False, indent=2)
        print(f"threshold={metrics['threshold']:.2f}, dsc={metrics['dsc']:.6f}, fg_iou={metrics['fg_iou']:.6f}, binary_miou={metrics['binary_miou']:.6f}, precision={metrics['precision']:.6f}, sensitivity={metrics['sensitivity']:.6f}")
        print(f"metrics saved: {metrics_path}")

        if args.scan_thresholds:
            thresholds = np.arange(args.scan_start, args.scan_end + 1e-8, args.scan_step)
            scan_results = [compute_metrics(probability_bank, target_bank, threshold) for threshold in thresholds]
            best_result = max(scan_results, key=lambda item: (item['dsc'], item['sensitivity']))
            scan_payload = {
                'evaluated_threshold': metrics,
                'best_threshold': best_result,
                'scan_results': scan_results,
            }
            scan_path = output_dir / 'threshold_scan.json'
            with scan_path.open('w', encoding='utf-8') as file_obj:
                json.dump(scan_payload, file_obj, ensure_ascii=False, indent=2)
            print(f"best threshold={best_result['threshold']:.2f}, best dsc={best_result['dsc']:.6f}, best sensitivity={best_result['sensitivity']:.6f}, saved: {scan_path}")
    else:
        print('no ground-truth masks found, metrics were not computed')

    summary = {
        'ablation_name': args.ablation,
        'weights': args.weights,
        'input': args.input,
        'output_dir': str(output_dir),
        'metrics_path': None if metrics_path is None else str(metrics_path),
        'threshold_scan_path': None if scan_path is None else str(scan_path),
        'evaluated_metrics': metrics,
        'best_threshold': best_result,
    }
    write_summary(args.summary_json, summary)
    return summary


if __name__ == '__main__':
    main()
