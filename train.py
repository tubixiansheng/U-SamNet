import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config_setting import create_config, setting_config
from datasets.dataset import NPY_datasets
from engine import test_one_epoch, train_one_epoch, val_one_epoch
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from models.usamnet import USamNet
from utils import get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

warnings.filterwarnings('ignore')


def build_model(config):
    model_cfg = config.model_config
    if config.network != 'u_samnet':
        raise ValueError(f'Unsupported network: {config.network}')
    return USamNet(
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
        rghpa_mode=model_cfg.get('rghpa_mode', 'dynamic'),
        attention_module=model_cfg.get('attention_module', 'rghpa'),
        rghpa_use_cab=model_cfg.get('rghpa_use_cab'),
        rghpa_use_boundary_gate=model_cfg.get('rghpa_use_boundary_gate'),
        rghpa_use_dynamic_routing=model_cfg.get('rghpa_use_dynamic_routing'),
        rghpa_router_mode=model_cfg.get('rghpa_router_mode', 'full'),
        rghpa_router_temperature=model_cfg.get('rghpa_router_temperature', 1.0),
        rghpa_router_residual=model_cfg.get('rghpa_router_residual', 0.0),
        rghpa_boundary_scale=model_cfg.get('rghpa_boundary_scale', 1.0),
        rghpa_boundary_gate_strength=model_cfg.get('rghpa_boundary_gate_strength', 1.0),
        rghpa_boundary_gate_residual=model_cfg.get('rghpa_boundary_gate_residual', 0.0),
        drpf_discrepancy_penalty=model_cfg.get('drpf_discrepancy_penalty', 0.0),
        enable_boundary_head=model_cfg.get('enable_boundary_head', True),
        enable_uncertainty_heads=model_cfg.get('enable_uncertainty_heads', True),
        use_sam_adapter=model_cfg.get('use_sam_adapter', True),
        use_shared_sam_decoder=model_cfg.get('use_shared_sam_decoder', False),
        shared_sam_decoder_dim=model_cfg.get('shared_sam_decoder_dim', 64),
        shared_sam_decoder_mode=model_cfg.get('shared_sam_decoder_mode', 'full'),
    )


def build_summary(config, checkpoint_dir, best_epoch, best_loss, best_dsc_epoch, best_dsc, stopped_early, final_test_loss=None):
    best_checkpoint = None
    best_dsc_checkpoint = None
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        renamed_best = sorted(checkpoint_path.glob('best-epoch*-loss*.pth'))
        renamed_best_dsc = sorted(checkpoint_path.glob('best-dsc-epoch*-dsc*.pth'))
        if renamed_best:
            best_checkpoint = str(renamed_best[-1])
        else:
            raw_best = checkpoint_path / 'best.pth'
            if raw_best.exists():
                best_checkpoint = str(raw_best)
        if renamed_best_dsc:
            best_dsc_checkpoint = str(renamed_best_dsc[-1])
        else:
            raw_best_dsc = checkpoint_path / 'best_dsc.pth'
            if raw_best_dsc.exists():
                best_dsc_checkpoint = str(raw_best_dsc)
    return {
        'ablation_name': getattr(config, 'ablation_name', 'full'),
        'ablation_description': getattr(config, 'ablation_description', ''),
        'dataset': config.datasets,
        'data_path': config.data_path,
        'run_name': config.run_name,
        'work_dir': config.work_dir,
        'best_epoch': int(best_epoch),
        'best_loss': float(best_loss),
        'best_dsc_epoch': int(best_dsc_epoch),
        'best_dsc': float(best_dsc),
        'stopped_early': bool(stopped_early),
        'best_checkpoint': best_checkpoint,
        'best_dsc_checkpoint': best_dsc_checkpoint,
        'final_test_loss': None if final_test_loss is None else float(final_test_loss),
    }


def write_summary(summary_path, payload):
    if not summary_path:
        return
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open('w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
    )
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=config.num_workers,
        drop_last=True,
    )

    print('#----------Prepareing Model----------#')
    model = build_model(config).to(device)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    best_loss = 999.0
    best_dsc = -1.0
    start_epoch = 1
    best_epoch = 1
    best_dsc_epoch = 1
    no_improve_epochs = 0
    stopped_early = False

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_loss = checkpoint['min_loss']
        best_epoch = checkpoint['min_epoch']
        best_dsc = checkpoint.get('max_dsc', best_dsc)
        best_dsc_epoch = checkpoint.get('max_dsc_epoch', best_dsc_epoch)
        loss = checkpoint['loss']
        no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        logger.info(
            f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {best_loss:.4f}, '
            f'min_epoch: {best_epoch}, best_dsc: {best_dsc:.4f}, best_dsc_epoch: {best_dsc_epoch}, loss: {loss:.4f}'
        )

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
        )

        val_metrics = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
        )
        loss = val_metrics['loss']
        dsc = val_metrics['dsc']

        improvement = best_loss - loss
        if improvement > config.early_stopping_min_delta:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_loss = loss
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if dsc > best_dsc:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_dsc.pth'))
            best_dsc = dsc
            best_dsc_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': best_loss,
                'min_epoch': best_epoch,
                'max_dsc': best_dsc,
                'max_dsc_epoch': best_dsc_epoch,
                'loss': loss,
                'no_improve_epochs': no_improve_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth'),
        )

        if config.early_stopping and epoch >= config.min_epochs_before_stop and no_improve_epochs >= config.early_stopping_patience:
            stopped_early = True
            log_info = f'early stopping triggered at epoch {epoch}. best epoch: {best_epoch}, best loss: {best_loss:.4f}, no_improve_epochs: {no_improve_epochs}'
            print(log_info)
            logger.info(log_info)
            break

    final_test_loss = None
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        final_test_loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-loss{best_loss:.4f}.pth')
        )
        best_dsc_path = os.path.join(checkpoint_dir, 'best_dsc.pth')
        if os.path.exists(best_dsc_path):
            renamed_best_dsc_path = os.path.join(checkpoint_dir, f'best-dsc-epoch{best_dsc_epoch}-dsc{best_dsc:.4f}.pth')
            if os.path.exists(renamed_best_dsc_path):
                os.remove(renamed_best_dsc_path)
            shutil.move(best_dsc_path, renamed_best_dsc_path)
        if stopped_early:
            logger.info('training stopped early and best checkpoint has been renamed.')

    writer.close()
    summary = build_summary(
        config,
        checkpoint_dir,
        best_epoch,
        best_loss,
        best_dsc_epoch,
        best_dsc,
        stopped_early,
        final_test_loss=final_test_loss,
    )
    write_summary(getattr(config, 'summary_json', None), summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-SamNet and its registered ablation variants.')
    parser.add_argument('--ablation', default='current_best', help='Ablation name registered in configs/ablation_registry.py.')
    parser.add_argument('--dataset', help='Dataset name, e.g. BUSI or MosMedData.')
    parser.add_argument('--dataset-root', help='Optional dataset root override. Expected layout: train/|val/ with images/ and masks|labels.')
    parser.add_argument('--experiment-tag', help='Override experiment tag prefix.')
    parser.add_argument('--run-name', help='Override full run name.')
    parser.add_argument('--work-dir', help='Override work directory.')
    parser.add_argument('--gpu-id', help='GPU id string, e.g. 0 or 0,1.')
    parser.add_argument('--epochs', type=int, help='Override total epochs.')
    parser.add_argument('--batch-size', type=int, help='Override batch size.')
    parser.add_argument('--threshold', type=float, help='Override validation threshold.')
    parser.add_argument('--summary-json', help='Optional JSON file to persist training summary.')
    parser.add_argument('--max-train-batches', type=int, help='Limit train batches for smoke tests.')
    parser.add_argument('--max-val-batches', type=int, help='Limit val batches for smoke tests.')
    parser.add_argument('--max-test-batches', type=int, help='Limit test batches for smoke tests.')
    parser.add_argument('--disable-early-stopping', action='store_true', help='Disable early stopping and run all requested epochs.')
    parser.add_argument('--min-epochs-before-stop', type=int, help='Override the minimum epoch before early stopping can trigger.')
    parser.add_argument('--early-stopping-patience', type=int, help='Override early stopping patience.')
    parser.add_argument('--early-stopping-min-delta', type=float, help='Override the minimum validation-loss improvement for resetting early stopping.')
    parser.add_argument('--disable-train-visualize', action='store_true', help='Disable train diagnostics export.')
    parser.add_argument('--smoke-test', action='store_true', help='Run a 1-epoch limited-batch sanity pass instead of full training.')
    return parser.parse_args()


def config_from_args(args):
    extra_overrides = {'summary_json': args.summary_json}
    epochs = args.epochs
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    max_test_batches = args.max_test_batches
    train_visualize = None if not args.disable_train_visualize else False

    if args.disable_early_stopping:
        extra_overrides['early_stopping'] = False
    if args.min_epochs_before_stop is not None:
        extra_overrides['min_epochs_before_stop'] = args.min_epochs_before_stop
    if args.early_stopping_patience is not None:
        extra_overrides['early_stopping_patience'] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        extra_overrides['early_stopping_min_delta'] = args.early_stopping_min_delta

    if args.smoke_test:
        epochs = 1 if epochs is None else epochs
        max_train_batches = 1 if max_train_batches is None else max_train_batches
        max_val_batches = 1 if max_val_batches is None else max_val_batches
        max_test_batches = 1 if max_test_batches is None else max_test_batches
        train_visualize = False
        extra_overrides.update({
            'val_interval': 1,
            'save_interval': 100000,
            'min_epochs_before_stop': 1,
            'early_stopping_patience': 1,
            'print_interval': 1,
        })

    return create_config(
        ablation_name=args.ablation,
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        experiment_tag=args.experiment_tag,
        run_name=args.run_name,
        work_dir=args.work_dir,
        gpu_id=args.gpu_id,
        epochs=epochs,
        batch_size=args.batch_size,
        threshold=args.threshold,
        train_visualize=train_visualize,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
        extra_overrides=extra_overrides,
    )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(setting_config)
    else:
        main(config_from_args(parse_args()))
