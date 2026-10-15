import numpy as np
import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs, save_visualization_bundle
from sklearn.metrics import precision_score


def _unwrap_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get('gt_pre', ()), outputs.get('masks', ()), outputs.get('prediction'), outputs.get('visual_maps'), outputs.get('extras', {})
    if isinstance(outputs, tuple) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2], None, {}
    return (), (), outputs, None, {}


def _maybe_save_train_diagnostics(model, images, outputs, epoch, iteration, config):
    if not getattr(config, 'train_visualize', False):
        return
    if iteration != 0:
        return
    interval = max(1, getattr(config, 'train_visualize_interval', 1))
    if epoch % interval != 0:
        return

    _, _, prediction, visual_maps, _ = _unwrap_outputs(outputs)
    if visual_maps is None:
        with torch.no_grad():
            outputs = model(images[:1], visualize=True)
        _, _, prediction, visual_maps, _ = _unwrap_outputs(outputs)

    max_samples = min(getattr(config, 'train_visualize_samples', 1), images.size(0))
    save_root = os.path.join(config.work_dir, 'train_diagnostics', f'epoch_{epoch:03d}')
    for sample_index in range(max_samples):
        sample_visual_maps = {key: value[sample_index:sample_index + 1] for key, value in (visual_maps or {}).items()}
        save_visualization_bundle(
            images[sample_index:sample_index + 1],
            prediction[sample_index:sample_index + 1],
            sample_visual_maps,
            save_root,
            f'iter_{iteration:04d}_sample_{sample_index}',
            threshold=config.threshold,
        )

def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()
    device = next(model.parameters()).device

    loss_list = []
    max_train_batches = getattr(config, 'max_train_batches', None)

    for iter, data in enumerate(train_loader):
        if max_train_batches is not None and iter >= max_train_batches:
            break
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        outputs = model(images)
        gt_pre, masks, out, _, extras = _unwrap_outputs(outputs)
        loss = criterion(gt_pre, masks, out, targets, extras=extras)

        loss.backward()
        optimizer.step()

        _maybe_save_train_diagnostics(model, images, outputs, epoch, iter, config)

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    device = next(model.parameters()).device
    preds = []
    gts = []
    loss_list = []
    max_val_batches = getattr(config, 'max_val_batches', None)
    with torch.no_grad():
        for batch_index, data in enumerate(tqdm(test_loader)):
            if max_val_batches is not None and batch_index >= max_val_batches:
                break
            img, msk = data
            img = img.to(device, non_blocking=True).float()
            msk = msk.to(device, non_blocking=True).float()

            gt_pre, masks, out, _, extras = _unwrap_outputs(model(img))
            loss = criterion(gt_pre, masks, out, msk, extras=extras)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=config.threshold, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
    precision = precision_score(y_true, y_pre)
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    metrics = {
        'loss': float(np.mean(loss_list)),
        'miou': float(miou),
        'dsc': float(f1_or_dsc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
    }

    if epoch % config.val_interval == 0:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                precision: {precision}, specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return metrics


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    device = next(model.parameters()).device
    preds = []
    gts = []
    loss_list = []
    max_test_batches = getattr(config, 'max_test_batches', None)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if max_test_batches is not None and i >= max_test_batches:
                break
            img, msk = data
            img = img.to(device, non_blocking=True).float()
            msk = msk.to(device, non_blocking=True).float()

            gt_pre, masks, out, _, extras = _unwrap_outputs(model(img))
            loss = criterion(gt_pre, masks, out, msk, extras=extras)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        precision = precision_score(y_true, y_pre)
        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                precision: {precision}, specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
