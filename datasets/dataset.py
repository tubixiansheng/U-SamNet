from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path


def _normalize_stem(path):
    stem = path.stem
    if stem.endswith('_mask'):
        stem = stem[:-5]
    return stem


def _build_pairs(image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_files = sorted([path for path in image_dir.iterdir() if path.is_file() and not path.name.startswith('.')])
    mask_files = sorted([path for path in mask_dir.iterdir() if path.is_file() and not path.name.startswith('.')])

    mask_map = {_normalize_stem(path): path for path in mask_files}
    pairs = []
    for image_path in image_files:
        mask_path = mask_map.get(_normalize_stem(image_path))
        if mask_path is not None:
            pairs.append([str(image_path), str(mask_path)])

    if not pairs:
        raise RuntimeError(f'No matched image-mask pairs found in {image_dir} and {mask_dir}')

    return pairs


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        image_dir_name = getattr(config, 'image_dir_name', 'images')
        mask_dir_name = getattr(config, 'mask_dir_name', 'masks')
        if train:
            self.data = _build_pairs(path_Data + f'train/{image_dir_name}/', path_Data + f'train/{mask_dir_name}/')
            self.transformer = config.train_transformer
        else:
            self.data = _build_pairs(path_Data + f'val/{image_dir_name}/', path_Data + f'val/{mask_dir_name}/')
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
