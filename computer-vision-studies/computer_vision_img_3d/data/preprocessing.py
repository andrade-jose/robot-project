import albumentations as A
import numpy as np
import cv2
from typing import Dict

class MultiViewAugmentor:
    def __init__(self, rgb_aug: bool = True, depth_aug: bool = True):
        """
        Args:
            rgb_aug: Se True, aplica aumentos completos em imagens RGB
            depth_aug: Se True, aplica aumentos geométricos em imagens Depth
        """
        self.rgb_aug = rgb_aug
        self.depth_aug = depth_aug

        # Augmentações para RGB (geométricas + fotométricas)
        self.rgb_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(rotate=(-45, 45), shear=(-10, 10), scale=(0.8, 1.2)),
            A.OneOf([
                A.Multiply((0.8, 1.2), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(0.0, 25.0), p=0.5)
            ], p=0.5)
        ])

        # Apenas transformações geométricas para Depth
        self.depth_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(rotate=(-45, 45), shear=(-10, 10), scale=(0.8, 1.2))
        ])

    def _validate_sample(self, sample: Dict) -> None:
        if not isinstance(sample, dict):
            raise ValueError("Input must be a dictionary with 'rgb' and 'depth' keys")
        if 'rgb' not in sample or 'depth' not in sample:
            raise ValueError("Sample must contain both 'rgb' and 'depth' keys")
        if sample['rgb'].shape[0] != sample['depth'].shape[0]:
            raise ValueError("Number of RGB and Depth views must match")

    def augment_sample(self, sample: Dict) -> Dict:
        self._validate_sample(sample)

        rgb = sample["rgb"]
        depth = sample["depth"]

        # Normaliza para uint8
        rgb = rgb.astype(np.float32)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)

        depth = depth.astype(np.float32)
        if depth.max() <= 1.0:
            depth = (depth * 255).astype(np.uint8)

        rgb_aug = []
        depth_aug = []

        for i in range(rgb.shape[0]):
            # Aplica mesma seed para manter coerência entre RGB e Depth
            seed = np.random.randint(0, 10000)

            if self.rgb_aug:
                A.seed_everything(seed)
                transformed_rgb = self.rgb_transform(image=rgb[i])["image"]
                rgb_aug.append(transformed_rgb.astype(np.float32) / 255.0)
            else:
                rgb_aug.append(rgb[i].astype(np.float32) / 255.0)

            if self.depth_aug:
                A.seed_everything(seed)
                # Remove canal para aplicar com albumentations (que espera HxW para cinza)
                dpt = depth[i][..., 0]
                transformed_depth = self.depth_transform(image=dpt)["image"]
                depth_aug.append(transformed_depth[..., np.newaxis].astype(np.float32) / 255.0)
            else:
                depth_aug.append(depth[i].astype(np.float32) / 255.0)

        sample["rgb"] = np.stack(rgb_aug)
        sample["depth"] = np.stack(depth_aug)
        return sample
