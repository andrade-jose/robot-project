import imgaug.augmenters as iaa
import numpy as np
import pandas as pd

class MultiViewAugmentor:
    
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Affine(rotate=(-45, 45), shear=(-10, 10), scale=(0.8, 1.2)),
            iaa.Multiply((0.8, 1.2)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
        ], random_order=True)

    def augment_sample(self, sample):
        rgb = sample["rgb"]
        depth = sample["depth"]
        rgb_aug, depth_aug = [], []

        for i in range(rgb.shape[0]):
            det = self.seq.to_deterministic()
            img_aug = det(image=(rgb[i] * 255).astype(np.uint8))
            dpt_aug = det(image=(depth[i][..., 0] * 255).astype(np.uint8))

            rgb_aug.append(img_aug.astype(np.float32) / 255.0)
            depth_aug.append((dpt_aug.astype(np.float32) / 255.0)[..., np.newaxis])

        sample["rgb"] = np.array(rgb_aug)
        sample["depth"] = np.array(depth_aug)
        return sample
