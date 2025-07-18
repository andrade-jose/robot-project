import imgaug.augmenters as iaa
import numpy as np

class MultiViewAugmentor:
    
    def __init__(self):
        # Transformações completas para RGB
        self.rgb_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Affine(rotate=(-45, 45), shear=(-10, 10), scale=(0.8, 1.2)),
            iaa.Multiply((0.8, 1.2)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),
        ], random_order=True)

        # Apenas transformações geométricas (sem ruído, brilho, contraste)
        self.geom_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Affine(rotate=(-45, 45), shear=(-10, 10), scale=(0.8, 1.2)),
        ], random_order=True)

    def augment_sample(self, sample):
        rgb = sample["rgb"]
        depth = sample["depth"]
        rgb_aug, depth_aug = [], []

        # Determinístico por amostra
        rgb_det = self.rgb_seq.to_deterministic()
        geom_det = self.geom_seq.to_deterministic()

        for i in range(rgb.shape[0]):
            # Aumenta RGB com transformação completa
            img_aug = rgb_det(image=(rgb[i] * 255).astype(np.uint8))
            rgb_aug.append(img_aug.astype(np.float32) / 255.0)

            # Aumenta DEPTH com transformação geométrica apenas
            dpt = (depth[i][..., 0] * 255).astype(np.uint8)
            dpt_aug = geom_det(image=dpt)
            dpt_aug = (dpt_aug.astype(np.float32) / 255.0)[..., np.newaxis]
            depth_aug.append(dpt_aug)

        sample["rgb"] = np.array(rgb_aug)
        sample["depth"] = np.array(depth_aug)
        return sample
