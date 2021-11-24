# -*- coding: utf-8 -*-

from src.sequences import BaseSequence
from src.sequences.Common import read_and_normalize_images

import imgaug.augmenters as iaa


class BaseImageSequence(BaseSequence):
    """ Esqueleto secuencia de imágenes: Retorna solamente la imagen, sin nunguna salida."""

    def __init__(self, parent_model, data_aug=False):
        BaseSequence.__init__(self, parent_model=parent_model)

        if data_aug:
            self.AUGMENTER = iaa.Sequential([iaa.Fliplr(0.5),
                                             iaa.Flipud(0.2),
                                             iaa.Affine(shear=(-16, 16), mode="wrap"),
                                             iaa.Affine(rotate=(-45, 45), mode="wrap"),
                                             iaa.Grayscale(alpha=(0.0, .5)),
                                             iaa.LinearContrast((0.4, 1.6)),
                                             iaa.PerspectiveTransform(scale=(0.01, 0.15)),
                                             iaa.pillike.EnhanceSharpness()
                                             ])

    def preprocess_input(self, btch):
        # read_and_normalize_images(btch.path.values, base_path=self.DATA["img_path"], img_shape=self.DATA["img_shape"][:-1], augmenter=self.AUGMENTER)
        raise NotImplementedError
