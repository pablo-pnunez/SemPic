# -*- coding: utf-8 -*-
from tensorflow.keras.utils import Sequence
import numpy as np


class BaseSequence(Sequence):
    """ Esqueleto de una secuencia."""

    def __init__(self, parent_model):

        self.MODEL = parent_model  # Modelo de keras que contiene todos los datos de configuración
        self.BATCH_SIZE = self.MODEL.CONFIG["model"]["batch_size"]
        self.MODEL_DATA = self.init_data()

        if len(self.MODEL_DATA) > self.BATCH_SIZE:
            self.BATCHES = np.array_split(self.MODEL_DATA, len(self.MODEL_DATA) // self.BATCH_SIZE)
        else:
            self.BATCHES = np.array_split(self.MODEL_DATA, 1)

    def init_data(self):
        return NotImplementedError

    def preprocess_input(self, batch_data):
        return NotImplementedError

    def preprocess_output(self, batch_data):
        return NotImplementedError

    def __len__(self):
        return len(self.BATCHES)

    def __getitem__(self, idx):
        batch_data = self.BATCHES[idx]

        input_data = self.preprocess_input(batch_data)
        output_data = self.preprocess_output(batch_data)

        return input_data, output_data


