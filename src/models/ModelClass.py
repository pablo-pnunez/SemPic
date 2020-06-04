# -*- coding: utf-8 -*-
import random
import os

import numpy as np
import tensorflow as tf

class ModelClass:

    def __init__(self,config,dataset):
        self.CONFIG     = config
        self.DATASET    = dataset

        self.MODEL_NAME = self.__class__.__name__
        self.CUSTOM_PATH = self.MODEL_NAME+"/"+self.CONFIG["id"]+"/"

        self.MODEL_PATH = "models/"+self.CUSTOM_PATH
        self.LOG_PATH = "logs/"+self.CUSTOM_PATH

        # Fijar las semillas de numpy y TF
        np.random.seed(self.CONFIG["seed"])
        random.seed(self.CONFIG["seed"])
        tf.random.set_seed(self.CONFIG["seed"])

        # Seleccionar la GPU más adecuada y limitar el uso de memoria
        self.__config_session__()

        #Crear el modelo
        self.MODEL = self.get_model()

    def get_model(self):
        raise NotImplementedError

    def train(self, save=False):
        raise NotImplementedError

    def __config_session__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CONFIG["gpu"])

        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)