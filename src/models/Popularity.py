# -*- coding: utf-8 -*-
from src.Common import print_g
from src.Metrics import precision, recall, f1
from src.sequences.BaseSequence import BaseSequence
from src.models.KerasModelClass import KerasModelClass

import time
import threading
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import tensorflow.keras.backend as K
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MultiLabelBinarizer


class Popularity(KerasModelClass):
    """ Baseline popularidad """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        return None

    def __perform_test__(self):

        ret = []

        # Obtener los restaurantes de train/dev ordenados por popularidad
        rest_pop = self.DATASET.DATA["TRAIN_DEV"].groupby("restaurantId").apply(lambda x: pd.Series({"n_reviews":len(x.reviewId.unique())})).reset_index()

        # Para cada usuario de test, obtener el restaurante más cercano
        for uid, udata in self.DATASET.DATA["TEST"].groupby("userId"):
            rsts = udata
            usr_rst_pos = rest_pop.loc[rest_pop.restaurantId.isin(udata.restaurantId.unique())]
            first_pos = usr_rst_pos.index[0]
            # Diccionario id_rest:posición
            usr_rst_pos = dict(zip(usr_rst_pos.restaurantId,usr_rst_pos.index))

            ret.append((uid, first_pos, usr_rst_pos, len(udata), udata.num_images.sum()))

        ret = pd.DataFrame(ret, columns=["userId","first_pos","usr_rst_pos","n_revs","n_imgs"])

        return ret
