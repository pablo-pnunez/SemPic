# -*- coding: utf-8 -*-
from src.sequences.BaseSequence import BaseSequence
from src.models.SemPic import SemPic
from src.Common import print_g

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import MultiLabelBinarizer


class SemPicCold(SemPic):
    """ SemPic img => rst(usrs) """
    def __init__(self, config, dataset):
        SemPic.__init__(self, config=config, dataset=dataset)

    def get_train_dev_sequences(self):

        train = SemPicColdSequence(self, is_dev=0)
        dev = SemPicColdSequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False, train=False):

        if test:
            test_set = SemPicColdSequence(self, set_name="TEST")
        elif train:
            test_set = SemPicColdSequence(self, is_dev=0)
        else:
            test_set = SemPicColdSequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))     


class SemPicColdSequence(BaseSequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        self.EPOCH = 0
        BaseSequence.__init__(self, parent_model=model)

    def init_data(self):
        self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_USR_PCTG"])))
        rev = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            rev = rev.loc[rev["dev"] == self.IS_DEV]

        return rev.sample(frac=1)
        
    def preprocess_input(self, batch_data):
        return np.row_stack(self.MODEL.DATASET.DATA['IMG'].loc[batch_data.id_img.values, "vector"].values)

    def preprocess_output(self, batch_data):
        return self.KHOT.fit_transform(batch_data.output.values)
