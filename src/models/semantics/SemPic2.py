# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd

from src.models.semantics.SemPic import *

########################################################################################################################

class SemPic2(SemPic):

    def __init__(self,config, dataset):
        SemPic.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        self.DATASET.DATA["N_USR"]= int( self.DATASET.DATA["N_USR"] * self.CONFIG["pctg_usrs"])

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(128, name="img_emb")(x)
        x = BatchNormalization()(x)
        output = Dense(self.DATASET.DATA["N_USR"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

