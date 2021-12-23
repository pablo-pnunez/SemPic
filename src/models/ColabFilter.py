# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf


class ColabFilter(KerasModelClass):
    """ Filtro colaborativo, usr, rst => dot """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()

        user_in = tf.keras.layers.Input(shape=(1,), name="user_id")
        rest_in = tf.keras.layers.Input(shape=(1,), name="rest_in")

        k=0
        if mv == "0": k=64

        user_embs = tf.keras.layers.Embedding(self.DATASET.DATA["N_USR"], k, embeddings_regularizer=tf.keras.regularizers.l2(self.CONFIG["model"]["l2_nu"]), name="user_embs")
        rest_embs = tf.keras.layers.Embedding(self.DATASET.DATA["N_RST"], k, embeddings_regularizer=tf.keras.regularizers.l2(self.CONFIG["model"]["l2_nu"]), name="rest_embs")
        
        usr_emb = tf.keras.layers.Flatten(name="usr_emb")(user_embs(user_in))
        rst_emb = tf.keras.layers.Flatten(name="rst_emb")(rest_embs(rest_in))      

        '''
        dot_prd = tf.keras.layers.Dot(axes=1, name="dot_prod")([usr_emb, rst_emb])
        out = tf.keras.layers.Activation("sigmoid", name="output")(dot_prd)
        '''
        
        concat = tf.keras.layers.Concatenate(name="concat")([usr_emb, rst_emb])
        h1 = tf.keras.layers.Dense(k, activation = "relu", name="h1")(concat)
        out = tf.keras.layers.Dense(1, activation = "sigmoid", name="output")(h1)
        
        '''
        concat = tf.keras.layers.Concatenate(name="concat")([usr_emb, rst_emb])
        out = tf.keras.layers.Dense(1, activation = "sigmoid", name="output")(concat)
        '''
        model = tf.keras.Model(inputs=[user_in, rest_in], outputs=[out])
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"]), metrics=['accuracy'])

        return model

    def get_train_dev_sequences(self):

        train = ColabFilterSequence(self, is_dev=0)
        dev = ColabFilterSequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False, train=False):

        if test:
            test_set = ColabFilterSequence(self, set_name="TEST")
        elif train:
            test_set = ColabFilterSequence(self, is_dev=0)
        else:
            test_set = ColabFilterSequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))

    def __perform_test__(self):
        ret = []

        # Cargar los pesos del modelo
        self.MODEL.load_weights(self.MODEL_PATH+"weights")

        # Predecir, para todos los usuarios de test, y todos los restaurantes, sus valoraciones
        nrsts = self.DATASET.DATA["N_RST"]
        usrs  = self.DATASET.DATA["TEST"].userId.unique()
        user_list = np.repeat(usrs, nrsts)
        rest_list = np.tile(range(nrsts), len(usrs))
        pred_list = self.MODEL.predict([user_list, rest_list], batch_size=8192, workers=2, verbose=1).flatten()
        user_rst_val = pd.DataFrame(zip(user_list, rest_list, pred_list), columns=["userId", "restaurantId", "val"])

        # Para cada usuario del test...
        for uid, r in tqdm(self.DATASET.DATA["TEST"].groupby("userId")):
            relevant = r["restaurantId"].unique()
            n_revs = len(r.reviewId.unique())
            n_imgs = r.num_images.sum()

            u_r_v = user_rst_val.loc[user_rst_val.userId==uid].sort_values("val", ascending=False).reset_index(drop=True)
            usr_rst_pos = u_r_v.loc[u_r_v.restaurantId.isin(relevant)]
            first_pos = usr_rst_pos.index[0]
            # Diccionario id_rest:posición
            usr_rst_pos = dict(zip(usr_rst_pos.restaurantId,usr_rst_pos.index))

            ret.append((uid, first_pos, usr_rst_pos, n_revs, n_imgs))

        ret = pd.DataFrame(ret, columns=["userId","first_pos","usr_rst_pos","n_revs","n_imgs"])

        return ret


class ColabFilterSequence(BaseSequence):

    def __init__(self, model, set_name="TRAIN_DEV", is_dev=-1):
        self.IS_DEV = is_dev
        self.SET_NAME = set_name
        self.EPOCH = 0
        BaseSequence.__init__(self, parent_model=model)

    def init_data(self):
        rev = self.MODEL.DATASET.DATA[self.SET_NAME]

        if self.IS_DEV >= 0:
            rev = rev.loc[rev["dev"] == self.IS_DEV]

        # Todos son positivos en este punto
        rev = rev.copy()
        rev["output"] = 1 

        # Crear ejemplos negativos (solo cuando procede)
        if self.SET_NAME == "TRAIN_DEV" and self.IS_DEV==0:
            neg_samples = []
            
            '''
            # Crear ejemplos negativos por restaurante
            for rid, rdt in rev.groupby("restaurantId"):
                posible_usrs = list(set(range(self.MODEL.DATASET.DATA["N_USR"]))-set(rdt.userId.unique()))
                selected_usrs = np.random.choice(posible_usrs, len(rdt)) # No se fija la semilla para variar en cada epoch los ejemplos
                neg_samples.extend(list(zip([rid]*len(rdt),selected_usrs, np.zeros(len(rdt), dtype=int))))
            '''
            
            # Crear ejemplos negativos por usuario
            for uid, udt in rev.groupby("userId"):
                nneg = len(udt)
                posible_rsts = list(set(range(self.MODEL.DATASET.DATA["N_RST"]))-set(udt.restaurantId.unique()))
                selected_rsts = np.random.choice(posible_rsts, nneg) # No se fija la semilla para variar en cada epoch los ejemplos
                neg_samples.extend(list(zip(selected_rsts, [uid]*nneg, np.zeros(nneg, dtype=int))))
            

            neg_samples = pd.DataFrame(neg_samples, columns=["restaurantId", "userId", "output"])
            rev = rev[["restaurantId", "userId", "output"]].append(neg_samples)
            rev = rev.sample(frac=1)

        return rev

    def on_epoch_end(self):
        
        # Rehacer negativos cada 5 epoch
        if self.EPOCH>0 and self.EPOCH%5==0:

            self.MODEL_DATA = self.init_data()

            if len(self.MODEL_DATA) > self.BATCH_SIZE:
                self.BATCHES = np.array_split(self.MODEL_DATA, len(self.MODEL_DATA) // self.BATCH_SIZE)
            else:
                self.BATCHES = np.array_split(self.MODEL_DATA, 1)

        self.EPOCH += 1
        
    def preprocess_input(self, batch_data):
        return [batch_data["userId"].values, batch_data["restaurantId"].values]

    def preprocess_output(self, batch_data):
        return batch_data["output"].values
