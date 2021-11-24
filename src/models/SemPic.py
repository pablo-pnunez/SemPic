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
import tensorflow as tf
from multiprocessing import Pool
import tensorflow.keras.backend as K
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MultiLabelBinarizer


''' Tiene que estar a este nivel para poder hacer multiproceso'''
def __user_test__(user_data, all_img_embs, train_dev_images, emb):
        # tss = time.time()
        uid, r = user_data

        relevant = r["restaurantId"].unique()
        n_revs = len(r.reviewId.unique())
        n_imgs = r.num_images.sum()

        # Obtener el idx de las imágenes del usuario
        img_idxs = np.concatenate(r.img_idx.values)

        # Obtener la imagen de train_dev más cercana al centroide de las del usuario
        mean_img = np.mean(all_img_embs[img_idxs], axis=0)

        if emb == "sem":
            dists = cdist([mean_img], all_img_embs[train_dev_images.index.values], metric=np.dot)[0]
            rst_train_dists = train_dev_images.iloc[np.argsort(-dists)].reset_index(drop=True)        
        else:
            dists = cdist([mean_img], all_img_embs[train_dev_images.index.values], metric="euclidean")[0]
            rst_train_dists = train_dev_images.iloc[np.argsort(dists)].reset_index(drop=True)

        # Eliminar restaurantes repetidos, si no el top no es justo con el Filtro colaborativo 
        # first_pos = rst_train_dists.loc[rst_train_dists.restaurantId.isin(relevant)].index[0]
        rst_train_dists = rst_train_dists.drop_duplicates("restaurantId").reset_index(drop=True)
        first_pos = rst_train_dists.loc[rst_train_dists.restaurantId.isin(relevant)].index[0]

        # print(uid, time.time()-tss)
        return(uid, first_pos, n_revs, n_imgs)
        

class SemPic(KerasModelClass):
    """ SemPic img => rst(usrs) """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def create_weighted_binary_crossentropy(self, zero_weight, one_weight):
        def weighted_binary_crossentropy(y_true, y_pred):
            y_true = K.cast(y_true, dtype=tf.float32)

            # Calculate the binary crossentropy
            b_ce = K.binary_crossentropy(y_true, y_pred)

            # Apply the weights
            weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
            weighted_b_ce = weight_vector * b_ce

            # Return the mean error
            return K.mean(weighted_b_ce)
        return weighted_binary_crossentropy

    def get_model(self):
        
        mv = int(self.CONFIG["model"]["model_version"])

        if mv==0 : model = self.__get_model_0__()
        elif mv==1 : model = self.__get_model_1__()
        else: exit()

        return model
    
    def __get_model_0__(self):
        
        input = tf.keras.layers.Input(shape=(self.DATASET.DATA["DENSENET_IMG_SIZE"],), name="in")
        x = input
        
        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.Dropout(.6)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.4)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.Dense(128, name="img_emb")(x)
        output = tf.keras.layers.Dense(self.DATASET.DATA["N_USR_PCTG"], activation="sigmoid")(x)
        opt = tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = tf.keras.models.Model(inputs=[input], outputs=[output])
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[f1])
        model.compile(optimizer=opt, loss=self.create_weighted_binary_crossentropy(1,5), metrics=[f1])

        return model

    def __get_model_1__(self):
        
        input = tf.keras.layers.Input(shape=(self.DATASET.DATA["DENSENET_IMG_SIZE"],), name="in")
        x = input
        
        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.Dropout(.6)(x)
        x = tf.keras.layers.Dense(600, activation="relu")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.4)(x)
        x = tf.keras.layers.Dense(400, activation="relu")(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(.2)(x)
        x = tf.keras.layers.Dense(200, name="img_emb")(x)
        output = tf.keras.layers.Dense(self.DATASET.DATA["N_USR_PCTG"], activation="sigmoid")(x)
        opt = tf.keras.optimizers.Adam(lr=self.CONFIG["model"]["learning_rate"])
        model = tf.keras.models.Model(inputs=[input], outputs=[output])
        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[f1])
        model.compile(optimizer=opt, loss=self.create_weighted_binary_crossentropy(1,5), metrics=[f1])

        return model

    def get_train_dev_sequences(self):

        train = SemPicSequence(self, is_dev=0)
        dev = SemPicSequence(self, is_dev=1)

        return train, dev

    def evaluate(self, test=False, train=False):

        if test:
            test_set = SemPicSequence(self, set_name="TEST")
        elif train:
            test_set = SemPicSequence(self, is_dev=0)
        else:
            test_set = SemPicSequence(self, is_dev=1)

        ret = self.MODEL.evaluate(test_set, verbose=0)

        print_g(dict(zip(self.MODEL.metrics_names, ret)))     

    def __perform_test__(self, emb):

        ret = []
        all_img_embs = None

        img_data = self.DATASET.DATA["IMG"].copy()

        # Cargar los vectores de imágenes correspondientes
        if emb == "sem" : all_img_embs = np.load(self.MODEL_PATH+"img_embs.npy").astype(np.float16)
        else: all_img_embs = np.row_stack(img_data["vector"].values).astype(np.float16)
        
        # Eliminar columnas innecesarias y pesadas
        img_data = img_data.drop(columns=["vector"])

        # Imágenes de train_dev
        train_dev_images =  img_data.loc[img_data.reviewId.isin(self.DATASET.DATA["TRAIN_DEV"].reviewId.values)].copy()
        train_dev_images["index"] = train_dev_images.index.values       
        
        # Para facilitar el test, se añaden los idx de las imágenes de los usuarios de test
        test_data = self.DATASET.DATA["TEST"].copy()
        test_data["img_idx"] = self.DATASET.DATA["TEST"].reviewId.apply(lambda x: img_data.loc[img_data.reviewId==x].index.values)
        users = test_data.userId.unique()

        pool = Pool(processes=8) # 8
        prt_fn = partial(__user_test__, all_img_embs=all_img_embs, train_dev_images=train_dev_images, emb=emb)
        ret = pool.map_async(prt_fn, test_data.groupby("userId"))       

        total = int(np.ceil(len(users)/ret._chunksize))
        pbar = tqdm(total=total)

        while not ret.ready():
            pbar.n = total-ret._number_left
            pbar.last_print_n = total-ret._number_left
            pbar.refresh()
            ret.wait(timeout=1)
        pbar.n = total
        pbar.last_print_n = total
        pbar.refresh()
        pbar.close()

        ret = ret.get()
        ret = pd.DataFrame(ret, columns=["userId","first_pos","n_revs","n_imgs"])

        return ret

    def test(self, **arguments):
        KerasModelClass.test(self, filename="test_results_"+arguments["emb"], **arguments)


class SemPicSequence(BaseSequence):

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

        # Primero obtener para cada restaurante de train (están todos), los usuarios del x%
        rst_output = self.MODEL.DATASET.DATA["TRAIN_DEV"].groupby("restaurantId").apply(lambda x: x.userId.values[x.userId.values<self.MODEL.DATASET.DATA["N_USR_PCTG"]])
        rst_output = rst_output.reset_index().rename(columns={0:"users"})
        rst_output["n_users"] = rst_output.users.apply(lambda x: len(x))
        rst_output = rst_output.loc[rst_output.n_users>0].reset_index(drop=True)

        # Cambiar de RVW por columna a IMG por columna
        ret = []
        for id_r, rows in tqdm(rev.groupby("restaurantId"), desc="Sequence data"):
            # Si el restaurante no tiene imágenes se salta
            if id_r not in rst_output.restaurantId.unique(): continue
            # Obtener imágenes del restaurante
            rst_imgs = self.MODEL.DATASET.DATA["IMG"].loc[self.MODEL.DATASET.DATA["IMG"].reviewId.isin(rows.reviewId)].index.values
            assert len(rst_imgs)>0 # Asegurarse de que siempre tiene imágenes...
           
            # Crear ejemplos
            r = [id_r] * len(rst_imgs)
            rn = [rows.rest_name.values[0]] * len(rst_imgs)
            x = rst_imgs
            y = [list(rst_output.loc[rst_output.restaurantId==id_r, "users"].tolist()[0])] * len(rst_imgs) # En la salida, solo aquellos usuarios que son del 25%

            ret.extend(list(zip(r, rn, x, y)))

        ret = pd.DataFrame(ret, columns=["id_restaurant", "rest_name", "id_img", "output"]).sample(frac=1)

        return ret
        
    def preprocess_input(self, batch_data):
        return np.row_stack(self.MODEL.DATASET.DATA['IMG'].loc[batch_data.id_img.values, "vector"].values)

    def preprocess_output(self, batch_data):
        return self.KHOT.fit_transform(batch_data.output.values)
