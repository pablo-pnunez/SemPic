# -*- coding: utf-8 -*-
from src.datasets.SemPicPoiData import SemPicPoiData
from src.Common import to_pickle

import numpy as np
import pandas as pd


class SemPicPoiColdData(SemPicPoiData):
    """ Separación TRAIN/DEV/TEST con todos los restaurantes en TRAIN pero usuarios diferentes en DEV y TEST """

    def __init__(self, config):
        SemPicPoiData.__init__(self, config=config)

    def __to_img_per_row__(self, rev, img, max_usr_id):
        ret = []
       
        for id_r, rows in rev.groupby("restaurantId"):
            # Obtener imágenes del restaurante y las mezclamos (para que sean de DEV unas aleatorias)
            rst_imgs = img.loc[img.reviewId.isin(rows.reviewId)].sample(frac=1, random_state=self.CONFIG["seed"])
            
            # Cuantas van a dev
            to_dev = int(len(rst_imgs)*self.CONFIG["test_dev_split"])
            dv = np.zeros(len(rst_imgs), dtype=int)
            dv[:to_dev]=1

            # Crear el resto de columnas de los ejemplos
            r = [id_r] * len(rst_imgs)
            rn = [rows.rest_name.values[0]] * len(rst_imgs)
            x = rst_imgs.index.values
            y = [rows.loc[rows.userId<=max_usr_id, "userId"].tolist()]*len(rst_imgs) # En la salida, solo aquellos usuarios que son del X%
            z = rst_imgs.reviewId.values

            ret.extend(list(zip(r, rn, x, y, z, dv)))

        ret = pd.DataFrame(ret, columns=["restaurantId", "rest_name", "id_img", "output", "reviewId", "dev"]).sample(frac=1).reset_index(drop=True)
        return ret 

    def __split_dataset__(self, rev):

        # Obtener todos los usuarios
        all_usrs = rev.userId.unique()
        np.random.seed(self.CONFIG["seed"])
        np.random.shuffle(all_usrs)

        # Dividir 50/50 de forma aleatoria
        n_test_usrs = int(len(all_usrs)*(self.CONFIG["test_dev_split"]))
        usrs_test = all_usrs[:n_test_usrs]
        usrs_train = all_usrs[n_test_usrs:]
        assert len(usrs_test)+len(usrs_train) == len(all_usrs)

        # Obtener reviews de train/test y sus restaurantes
        train = rev.loc[rev.userId.isin(usrs_train)]
        test = rev.loc[rev.userId.isin(usrs_test)]

        # Mover todos los restaurantes a train
        train, test = self.__poi_to_train__(train, test)

        return train, test

    def get_data(self, load=["DENSENET_IMG_SIZE", "N_USR_PCTG", "N_USR", "N_RST", "TRAIN_DEV", "TEST", "IMG"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data

        # Si no existe, crear
        else:
            rev, img = self.__basic_filtering__()
            
            # Obtener nuevo id de restaurantes (se hace por actividad, pero no hace falta)
            rst_list = rev.groupby("restaurantId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_rst"})
            rst_list["new_id_rst"] = list(range(len(rst_list)))
            rev = rev.merge(rst_list).drop(columns=["restaurantId"]).rename(columns={"new_id_rst": "restaurantId"})
            img = img.merge(rst_list).drop(columns=["restaurantId"]).rename(columns={"new_id_rst": "restaurantId"})

            # Separar el conjunto en train/dev/test con todos los restaurantes en TRAIN pero usuarios diferentes en DEV y TEST
            train, test = self.__split_dataset__(rev)
            
            # Obtener id de usuario ordenado por actividad (más reviews)
            usr_list = train.groupby("userId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})
            usr_list["new_id_user"] = list(range(len(usr_list)))
            train = train.merge(usr_list).drop(columns=["userId"]).rename(columns={"new_id_user": "userId"})
            # Obtener, el número de usuario máximo dentro del "pctg_usrs"
            max_usr_id = int(len(train.userId.unique())*self.CONFIG["pctg_usrs"])

            # Cambiar en train de RVW por columna a IMG por columna y usuarios más activos y separar train en train/dev
            train_dev = self.__to_img_per_row__(train, img, max_usr_id)

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "DENSENET_IMG_SIZE", len(img.iloc[0].vector))
            to_pickle(self.DATASET_PATH, "N_USR_PCTG", max_usr_id)
            to_pickle(self.DATASET_PATH, "N_USR", len(rev.userId.unique()))
            to_pickle(self.DATASET_PATH, "N_RST", len(rev.restaurantId.unique()))

            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)
            to_pickle(self.DATASET_PATH, "IMG", img)

            return self.get_dict_data(self.DATASET_PATH, load)
