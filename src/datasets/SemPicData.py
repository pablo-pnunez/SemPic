# -*- coding: utf-8 -*-
from src.datasets.BasicData import BasicData
from src.Common import to_pickle

import numpy as np
import pandas as pd


class SemPicData(BasicData):
    """ Separación TRAIN/DEV/TEST asegurando todos los usuarios y restaurantes en TRAIN"""

    def __init__(self, config):
        BasicData.__init__(self, config=config)

    def __rst_to_train__(self, train, test):
        # Obtener la lista de restaurantes de los dos conjuntos
        rsts_train = set(train.restaurantId.unique())
        rsts_test = set(test.restaurantId.unique())

        # Mirar cuales no están en train y si en test
        move_to_train = list(rsts_test-rsts_train)
        # Mover los usuarios al completo, no solo las reviews
        move_to_train_usrs = test.loc[test.restaurantId.isin(move_to_train)].userId.values
        train = train.append(test.loc[test.userId.isin(move_to_train_usrs)])
        test = test.loc[~test.userId.isin(move_to_train_usrs)]

        return train, test


    def __split_dataset__(self, rev):
        rvws_to_train, rvws_to_dev, rvws_to_test = [],[],[]

        # Todos los usuarios con menos de 3 reviews (no se pueden separar), a train directamente
        revs_per_user = rev.groupby("userId").agg(n_revs=("restaurantId","count")).reset_index()
        usrs_less_three = revs_per_user.loc[revs_per_user.n_revs<3, "userId"].to_list()
        rvws_to_train.extend(rev.loc[rev.userId.isin(usrs_less_three), "reviewId"].to_list())

        # Para el resto, hay que dividir utilizando el porcentaje de la configuración
        othr_revs = rev.loc[~rev.userId.isin(usrs_less_three)]
        
        for _, u in othr_revs.groupby("userId"):
            to_dev_test = int(np.ceil(len(u)*self.CONFIG["test_dev_split"]))
            to_train = len(u)-(2*to_dev_test)
            train, dev, test = np.split(u.sample(frac=1, random_state=self.CONFIG["seed"]), [to_train, to_train+to_dev_test])
            rvws_to_train.extend(train.reviewId.to_list())
            rvws_to_dev.extend(dev.reviewId.to_list())
            rvws_to_test.extend(test.reviewId.to_list())

        # Crear los conjuntos a partir de la separación de reviews
        train = rev.loc[rev.reviewId.isin(rvws_to_train)]
        dev = rev.loc[rev.reviewId.isin(rvws_to_dev)] 
        test = rev.loc[rev.reviewId.isin(rvws_to_test)].reset_index(drop=True)

        # Hasta ahora ya tenemos todos los usuarios en train, falta verificar que están también todos los restaurantes
        train, dev = self.__rst_to_train__(train, dev)
        train, test = self.__rst_to_train__(train, test)

        # Juntar train y dev
        train["dev"] = 0
        dev["dev"] = 1
        train_dev = train.append(dev).reset_index(drop=True)

        return train_dev, test


    def get_data(self, load=["DENSENET_IMG_SIZE", "N_USR_PCTG", "N_USR", "N_RST", "TRAIN_DEV", "TEST", "IMG"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data:
            return dict_data

        # Si no existe, crear
        else:
            rev, img = self.__basic_filtering__()
            
            # Obtener id de usuario ordenado por actividad (más reviews)
            usr_list = rev.groupby("userId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_user"})
            usr_list["new_id_user"] = list(range(len(usr_list)))
            rev = rev.merge(usr_list).drop(columns=["userId"]).rename(columns={"new_id_user": "userId"})
            # Obtener, el número de usuario máximo dentro del "pctg_usrs"
            max_usr_id = int(len(rev.userId.unique())*self.CONFIG["pctg_usrs"])

            # Obtener nuevo id de restaurantes (se hace por actividad, pero no hace falta)
            rst_list = rev.groupby("restaurantId").like.count().sort_values(ascending=False).reset_index().rename(columns={"like": "new_id_rst"})
            rst_list["new_id_rst"] = list(range(len(rst_list)))
            rev = rev.merge(rst_list).drop(columns=["restaurantId"]).rename(columns={"new_id_rst": "restaurantId"})
            img = img.merge(rst_list).drop(columns=["restaurantId"]).rename(columns={"new_id_rst": "restaurantId"})

            # Separar el conjunto en train/dev/test asegurando todos los usuarios y restaurantes en train
            train_dev, test = self.__split_dataset__(rev)

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "DENSENET_IMG_SIZE", len(img.iloc[0].vector))
            to_pickle(self.DATASET_PATH, "N_USR_PCTG", max_usr_id)
            to_pickle(self.DATASET_PATH, "N_USR", len(rev.userId.unique()))
            to_pickle(self.DATASET_PATH, "N_RST", len(rev.restaurantId.unique()))

            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)
            to_pickle(self.DATASET_PATH, "IMG", img)

            return self.get_dict_data(self.DATASET_PATH, load)
