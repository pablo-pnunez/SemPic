# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import DatasetClass

import numpy as np
import pandas as pd
from tqdm import tqdm


class BasicData(DatasetClass):

    def __init__(self, config):
        DatasetClass.__init__(self, config=config)

    def __drop_no_food__(self, rev, img):
        
        # Copia para evitar error mítico
        rev = rev.copy()

        # Obtener las que tienen y las que no comida
        img_no = img.loc[img.comida==0]
        img = img.loc[img.comida==1].reset_index(drop=True)
        
        # Eliminar las reviews que tienen todas sus fotos de "NO COMIDA"
        img_no_num = img_no.groupby("reviewId").image.count().reset_index(name="drop")
        img_no_num = img_no_num.merge(rev.loc[rev.reviewId.isin(img_no_num.reviewId.values)][["reviewId", "num_images"]], on="reviewId")
        img_no_num["delete"] = img_no_num["drop"] == img_no_num["num_images"]
        rev = rev.loc[~rev.reviewId.isin(img_no_num.loc[img_no_num.delete == True].reviewId.values)]
        
        # En las otras se actualiza el número de fotos
        for _, r in img_no_num.loc[img_no_num.delete==False].iterrows():
            current_no = rev.loc[rev.reviewId == r.reviewId].num_images.values[0]
            rev.loc[rev.reviewId == r.reviewId, "num_images"] = current_no - r["drop"]
        assert rev.num_images.sum() == len(img) # Verificar que hay las mismas fotos en ambos conjutos

        return rev, img

    def __drop_multiple_visits__(self, data):
        # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
        multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
        return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

    def __basic_filtering__(self):
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar imágenes
        img = pd.read_pickle(self.CONFIG["data_path"] + self.CONFIG["city"] + "_data/img-hd-densenet.pkl")
        img = img.astype({'review': 'int64'})

        # Cargar restaurantes
        res = pd.read_pickle(self.CONFIG["data_path"] + self.CONFIG["city"] + "_data/restaurants.pkl")
        res.rename(columns={"name": "rest_name"}, inplace=True)
        res = res.astype({'id': 'int64'})

        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + self.CONFIG["city"] + "_data/reviews.pkl")
        rev = rev.astype({'reviewId': 'int64', 'restaurantId': 'int64', 'rating': 'int64'})
        rev = rev.merge(res[["id","rest_name"]], left_on="restaurantId", right_on="id", how="left")
        rev["num_images"] = rev.images.apply(lambda x: len(x))
        rev["like"] = rev.rating.apply(lambda x: 1 if x > 30 else 0)

        # Añadir URL a imágenes
        img = img.merge(rev[["reviewId", "restaurantId", "images"]], left_on="review", right_on="reviewId", how="left")
        img["url"] = img.apply(lambda x: x.images[x.image]['image_url_lowres'], axis=1)
        img = img[["reviewId","restaurantId","image","url","vector","comida"]]

        # Eliminar columnas irrelevantes
        rev = rev[['reviewId', 'userId', 'restaurantId', 'like', 'rest_name', 'num_images']]

        # Eliminar reviews sin foto
        rev = rev.loc[rev.num_images>0]
        assert rev.num_images.sum() == len(img) # Verificar que hay las mismas fotos en ambos conjutos

        # Eliminar fotos que no sean de comida
        rev, img = self.__drop_no_food__(rev, img)

        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        rev = self.__drop_multiple_visits__(rev)
        img = img.loc[img.reviewId.isin(rev.reviewId)]
        assert rev.num_images.sum() == len(img) # Verificar que hay las mismas fotos en ambos conjutos

        return rev, img

