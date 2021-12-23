# -*- coding: utf-8 -*-
from genericpath import exists
from tensorflow.keras.utils import Sequence
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.training.tracking import base
from src.datasets.DatasetClass import DatasetClass
from src.sequences.Common import read_and_normalize_images

import re
import os
import ssl
import time
import socket
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from functools import partial
from multiprocessing.dummy import Pool
from urllib.request import urlretrieve


def __image_download__(row_data, base_save_path):
    socket.setdefaulttimeout(2)

    options = {0:"/photo-o/", 1:"/photo-s/", 2:"/photo-l/", 3:"/photo-w/", 4:"/photo-g/", 5:"/photo-a/", 6:"/photo-b/"}
    _, data = row_data

    path = f"{base_save_path}{data.restaurantId}/{data.reviewId}"
    os.makedirs(path, exist_ok=True)
    path = f"{path}/{data.imageId}.jpg"

    # Si está descargada, skip
    if (os.path.isfile(path)): return (path, True)

    downloaded = False
    attempt = 0

    while attempt<=6 and not downloaded:
        try:
            url = re.sub(r"/photo-./", options[attempt], data.imageUrl)
            url = url.replace(" ", "%20")
            # context = ssl.create_default_context()
            # a = urllib.request.urlopen(url, context=context, timeout=5)
            urlretrieve(url, path)
            downloaded = True
        except socket.timeout as e:
            retries = 5
            while retries>=0 and not downloaded:
                #print(f"Timeout ({retries} left)")
                try:
                    urlretrieve(url, path)
                    downloaded = True
                except socket.timeout as e:
                    # Si
                    time.sleep(1)
                except Exception:
                    # Si no funciona el link por otra cosa, salir a bucle de fuera
                    retries = 0 
                retries-=1

        except Exception as e:
            print(url, attempt, e)
        
        attempt+=1

    if not downloaded:
        print(url, attempt, path)
        return (path, False)


class BasicPoiData(DatasetClass):

    def __init__(self, config):
        DatasetClass.__init__(self, config=config)

    def __img_download__(self, data):

        pool = Pool(processes=48) # 8
        prt_fn = partial(__image_download__, base_save_path="/media/nas/pois/tripadvisor_pois/images/"+self.CONFIG["city"]+"/")
        ret = pool.map_async(prt_fn, data.iterrows())       

        total = int(np.ceil(len(data)/ret._chunksize))
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

        return ret

    def __dense_predict__(self, img):

        img_path ="/media/nas/pois/tripadvisor_pois/images/"
        save_path = f"{img_path}{self.CONFIG['city']}_densenet.npy"

        if not os.path.exists(save_path):
            dense_model = tf.keras.applications.DenseNet121(include_top=True, weights="imagenet", input_shape=None)
            dense_model = tf.keras.models.Model(inputs=[dense_model.input], outputs=[dense_model.get_layer("avg_pool").output])
            
            preds = []
            batches = np.array_split(img, 15)

            for batch in batches:
                dense_model_sequence = DenseNetSequence(batch, batch_size=32, img_path = f'{img_path}{self.CONFIG["city"]}')
                pred = dense_model.predict(dense_model_sequence, verbose=1, workers=24, max_queue_size=25)
                preds.extend(pred.tolist())
            
            preds = np.row_stack(preds)
            np.save(save_path, preds)
        else:
            preds = np.load(save_path)

        return preds

    def __drop_multiple_visits__(self, data):
        # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
        multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
        return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

    def __load_raw_data__(self):
        
        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + self.CONFIG["city"] + "/df_"+self.CONFIG["city"]+".pickle").reset_index(drop=True)
        rev = rev.rename(columns={"idPOI":"restaurantId", "namePOI":"rest_name"})
        rev = rev.astype({'restaurantId': 'int64', 'reviewId': 'int64', 'rating': 'int64'})
        rev["num_images"] = rev.imageId.apply(lambda x: len(x))
        rev["like"] = rev.rating.apply(lambda x: 1 if x > 3 else 0)
        
        return rev

    def __basic_filtering__(self):
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""
        # Cargar datos en crudo
        rev = self.__load_raw_data__()

        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        rev = self.__drop_multiple_visits__(rev)

        # Eliminar reviews sin foto
        rev = rev.loc[rev.num_images>0].reset_index(drop=True)

        # Los vectores dense de las imágenes hay que generarlos
        rev_img = rev.copy()
        rev_img['reviewId'] = rev_img.apply(lambda x: [x.reviewId]*x.num_images,1)
        rev_img['restaurantId'] = rev_img.apply(lambda x: [x.restaurantId]*x.num_images,1)

        img = pd.DataFrame(zip(np.concatenate(rev_img.reviewId.values), np.concatenate(rev_img.restaurantId.values), np.concatenate(rev_img.imageId.values), np.concatenate(rev_img.imageUrl.values)), columns=["reviewId", "restaurantId", "imageId", "imageUrl"])
        img = img.astype({'imageId': 'int64', 'imageUrl': str})
        img["imageUrl"] = img["imageUrl"].apply(lambda x: x.replace("https","http"))

        # Descargar imágenes
        dwnl_imgs = self.__img_download__(img)
        assert len(dwnl_imgs)==len(img)

        # Predecir vectores densenet
        img_vec = self.__dense_predict__(img)
        img["vector"] = img_vec.tolist()

        # Eliminar columnas irrelevantes
        rev = rev[['reviewId', 'userId', 'restaurantId', 'like', 'rest_name', 'imageId', 'num_images']]

        return rev, img

class DenseNetSequence(Sequence):

    def __init__(self, data, batch_size,  img_path):
        Sequence.__init__(self)
        
        self.DATA = data
        self.BATCH_SIZE = batch_size
        self.IMG_PATH = img_path

        if len(self.DATA) > self.BATCH_SIZE:
            self.BATCHES = np.array_split(self.DATA, len(self.DATA) // self.BATCH_SIZE)
        else:
            self.BATCHES = np.array_split(self.DATA, 1)

    def __len__(self):
        return len(self.BATCHES)

    def __getitem__(self, idx):
        batch_data = self.BATCHES[idx]
        paths = batch_data.apply(lambda x: f"/{x.restaurantId}/{x.reviewId}/{x.imageId}.jpg", 1)
        input_data = read_and_normalize_images(paths, img_shape=(224, 224), base_path=self.IMG_PATH)
        return input_data