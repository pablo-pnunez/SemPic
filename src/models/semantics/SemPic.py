# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd

from src.models.semantics.ModelSemantics import *
from src.Common import print_e

import keras
import keras.backend as K

from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from sklearn.preprocessing import MultiLabelBinarizer

########################################################################################################################

class SemPic(ModelSemantics):

    def __init__(self,config, dataset):
        ModelSemantics.__init__(self, config=config, dataset=dataset)

    def recall(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(self,y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def get_model(self):

        self.DATASET.DATA["N_USR"]= int( self.DATASET.DATA["N_USR"] * self.CONFIG["pctg_usrs"])

        input = Input(shape=(self.DATASET.DATA["V_IMG"],), name="in")
        x = BatchNormalization()(input)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu", name="img_emb")(x)
        x = BatchNormalization()(x)
        output = Dense(self.DATASET.DATA["N_USR"], activation="sigmoid")(x)
        opt = Adam(lr=self.CONFIG["learning_rate"])
        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[self.precision, self.recall, self.f1, "accuracy"])

        return model

    def train(self, save=False):

        if os.path.exists(self.MODEL_PATH):
            print_e("The model already exists...")
            return
        else:
            if save: os.makedirs(self.MODEL_PATH, exist_ok=True)

        # Conjuntos de entrenamiento
        train_sequence = self.Sequence(self)

        callbacks = []

        if os.path.exists(self.LOG_PATH):
            print_e("TensorBoard path already exists...")
            exit()

        if save:
            os.makedirs(self.LOG_PATH, exist_ok=True)
            tb_call = keras.callbacks.TensorBoard(log_dir=self.LOG_PATH, update_freq='epoch')
            mc = ModelCheckpoint(self.MODEL_PATH+"/weights", save_weights_only=True, save_best_only=True, monitor="loss")
            #mc = ModelCheckpoint(self.MODEL_PATH+"/weights", save_weights_only=True, save_best_only=True, monitor="f1",mode="max")

            callbacks.append(tb_call)
            callbacks.append(mc)

        es = EarlyStopping(patience=self.CONFIG['epochs'], monitor="loss", mode="min")
        #es = EarlyStopping(patience=40, monitor="f1", mode="max")
        callbacks.append(es)

        '''
        def cosine_decay(epoch):

            num_periods = 0.5
            alpha = 0.0
            beta = 0.001
            decay_steps = self.CONFIG["epochs"]
            global_step = epoch
            learning_rate = self.CONFIG["learning_rate"]

            linear_decay = (decay_steps - global_step) / decay_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * 2 * num_periods * global_step / decay_steps))
            decayed = (alpha + linear_decay) * cosine_decay + beta
            decayed_learning_rate = learning_rate * decayed

            tf.summary.scalar('learning_rate', decayed_learning_rate)

            return decayed_learning_rate

        lrs = LearningRateScheduler(cosine_decay)
        callbacks.append(lrs)
        
        '''


        self.MODEL.fit_generator(train_sequence,
                                  steps_per_epoch=train_sequence.__len__(),
                                  epochs=self.CONFIG['epochs'],
                                  verbose=2,
                                  #shuffle=True,
                                  workers=1,
                                  callbacks=callbacks,
                                  max_queue_size=25)

        K.clear_session()

    def test(self, encoding="", n_relevant=1 , previous_result=None):
        if "emb" in encoding:
            ModelSemantics.test(self,encoding=encoding, metric="dot", n_relevant=n_relevant , previous_result=previous_result)
        if "densenet" in encoding:
            ModelSemantics.test(self,encoding=encoding, metric="dist", n_relevant=n_relevant , previous_result=previous_result)

    def __get_image_encoding__(self, encoding="emb"):

        if "emb" in encoding:
            self.MODEL.load_weights(self.MODEL_PATH + "/weights")
            sub_model = Model(inputs=[self.MODEL.get_layer("in").input],outputs=[self.MODEL.get_layer("img_emb").output])
            all_img_embs = sub_model.predict(self.DATASET.DATA["IMG_VEC"], batch_size=self.CONFIG["batch_size"])
        if "densenet" in encoding:
            all_img_embs = self.DATASET.DATA["IMG_VEC"]

        return all_img_embs

    ####################################################################################################################

    class Sequence(Sequence):

        def __init__(self, model):

            self.MODEL = model
            self.N_RESTAURANTS = len(self.MODEL.DATASET.DATA["RST_ADY"])
            self.BATCH_SIZE = self.MODEL.CONFIG["batch_size"]

            self.init_data()

        def init_data(self):

            n_usrs = self.MODEL.DATASET.DATA["N_USR"]

            self.KHOT = MultiLabelBinarizer(classes=list(range(self.MODEL.DATASET.DATA["N_USR"])))

            x = []
            y = []

            for id_r, rows in tqdm(self.MODEL.DATASET.DATA['TRAIN'].groupby("id_restaurant"), desc="USRS DATA"):
                rst_imgs = self.MODEL.DATASET.DATA["TRAIN_RST_IMG"].loc[id_r]

                # Obtener usuarios
                rltd_usrs = rows.id_user.unique()
                rltd_usrs = rltd_usrs[np.argwhere(rltd_usrs < n_usrs).flatten()]

                x.extend(rst_imgs)
                y.extend([list(rltd_usrs)] * len(rst_imgs))

            ############################################################################################################

            ret = pd.DataFrame(list(zip(x, y)), columns=["id_img", "output"]).sample(frac=1)

            ############################################################################################################

            self.ALL_DATA = ret

            if (len(ret) > self.BATCH_SIZE):
                self.BATCH = np.array_split(ret, len(ret) // self.BATCH_SIZE)

            else:
                self.BATCH = np.array_split(ret, 1)

        def __len__(self):
            return len(self.BATCH)

        def __getitem__(self, idx):

            btch = self.BATCH[idx]

            x = self.MODEL.DATASET.DATA['IMG_VEC'][btch.id_img.values]
            y = self.KHOT.fit_transform(btch.output.values)

            return (x, y)

    ####################################################################################################################
