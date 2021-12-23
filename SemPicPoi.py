# -*- coding: utf-8 -*-
from src.datasets.SemPicPoiData import SemPicPoiData
from src.models.ColabFilter import ColabFilter
from src.models.Popularity import Popularity
from src.Common import parse_cmd_args
from src.models.SemPic import SemPic

import os
import json
import nvgpu
import numpy as np
import tensorflow as tf

########################################################################################################################

args = parse_cmd_args()

city = "Madrid".replace(" ", "") if args.ct is None else args.ct

stage = 3 if args.stg is None else args.stg
model_v = "0" if args.mv is None else args.mv

pctg_usrs = .15 if args.pctg is None else args.pctg
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 1e-4 if args.lr is None else args.lr
n_epochs = 4000 if args.ep is None else args.ep
b_size = 128 if args.bs is None else args.bs
l2_nu = 1e-4 if args.l2 is None else args.l2

# Conjunto base ########################################################################################################

dts_cfg = {"city": city, "pctg_usrs": pctg_usrs, "seed": seed,
           "data_path": "/media/nas/pois/tripadvisor_pois/DATA_byList/", "save_path": "data/", "test_dev_split": .15}
sempic_dataset = SemPicPoiData(dts_cfg)
#sempic_dataset.paper_stats()

exit()
# SemPic ###############################################################################################################

sempic_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/10, 
                              "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                              "early_st_monitor": "val_f1", "early_st_monitor_mode": "max", 
                              "early_st_first_epoch": 30, "early_st_patience": 20},
                   "session": {"gpu": gpu, "in_md5": False}}

if stage == 0:
    sempic_mdl = SemPic(sempic_cfg, sempic_dataset)
    sempic_mdl.train(dev=True, save_model=True)
    # sempic_mdl.evaluate(test=False)

if stage == 1 or stage==2  or stage==3:
    bst_cfg = {"Barcelona": "90c0110aa217ecf024ebbfc28a6d139a","NYC":"28bb1fef71ff5a20736305e1c10ca3a6", "London":"c2e4f24beec2c82856b03091797508b7"}
    with open('models/SemPic/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)
    dts_cfg = best_cfg_data["dataset_config"]
    sempic_dataset = SemPicPoiData(dts_cfg)
    sempic_cfg["model"] = best_cfg_data["model"]
    sempic_mdl = SemPic(sempic_cfg, sempic_dataset)
    sempic_mdl.train(dev=False, save_model=True)
    # sempic_mdl.evaluate(test=True)

    if stage == 2: sempic_mdl.test(emb="dense")
    if stage == 3: 
        if not os.path.exists(sempic_mdl.MODEL_PATH+"img_embs.npy"):
            sempic_mdl.MODEL.load_weights(sempic_mdl.MODEL_PATH+"weights")
            all_img_embs = tf.keras.models.Model(inputs=[sempic_mdl.MODEL.get_layer("in").input], outputs=[sempic_mdl.MODEL.get_layer("img_emb").output])
            all_img_embs = all_img_embs.predict(np.row_stack(sempic_mdl.DATASET.DATA["IMG"]["vector"]), verbose=1, batch_size=8192, workers=5)
            np.save( sempic_mdl.MODEL_PATH+"img_embs.npy", all_img_embs)
        else:
            sempic_mdl.test(emb="sem")

# Colab filter #########################################################################################################

colab_filter_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/10, 
                              "epochs": n_epochs, "batch_size": b_size, "seed": seed, "l2_nu":l2_nu,
                              "early_st_monitor": "val_accuracy", "early_st_monitor_mode": "max", 
                              "early_st_first_epoch": 0, "early_st_patience": 10},
                   "session": {"gpu": gpu, "in_md5": False}}
if stage == 4:
    colab_filter_mdl = ColabFilter(colab_filter_cfg, sempic_dataset)
    colab_filter_mdl.train(dev=True, save_model=True)
    colab_filter_mdl.evaluate(test=False)

if stage == 5 :
    bst_cfg = {"Barcelona": "9a48c87e433bb6c6dc63b9f2ed337f2d","NYC":"936905857bf4ac65a367fd82ed49f05b", "London":"26fe3b09916f2aea0c930d1fad7ebf37"}
    with open('models/ColabFilter/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)
    dts_cfg = best_cfg_data["dataset_config"]
    sempic_dataset = SemPicPoiData(dts_cfg)
    colab_filter_cfg["model"] = best_cfg_data["model"]
    colab_filter_mdl = ColabFilter(colab_filter_cfg, sempic_dataset)
    colab_filter_mdl.train(dev=False, save_model=True)
    
    colab_filter_mdl.test()

