# -*- coding: utf-8 -*-
from src.datasets.SemPicData import SemPicData
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

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct

stage = 3 if args.stg is None else args.stg
model_v = "0" if args.mv is None else args.mv

pctg_usrs = .15 if args.pctg is None else args.pctg
gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info()))))
seed = 100 if args.sd is None else args.sd
l_rate = 1e-4 if args.lr is None else args.lr
n_epochs = 4000 if args.ep is None else args.ep
b_size = 128 if args.bs is None else args.bs

# Conjunto base ########################################################################################################

dts_cfg = {"city": city, "pctg_usrs": pctg_usrs, "seed": seed,
           "data_path": "/media/nas/pperez/data/TripAdvisor/", "save_path": "data/", "test_dev_split": .15}
sempic_dataset = SemPicData(dts_cfg)
# sempic_dataset.paper_stats()

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
    bst_cfg = {"gijon": "572b70d448a3c8818f027e1bed3e522f", "barcelona": "4e75d5f3b77170584a547b9027022de7", "madrid": "9e7d920b23cd26881da1ef7b1bf2502f",
                "paris":"256f645031d87bd342fe2e546dd3eda5", "newyorkcity":"cc035dbe22e40d60c30b250c162ddad6", "london":"c401fcdd50f685252bc7111338c4eb7b"}
    with open('models/SemPic/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)
    dts_cfg = best_cfg_data["dataset_config"]
    sempic_dataset = SemPicData(dts_cfg)
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

if stage == 4:
    pop_mdl = Popularity({"model": {"seed": seed}, "session": {"gpu": gpu, "in_md5": False}}, sempic_dataset)
    pop_mdl.test()
