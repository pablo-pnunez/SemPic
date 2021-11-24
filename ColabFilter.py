# -*- coding: utf-8 -*-
from src.Common import parse_cmd_args
from src.datasets.SemPicData import SemPicData
from src.models.ColabFilter import ColabFilter

import json
import nvgpu
import numpy as np

########################################################################################################################

args = parse_cmd_args()

city = "gijon".lower().replace(" ", "") if args.ct is None else args.ct

stage = 2 if args.stg is None else args.stg
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
           "data_path": "/media/nas/pperez/data/TripAdvisor/", "save_path": "data/", "test_dev_split": .15}
sempic_dataset = SemPicData(dts_cfg)

# Filtro colaborativo ##################################################################################################

colab_filter_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/10, 
                              "epochs": n_epochs, "batch_size": b_size, "seed": seed, "l2_nu":l2_nu,
                              "early_st_monitor": "val_accuracy", "early_st_monitor_mode": "max", 
                              "early_st_first_epoch": 0, "early_st_patience": 10},
                   "session": {"gpu": gpu, "in_md5": False}}

if stage == 0:
    colab_filter_mdl = ColabFilter(colab_filter_cfg, sempic_dataset)
    colab_filter_mdl.train(dev=True, save_model=True)
    colab_filter_mdl.evaluate(test=False)

if stage == 1 or stage==2 :
    bst_cfg = {"gijon": "3c37f82f3038cfb650e4bdc0dad75c83", "barcelona": "5724e29ce20c4a10243a940e0c80fa95", "madrid": "704c59b77e4623f1bfe70cd52ce793ff", 
               "paris":"eb520343aa55165425952033d52e4bba", "newyorkcity":"525303a9d421643d426d445018cfffb3", "london":"73048b8de08d0a034772457317b71139"}
    with open('models/ColabFilter/%s/%s/cfg.json' % (city, bst_cfg[city])) as f: best_cfg_data = json.load(f)
    dts_cfg = best_cfg_data["dataset_config"]
    sempic_dataset = SemPicData(dts_cfg)
    colab_filter_cfg["model"] = best_cfg_data["model"]
    colab_filter_mdl = ColabFilter(colab_filter_cfg, sempic_dataset)
    colab_filter_mdl.train(dev=False, save_model=True)
    # colab_filter_mdl.evaluate(test=True)

    if stage == 2: colab_filter_mdl.test()

