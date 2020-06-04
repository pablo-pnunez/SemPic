import argparse
import nvgpu

from src.datasets.semantics.OnlyFood import *
from src.datasets.semantics.OnlyFoodAndImages import *

from src.models.semantics.SemPic import *
from src.models.semantics.SemPic2 import *

########################################################################################################################

def cmdArgs():
    # Obtener argumentos por linea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, help="Test index")
    parser.add_argument('-c', type=str, help="City", )
    parser.add_argument('-s', type=str, help="Stage", )
    parser.add_argument('-e', type=str, help="Embedding", )
    parser.add_argument('-gpu', type=str, help="Gpu")
    args = parser.parse_args()
    return args

########################################################################################################################

args = cmdArgs()

stage = "test" if args.s is None else args.s
encoding = "emb" if args.e is None else args.e

gpu = 0
cfg_no = 0 if args.i is None else args.i
gpu = np.argmin(list(map(lambda x: x["mem_used_percent"],nvgpu.gpu_info()))) if args.gpu is None else args.gpu

city  = "gijon" if args.c is None else args.c
city = city.lower().replace(" ","")

# DATASETS #############################################################################################################

data_cfg  = {"city":city,"data_path":"/media/HDD/pperez/TripAdvisor/"+city+"_data/"}
#dts = OnlyFood(data_cfg)
dts = OnlyFoodAndImages(data_cfg)

#dts.dataset_stats()
#dts.test_baseline_hdp()
#dts.rest_most_popular()

# MODELS ###############################################################################################################

seeds = [100,12,8778,0,99968547,772,8002,4658,9,34785]
cfg_u = {"id":city+"_"+str(cfg_no),"pctg_usrs":.25, "learning_rate":5e-4, "epochs":500, "batch_size":2048, "gpu":gpu,"seed":seeds[cfg_no]}
mdl = SemPic(cfg_u, dts)

# STAGES ###############################################################################################################

if "train" in stage: mdl.train(save=True)
if "test"  in stage: mdl.test(encoding=encoding)
if "plot"  in stage: mdl.emb_tsne()
