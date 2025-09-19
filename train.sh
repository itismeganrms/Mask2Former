#!/bin/bash

#SBATCH --job-name=mask2former
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/home/mrajaraman/slurm/mask2former/train/output-%A.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"


echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

source activate /home/mrajaraman/conda/mask2former

# Debugging outputs
pwd
which conda
python --version
# pip freeze

# LazyConfig Training Script - pretrained new baseline
TILE_SIZE=512

python train_net.py --num-gpus 1 \
--resume \
--exp_id ${TILE_SIZE} \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
OUTPUT_DIR output_lifeplan_b_${TILE_SIZE}_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters \
DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \