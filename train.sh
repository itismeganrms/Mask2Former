#!/bin/bash

#SBATCH --job-name=mask2former
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --output=/home/mrajaraman/slurm/mask2former/merged-train/output-%A.out
#SBATCH --gres=gpu:1

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
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# ---- TRAINING COMMANDS FOR ORIGINAL DATASET BELOW ----
# python train_net.py --num-gpus 1 \
# --exp_id ${TILE_SIZE} \
# --train_iter 2500 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
# DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# # MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \

# python train_net.py --num-gpus 1 \
# --exp_id ${TILE_SIZE} \
# --train_iter 5000 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
# DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# # MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \

# python train_net.py --num-gpus 1 \
# --exp_id ${TILE_SIZE} \
# --train_iter 25000 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
# --dataset_path /home/mrajaraman/dataset/coco-merged/ \
# OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# DATASETS.TRAIN "(\"dragonfly_${TILE_SIZE}_train\",)" \
# DATASETS.TEST "(\"dragonfly_${TILE_SIZE}_valid\",)"  \
# # MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \

# ---- TRAINING COMMANDS FOR MERGED DATASET BELOW ----
python train_net_merged.py --num-gpus 1 \
--exp_id ${TILE_SIZE} \
--train_iter 25000 \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--dataset_path /home/mrajaraman/dataset/dataset-v2-coco/ \
OUTPUT_DIR output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
DATASETS.TRAIN "(\"dragonfly_merged_train\",)" \
DATASETS.TEST "(\"dragonfly_merged_valid\",)"  \
# MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \