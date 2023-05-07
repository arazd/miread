#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mincpus=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --partition=t4v2
#SBATCH --qos=high
#SBATCH --exclude=gpu107
#SBATCH --output=./slurm-%A_%a.out
#SBATCH --error=./slurm-%A_%a.err
#SBATCH --open-mode=append

CHECKPOINTS=(
  "DistilBART_full_pubmed"
)

# set up checkpointing
#ckpt=$PWD/runs/checkpoints_${SLURM_JOB_ID}_${CHECKPOINTS[SLURM_ARRAY_TASK_ID]}
log_file=$PWD/logs/NLP_${SLURM_JOB_ID}_${CHECKPOINTS[SLURM_ARRAY_TASK_ID]}.log

#ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $ckpt
#touch $ckpt/DELAYPURGE

export PATH=/pkgs/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pkgs/cudnn-10.1-v7.6.3.30/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pkgs/TensorRT-6/lib:$LD_LIBRARY_PATH
export PATH=/h/anastasia/anaconda3/bin:$PATH
source activate nlp_env


HPARAMS=(
   "--coef_lm=1 --coef_cls=15 --lr=1e-6 --num_epochs=298 --checkpoint_dir=./runs/DistilBART_full_pubmed/" #  --dataset_df_path df_q1.csv
)

#cmd="python train_distributed.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
#      --num_epoch 1 --checkpoint_dir $ckpt --log_file $log_file" # --num_epoch 298

cmd="accelerate launch --num_processes 2 nlp_example.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} --log_file $log_file " # --num_epoch 298

echo $cmd
eval $cmd
