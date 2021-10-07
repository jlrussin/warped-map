#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --output=mlp_ctx_scale0p5.%j.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate warp

python main.py \
--use_cuda \
--out_file mlp_ctx_scale0p5.P \
--use_images \
--model_name mlp \
--ctx_scale 0.5