#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --output=mlp_6x6.%j.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate warp

python main.py \
--use_cuda \
--out_file mlp_6x6.P \
--use_images \
--model_name mlp \
--n_steps 2000 \
--test_every 100 \
--analyze_every 100 \
--training_regime ungrouped \
--image_dir images/faces36 \
--grid_size 6 \
--inner_4x4