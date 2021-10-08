#!/bin/bash

# Defaults
sbatch scripts/mlp.sh
sbatch scripts/rnn.sh
sbatch scripts/rnn_ctx_last.sh

# Balance
sbatch scripts/rnn_balance.sh
sbatch scripts/mlp_balance.sh

# Step-wise MLP
sbatch scripts/stepmlp.sh

# Truncated RNN
sbatch scripts/trunc_rnn.sh
sbatch scripts/trunc_rnn_ctx_last.sh

# MLP: context scale
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p1.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p2.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p3.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p4.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p5.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p6.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p7.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p8.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale0p9.sh
sbatch scripts/mlp_ctx_scale/mlp_ctx_scale1p0.sh

# RNN: context scale
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p1.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p2.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p3.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p4.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p5.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p6.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p7.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p8.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale0p9.sh
sbatch scripts/rnn_ctx_scale/rnn_ctx_scale1p0.sh

# Grad norm
sbatch scripts/mlp_grad_norm.sh 
sbatch scripts/rnn_grad_norm.sh 

# 6x6 grid
sbatch scripts/mlp_6x6.sh 
sbatch scripts/rnn_6x6.sh 

# Step-by-step
sbatch scripts/mlp_sbs.sh
sbatch scripts/rnn_sbs.sh




