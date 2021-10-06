import argparse
import torch 
import pickle
import random
import numpy as np 

from dataset import get_loaders
from models import get_model
from train import train
from analyze import analyze

parser = argparse.ArgumentParser()
# Setup
parser.add_argument('--use_cuda', action='store_true',
                    help='Use GPU, if available')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--print_every', type=int, default=200,
                    help='Number of steps before printing average loss')
parser.add_argument('--analyze_every', type=int, default=50,
                    help='Number of steps before running analyses')
parser.add_argument('--out_file', default='results.P')
parser.add_argument('--verbose', action='store_true',
                    help='Lots of printing (e.g., every analysis performed')
# Dataset
parser.add_argument('--use_images', action='store_true',
                    help='Use full face images and CNN for cortical system')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--training_regime', default='grouped', 
                    choices=['grouped', 'ungrouped', 'train_all', 'balanced'],
                    help='Structure of the training set')
parser.add_argument('--grid_size', type=int, default=4,
                    help='Length of one side of grid')                    
parser.add_argument('--ctx_order', type=str, default='first',
                    help='Present context first or last') 
# Training
parser.add_argument('--n_runs', type=int, default=2,
                    help='Number of runs for cortical system')
parser.add_argument('--n_steps', type=int, default=6000,
                    help='Number of steps for training cortical system')
parser.add_argument('--bs', type=int, default=1,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
# Model
parser.add_argument('--model_name', default='rnn',
                    choices=['mlp','rnn','step_mlp','mlp_cc','trunc_rnn'], 
                    help='Type of model to use')
parser.add_argument('--trunc_mlp', action='store_true',
                    help='Truncate gradients in step-wise MLP')
parser.add_argument('--ctx_scale', type=float, default=1.0,
                    help='Scalar to multiply context layer (for "lesions")')
# Analyses
parser.add_argument('--dim_red_method', default='pca', 
                    choices=['pca', 'mds', 'tsne'], 
                    help='Method to use for dimensionality reduction')
parser.add_argument('--no_base_analyses', action='store_true',
                    help='Dont run any base analyses (regression, pca, etc.)')
parser.add_argument('--measure_grad_norm', action='store_true',
                    help='Measure the norm of the gradient w.r.t the inputs')
parser.add_argument('--inner_4x4', action='store_true',
                    help='Only analyze inner 4x4 grid')
parser.add_argument('--sbs_analysis', action='store_true',
                    help='Whether analyzing step by step')
parser.add_argument('--sbs_every', type=int, default=1, 
                    help='Number of steps before analyzing step-by-step')

def main(args):
    # CUDA
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        use_cuda = False
        device = "cpu"
    args.device = device
    print("Using CUDA: ", use_cuda)

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Loop over runs
    results = [] # results for each run (n_runs)
    analysis = [] # analysis for each run (n_runs)
    for run_i in range(args.n_runs):
        # Data
        data = get_loaders(args)

        # Model
        model = get_model(args)   
        model.to(device)
        
        # Train (testing and analysis happens throughout training)
        results_i, analysis_i = train(run_i, model, data, args)
        results.append(results_i)
        analysis.append(analysis_i)
    
    # Save results
    data = {'results': results,
            'analysis': analysis}
    with open('../results/'+args.out_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)