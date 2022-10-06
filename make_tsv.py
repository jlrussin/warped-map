import argparse
import pickle
import os
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='../results/',
                    help='Path to results directory with pickle files')
parser.add_argument('--out_dir', default='../results/tsv/',
                    help='Path to directory to output tsv files')
parser.add_argument('--model_name', default='rnn',
                    choices=['mlp','rnn'], 
                    help='Type of model to use')

def make_tsvs(args):
    # Make output directory
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Load basic results
    fn = os.path.join(args.in_dir, f"{args.model_name}.P")
    with open(fn, 'rb') as f:
        results = pickle.load(f)

    # Representation
    if args.model_name == 'mlp':
        rep_name = 'hidden'
    elif args.model_name == 'rnn':
        rep_name = 'average'

    # Accuracy results
    header = []
    data = []
    acc_results = results['results']
    for run_i, run in enumerate(acc_results):
        for acc_type in ['acc', 'cong_acc', 'incong_acc']:
            # Get column of data
            col = np.array([s[acc_type] for s in run['analyze_accs']])
            col = np.expand_dims(col, axis=1)
            data.append(col)

            # Header
            run_name = f"run{run_i}"
            acc_name = acc_type.replace('_', '-')
            head = f"{run_name}_{acc_name}"
            header.append(head)
    data = np.concatenate(data, axis=1)
    header = "\t".join(header)
    out_file = os.path.join(args.out_dir, f'acc_{args.model_name}.tsv')
    print(f"Writing accuracy results to {out_file}")
    header_len = len(header.split('\t'))
    print(f"Shape = {data.shape}, header = {header_len}")
    np.savetxt(out_file, data, delimiter='\t', header=header)
    
    
    # RSA results (2D Euclidean, 1D Compression)
    header = []
    data = []
    analysis = results['analysis']
    for run_i, run in enumerate(analysis):
        for stat in ['betas', 't_statistic', 'p_value']:
            for var_i, var in enumerate(['int', '2D', '1D', 'ctx']):
                # Get column of data
                col = np.array([s['rsa'][rep_name][stat][var_i] for s in run])
                col = np.expand_dims(col, axis=1)
                data.append(col)

                # Header
                run_name = f"run{run_i}"
                stat_name = stat.replace('_', '-')
                head = f"{run_name}_{stat_name}_{var}"
                header.append(head)
    data = np.concatenate(data, axis=1) # [n_steps, n_runs * n_stats * n_vars]
    header = "\t".join(header) # n_runs * n_stats * n_vars
    out_file = os.path.join(args.out_dir, f'rsa_{args.model_name}.tsv')
    print(f"Writing RSA results to {out_file}")
    header_len = len(header.split('\t'))
    print(f"Shape = {data.shape}, header = {header_len}")
    np.savetxt(out_file, data, delimiter='\t', header=header)

    # Warping results (t-test and ratio)
    header = []
    data = []
    for run_i, run in enumerate(analysis):
        # T-test results
        for stat in ['t_statistic', 'p_value']:
            # Get column of data
            col = np.array([s['ttest'][rep_name][stat] for s in run])
            col = np.expand_dims(col, axis=1)
            data.append(col)

            # Header
            run_name = f"run{run_i}"
            stat_name = stat.replace('_', '-')
            head = f"{run_name}_{stat_name}"
            header.append(head)

        # Ratio results
        for stat in ['ratio', 'ave_cong_dist', 'ave_incong_dist']:
            # Get column 
            col = np.array([s['distance_ratio'][rep_name][stat] for s in run])
            col = np.expand_dims(col, axis=1)
            data.append(col)

            # Header
            run_name = f"run{run_i}"
            stat_name = stat.replace('_', '-')
            head = f"{run_name}_{stat_name}"
            header.append(head)
    data = np.concatenate(data, axis=1) # [n_steps, n_runs * n_stats * n_vars]
    header = "\t".join(header) # n_runs * n_stats * n_vars
    out_file = os.path.join(args.out_dir, f'warping_{args.model_name}.tsv')
    print(f"Writing warping results to {out_file}")
    header_len = len(header.split('\t'))
    print(f"Shape = {data.shape}, header = {header_len}")
    np.savetxt(out_file, data, delimiter='\t', header=header)

    # Ablation results (warping: t-test and ratio)
    header = []
    data = []
    scales = [f'0p{s}' for s in range(1,10)]
    scales.append('1p0')
    for scale in scales:
        fn = f"{args.model_name}_ctx_scale{scale}.P"
        path = os.path.join(args.in_dir, fn)
        with open(path, 'rb') as f:
            results = pickle.load(f)
        analysis = results['analysis']
        for run_i, run in enumerate(analysis):
            # T-test results
            for stat in ['t_statistic', 'p_value']:
                # Get column of data
                col = np.array([s['ttest'][rep_name][stat] for s in run])
                col = np.expand_dims(col, axis=1)
                data.append(col)

                # Header
                scale_name = f"w{scale}"
                run_name = f"run{run_i}"
                stat_name = stat.replace('_', '-')
                head = f"{scale_name}_{run_name}_{stat_name}"
                header.append(head)

            # Ratio results
            for stat in ['ratio', 'ave_cong_dist', 'ave_incong_dist']:
                # Get column 
                col = np.array([s['distance_ratio'][rep_name][stat] for s in run])
                col = np.expand_dims(col, axis=1)
                data.append(col)

                # Header
                scale_name = f"w{scale}"
                run_name = f"run{run_i}"
                stat_name = stat.replace('_', '-')
                head = f"{scale_name}_{run_name}_{stat_name}"
                header.append(head)
    data = np.concatenate(data, axis=1) # [n_steps, n_runs * n_stats * n_vars]
    header = "\t".join(header) # n_runs * n_stats * n_vars
    out_file = os.path.join(args.out_dir, f'ablation_{args.model_name}.tsv')
    print(f"Writing ablation results to {out_file}")
    header_len = len(header.split('\t'))
    print(f"Shape = {data.shape}, header = {header_len}")
    np.savetxt(out_file, data, delimiter='\t', header=header)

    # TODO: regression exclusion results

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    make_tsvs(args)