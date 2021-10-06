from itertools import combinations
import torch 
import numpy as np
from numpy.core.fromnumeric import reshape
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm

from utils import get_congruency

# TODO: split out into one file per analysis function? Can import that way?
#           -Should do this after fixing everything?
# TODO: can get rid of analyze_test_seq (just do it in notebook with train results?)
# TODO: deal with having different numbers of embeddings, different n_states when inner_4x4

def collect_representations(model, analyze_loader, args):
    """
    Run model through samples in analyze_loader, collecting embeddings and 
    hidden states on each sample. These get saved in a dictionary where keys
    are the name of the representation (e.g., "hidden_f1") and values are a
    [n_states x hidden_dim] matrix containing the average representations for 
    each face. 
    """
    # Data structures for saving representations
    n_states = args.n_states_a
    reps = {n: [[]] * n_states for n in model.rep_names} # all trials
    reps_ctx0 = {n: [[]] * n_states for n in model.rep_names} # context 0 trials
    reps_ctx1 = {n: [[]] * n_states for n in model.rep_names} # context 1 trials

    # Collect representations
    idx2tensor = args.idx2tensor_a
    model.eval()
    with torch.no_grad():
        # Get embeddings from model for each face
        face_embedding = model.face_embedding
        face_embedding.to(args.device)
        embeddings = []
        for idx in range(n_states):
            face_tensor = idx2tensor[idx].unsqueeze(0).to(args.device) 
            embedding = face_embedding(face_tensor) # [1, state_dim]
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
        embeddings = np.concatenate(embeddings, axis=0) # [n_states, state_dim]

        # Get hidden states from model for each sample
        for batch in analyze_loader:
            ctx, f1, f2, y, info = batch
            batch_size = ctx.shape[0]
            ctx = ctx.to(args.device)
            f1 = f1.to(args.device) 
            f2 = f2.to(args.device) 

            # Run model to get hidden representations 
            y_hat, reps_batch = model(ctx, f1, f2)

            # Save hidden representations (unbatched)
            for i in range(batch_size):
                idx1 = info['idx1'][i]
                idx2 = info['idx2'][i]
                for rep_name, rep_batch in reps_batch.items():
                    if model.rep_sort[rep_name][0]: # add rep to face1 list
                        reps[rep_name][idx1].append(rep_batch[i])
                    if model.rep_sort[rep_name][1]: # add rep to face2 list
                        reps[rep_name][idx2].append(rep_batch[i])
                    if ctx[i] == 0:
                        if model.rep_sort[rep_name][0]: # add rep to face1 list
                            reps_ctx0[rep_name][idx1].append(rep_batch[i])
                        if model.rep_sort[rep_name][1]: # add rep to face2 list
                            reps_ctx0[rep_name][idx2].append(rep_batch[i])
                    elif ctx[i] == 1:
                        if model.rep_sort[rep_name][0]: # add rep to face1 list
                            reps_ctx1[rep_name][idx1].append(rep_batch[i])
                        if model.rep_sort[rep_name][1]: # add rep to face2 list
                            reps_ctx1[rep_name][idx2].append(rep_batch[i])
    
    # Function for averaging representations
    def average_reps(reps, args):
        ave_reps = {}
        # Average representations over trials
        for name, rep_list in reps:
            ave_rep = []
            for sample_list in rep_list:
                samples = [s.unsqueeze(0) for s in sample_list]
                samples = torch.cat(samples, dim=0)
                ave = torch.mean(samples, dim=0)
                ave_rep.append(ave)
            ave_rep = [r.unsqueeze(0) for r in ave_rep]
            ave_rep = torch.cat(ave_rep, dim=0) 
            ave_reps[name] = ave_rep.cpu().numpy() # [n_states, hidden_dim]
        # Average averaged representations (if applicable)
        for ave_name, rep_names in model.rep_aves.items():
            ave_rep = [np.expand_dims(ave_reps[n],axis=0) for n in rep_names]
            ave_rep = np.concatenate(ave_rep, axis=0)
            ave_rep = np.mean(ave_rep, axis=0)
            ave_reps[ave_name] = ave_rep
        return ave_reps
    
    # Get average representation for each face
    ave_reps = average_reps(reps) # dict: keys are [n_states, hidden]
    ave_reps_ctx0 = average_reps(reps_ctx0) # dict: keys are [n_states, hidden]
    ave_reps_ctx1 = average_reps(reps_ctx1) # dict: keys are [n_states, hidden]

    # Consolidate into one dictionary
    representations = {'embeddings': embeddings}
    for rep_name in ave_reps.keys():
        representations[rep_name] = ave_reps[rep_name]
        representations['%s_ctx0' % rep_name] = ave_reps_ctx0[rep_name]
        representations['%s_ctx1' % rep_name] = ave_reps_ctx1[rep_name]

        # Concatenate context 0 and context 1 (for dimensionality reduction)
        ctx0 = ave_reps_ctx0[rep_name]
        ctx1 = ave_reps_ctx1[rep_name]
        ctx = np.concatenate([ctx0, ctx1], axis=0) # [2*n_states, hidden_dim]
        representations['%s_ctx' % rep_name] = ctx
    return representations

def compute_distances(reps, args):
    """
    Compute the Euclidean distance between the averaged representations for 
    every pair of faces, for each type of representation. Additional information
    about each pair of faces is also included (e.g., congruency, ditance in
    ground-truth grid, etc.s)
    """
    n_states = args.n_states_a
    loc2idx = args.loc2idx_a
    idx2loc = {idx:loc for loc, idx in loc2idx.items()}
    distance_data = {'rep_dists': {n:[] for n in reps.keys()},
                     'rep_dists_cong': {n:[] for n in reps.keys()},
                     'rep_dists_incong': {n:[] for n in reps.keys()},
                     'rank_diff0': [], # 1D rank differences along axis 0
                     'rank_diff1': [], # 1D rank differences along axis 1
                     'grid_dists': [], # distances in ground truth grid
                     'idx1': [],  # index of face 1
                     'idx2': [],  # index of face 2
                     'loc1': [],  # location of face 1
                     'loc2': [],  # location of face 2
                     'cong': [],  # congruency
                     'theta': [], # angle between loc1 and loc2
                     'phi': []}   # sin(2*theta) (continuous measure of cong)
    idxs = [idx for idx in range(n_states)]
    # Loop through every pair of faces
    for idx1, idx2 in combinations(idxs, 2):
        # Record basic info
        loc1 = idx2loc[idx1]
        loc2 = idx2loc[idx2]
        cong = get_congruency(loc1, loc2)
        distance_data['idx1'].append(idx1)
        distance_data['idx2'].append(idx2)
        distance_data['loc1'].append(loc1)
        distance_data['loc2'].append(loc2)
        distance_data['cong'].append(cong)

        # Compute 1D rank differences
        (x1, y1) = loc1 
        (x2, y2) = loc2
        rank_diff0 = abs(x1 - x2)
        rank_diff1 = abs(y1 - y2)
        distance_data['rank_diff0'].append(rank_diff0)
        distance_data['rank_diff1'].append(rank_diff1)

        # Compute Euclidean distance in ground truth grid
        grid_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        distance_data['grid_dists'].append(grid_dist)

        # Compute the angle between the two locations
        theta = np.arctan2((y2-y1),(x2-x1))
        phi = np.sin(2*theta)
        distance_data['theta'].append(theta) # angle
        distance_data['phi'].append(phi)     # continuous measure of congruency

        # Compute Euclidean distance between representations
        for rep_name, rep in reps.items():
            if len(rep) == n_states: # don't compute distances for _ctx
                rep1 = rep[idx1]
                rep2 = rep[idx2]
                rep_dist = np.linalg.norm(rep1 - rep2)
                distance_data['rep_dists'][rep_name].append(rep_dist)
                if cong == 1:
                    distance_data['rep_dists_cong'][rep_name].append(rep_dist)
                elif cong == -1:
                    distance_data['rep_dists_incong'][rep_name].append(rep_dist)

    return distance_data

def dimensionality_reduction(reps, dists, args):
    """
    Reduce dimensionality using PCA, MDS, or t-SNE for visualization purposes.
    This is done for each type of averaged representation.
    """
    method = args.dim_red_method
    n_components = 2
    if method == 'pca':
        reduction = PCA(n_components=n_components)
    elif method == 'mds':
        reduction = MDS(n_components=n_components)
    elif method == 'tsne':
        reduction = TSNE(n_components=n_components)
    
    results = {'locs': [(l1,l2) for l1,l2 in zip(dists['loc1'], dists['loc2'])]}
    for rep_name, rep in reps.items():
        embedding = reduction.fit_transform(rep) # [n_states, 2]
        results[rep_name] = embedding
    return results

def distance_ratio(reps, dists, args):
    """
    Compute the ratio of average distances between representations of 
    congruent pairs of faces and incongruent pairs of faces (cong / incong)
    """
    results = {}
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    for rep_name in rep_names:
        rep_dists_cong = dists['rep_dists_cong'][rep_name]
        rep_dists_incong = dists['rep_dists_incong'][rep_name]
        ave_cong_dist = np.mean(rep_dists_cong)
        ave_incong_dist = np.mean(rep_dists_incong)
        ratio = ave_cong_dist / ave_incong_dist
        results[rep_name] = ratio 
    return results

def ttest(reps, dists, args):  
    """
    Perform an independent-samples t-test to determine if there is a 
    statistically significant difference between the average distance between
    faces in congruent pairs vs. incongruent pairs
    """
    results = {}
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    for rep_name in rep_names:
        rep_dists_cong = dists['rep_dists_cong'][rep_name]
        rep_dists_incong = dists['rep_dists_incong'][rep_name]
        t, p = ttest_ind(rep_dists_cong, rep_dists_incong)
        results[rep_name] = {'t_statistic': t, 'p_value': p}
    return results

def correlation(reps, dists, args):
    """
    Correlate distances between representations with distances in the 
    underlying ground-truth grid. 
    """
    results = {}
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    grid_dists = dists['grid_dists'] # [n_states choose 2]
    for rep_name in rep_names:
        rep_dists = dists['rep_dists'][rep_name] # [n_states choose 2]
        r, p = pearsonr(grid_dists, rep_dists)
        results[rep_name] = {'r_statistic': r, 'p_value': p}
    return results

def ols(x, y, grid_dist):
    """
    Helper function for running ordinary least squares (OLS) regression
    """
    stats_model = sm.OLS(y,x).fit() 
    y_hat_E = stats_model.params[0] + (stats_model.params[1]*grid_dist)        
    p = stats_model.pvalues
    t = stats_model.tvalues
    beta = stats_model.params
    bse = stats_model.bse
    results = {'y_hat_E': y_hat_E,
               'p_value': p,
               't_statistic': t,
               'betas': beta,
               'bse': bse}
    return results

def regression(reps, dists, args):
    """
    Regress distances in ground-truth grid against distances between 
    representations:
        rep_dists = beta0 + beta1 * grid_dists + beta2 * congruency
    congruency can either be categorical or continuous ('phi')
    """
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    grid_2d = np.expand_dims(dists['grid_dists'], axis=1) # [n_pairs, 1]
    # Categorical measure of congruency
    cong = np.expand_dims(dists['cong'], axis=1) # [n_pairs, 1]
    x_cat = np.concatenate([grid_2d, cong], axis=1]) # categorical measure
    x_cat = sm.add_constant(x_cat) # add intercept
    # Continuous measure of congruency
    phi = np.expand_dims(dists['phi'], axis=1) # [n_pairs, 1]
    x_con = np.concatenate([grid_2d, phi], axis=1)# continuous measure 
    x_con = sm.add_constant(x_con) # add intercept
    # Loop through representations performing regression
    results = {}
    for rep_name in rep_names:
        rep_dists = np.array(dists['rep_dists'][rep_name]) # [n_pairs]
        cat_results = ols(x_cat, rep_dists, grid_2d)
        con_results = ols(x_con, rep_dists, grid_2d)
        results[rep_name] = {'categorical_regression': cat_results,
                             'continuous_regression': con_results}
    return results

def regression_with_1D(reps, dists, args):
    """
    Regress distances in ground-truth grid and 1D rank differences against 
    distances between representations:
        rep_dists = beta0 + beta1 * grid_dists + beta2 * congruency
                        + beta3 * 1D_rank_diffs
    congruency can either be categorical or continuous ('phi')
    """
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    # 1D rank differences along appropriate axis
    grid_1d_0 = np.expand_dims(dists['rank_diff0'], axis=1) # [n_pairs, 1]
    grid_1d_1 = np.expand_dims(dists['rank_diff1'], axis=1) # [n_pairs, 1]
    grid_1d = np.concatenate([grid_1d_0, grid_1d_1], axis=0) # [2*n_pairs, 1]
    # 2D Euclidean distances in ground-truth grid
    grid_2d = np.expand_dims(dists['grid_dists'], axis=1) # [n_pairs, 1]
    grid_2d = np.concatenate([grid_2d, grid_2d], axis=0) # [2*n_pairs, 1]
    # Categorical measure of congruency
    cong = np.expand_dims(dists['cong'], axis=1) # [n_pairs, 1]
    cong = np.concatenate([cong, cong], axis=0) # [2*n_pairs, 1]
    x_cat = np.concatenate([grid_1d, grid_2d, cong], axis=1) # categorical
    x_cat = sm.add_constant(x_cat) # add intercept
    # Continuous measure of congruency
    phi = np.expand_dims(dists['phi'], axis=1) # [n_pairs, 1]
    phi = np.concatenate([phi, phi], axis=0) # [2*n_pairs, 1]
    x_con = np.concatenate([grid_1d, grid_2d, phi], axis=1)# continuous 
    x_con = sm.add_constant(x_con) # add intercept
    # Loop through representations performing regression
    results = {}
    for rep_name in rep_names:
        name0= '%s_ctx0' % rep_name
        name1 = '%s_ctx1' % rep_name
        if name0 in dists['rep_dists'] and name1 in dists['rep_dists']:
            dists0 = np.array(dists['rep_dists'][name0]) # [n_pairs]
            dists1 = np.array(dists['rep_dists'][name1]) # [n_pairs]
            rep_dists = np.concatenate([dists0, dists1], axis=0) # [2*n_pairs]
            cat_results = ols(x_cat, rep_dists, grid_2d)
            con_results = ols(x_con, rep_dists, grid_2d)
            results[rep_name] = {'categorical_regression': cat_results,
                                'continuous_regression': con_results}
    return results

def regression_exclusion(reps, dists, args):
    """
    Regress distances in ground-truth grid against distances between
    representations, excluding all pairs including one face at a time. 
    The point of this is to see if the measured warping is due to some outlier
    faces, or is consistent across all faces. 
    """
    n_states = args.n_states_a 
    rep_names = [rep_name for rep_name in dists['rep_dists_cong'].keys()]
    grid_2d = np.expand_dims(dists['grid_dists'], axis=1) # [n_pairs, 1]
    # Categorical measure of congruency
    cong = np.expand_dims(dists['cong'], axis=1) # [n_pairs, 1]
    idxs1 = dists['idx1']
    idxs2 = dists['idx2']

    # Loop through face indexes to exclude one at a time
    results_each = {rep_name:[] for rep_name in rep_names}
    for idx in range(n_states):
        # Get indices for pairs that do not include idx
        s_idxs = []
        for i, (idx1, idx2) in enumerate(zip(idxs1, idxs2)):
            if idx not in [idx1, idx2]:
                s_idxs.append(i)
        # Perform regression, excluding idx
        grid_2d_ex = grid_2d[s_idxs]
        cong_ex = cong[s_idxs]
        x_cat = np.concatenate([grid_2d_ex, cong_ex], axis=1)
        x_cat = sm.add_constant(x_cat)
        for rep_name in rep_names:
            rep_dists = np.array(dists['rep_dists'][rep_name][s_idxs])
            cat_results = ols(x_cat, rep_dists, grid_2d)
            results_each[rep_name][idx].append(cat_results)
        
    # Regression after removing both "major" corners - (0,0) and (3,3)
    # TODO
    
    s_idxs = []
    for i, (idx1, idx2) in enumerate(zip(idxs1, idxs2)):
        if idx not in [idx1, idx2]:
            s_idxs.append(i)
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model == 'stepwisemlp':
        for h in range(2):
            y = hidd_dists[s_idxs,h]
            _, p_val, t_val, param, bse = run_regression(x_cat,y,grid_dists)
            p_vals[h].append(p_val)
            t_vals[h].append(t_val)
            params[h].append(param)
            bses[h].append(bse)
    else:
        y = hidd_dists[s_idxs]
        _, p_val, t_val, param, bse = run_regression(x_cat,y,grid_dists)
        p_vals.append(p_val)
        t_vals.append(t_val)
        params.append(param)
        bses.append(bse)
    states.append(16)
    
    # regression analysis - after removing (0,0) and (3,3), (3,0) and (0.3)
    s_idxs = [i for i, sample in enumerate(samples) if ((0 not in sample) & (15 not in sample) &
                                                        (3 not in sample) & (12 not in sample))] #[66]
    x_cat = np.concatenate((grid_dists[s_idxs].reshape((-1,1)), binary_phi[s_idxs].reshape((-1,1))),axis=1)
    x_cat = sm.add_constant(x_cat)
    if args.cortical_model == 'stepwisemlp':
        for h in range(2):
            y = hidd_dists[s_idxs,h]  
            _, p_val, t_val, param, bse = run_regression(x_cat,y,grid_dists)
            p_vals[h].append(p_val)
            t_vals[h].append(t_val)
            params[h].append(param)
            bses[h].append(bse)
    else:
        y = hidd_dists[s_idxs]
        _, p_val, t_val, param, bse = run_regression(x_cat,y,grid_dists)
        p_vals.append(p_val)
        t_vals.append(t_val)
        params.append(param)
        bses.append(bse)
    states.append(17)

    states = np.array(states)
    p_vals = np.array(p_vals)
    t_vals = np.array(t_vals)
    params = np.array(params)
    bses = np.array(bses)
    
    exc_reg_results = {'excluded_states': states,
                       'p_vals': p_vals,
                       't_vals': t_vals,
                       'params': params,
                       'bses': bses}                   

    return exc_reg_results

def analyze_test_seq(args, test_data, cortical_result, dist_results):
    import sys
    sys.path.append("..")
    data = get_loaders(batch_size=32, meta=False,
                      use_images=True, image_dir='./images/',
                      n_episodes=None,
                      N_responses=args.N_responses, N_contexts=args.N_contexts,
                      cortical_task = args.cortical_task, #ToDo:check why it was set to cortical_task='face_task',
                      balanced = args.balanced)
    train_data, train_loader, test_data, test_loader, analyze_data, analyze_loader = data

    idx2loc = {idx:loc for loc, idx in test_data.loc2idx.items()}

    # ctx_order = 'first'
    # ctx_order_str = 'ctxF'
    
    analyze_correct = cortical_result['analyze_correct'] # [n_trials, time_steps]: [384, 3]
    analyze_correct = np.asarray(analyze_correct).squeeze()

    hidd_t_idx = 1 # at what time step, t = 1 means at the time of face1 
                                # and t = 2 means at the time of face2
                                # in axis First (axis is at t=0), it should be t = 1
    # create groups based on the row or columns
    # e.g, for context0 (xaxis), first column is group 1, sec col is group 2, and so on.
    # 4 groups for each axis/context; total 8 groups

    # ToDo: why it is always loc1???

    ctx0_g0=[]
    ctx0_g1=[]
    ctx0_g2=[]
    ctx0_g3=[]

    ctx1_g0=[]
    ctx1_g1=[]
    ctx1_g2=[]
    ctx1_g3=[]

    for i, batch in enumerate(analyze_loader):
        if args.cortical_task == 'face_task':
            f1, f2, ctx, y, idx1, idx2 = batch # face1, face2, context, y, index1, index2
        elif args.cortical_task == 'wine_task':
            f1, f2, ctx, y1, y2, idx1, idx2 = batch # face1, face2, context, y1, y2, index1, index2        
            msg = 'analyze_test_seq is only implemented for one response, two contexts'
            assert args.N_responses == 'one' and args.N_contexts == 2, msg

            if args.N_responses == 'one':
                y = y1
        # f1, f2, ax, y, idx1, idx2 = batch
        acc = analyze_correct[i][hidd_t_idx]
        ctx = ctx.cpu().numpy().squeeze()
        idx1 = idx1[0]
        idx2 = idx2[0]
        loc1 = idx2loc[idx1]
        loc2 = idx2loc[idx2]
        if ctx==0:
            if loc1[ctx]==0: ctx0_g0.append(acc) # (len(all_perms)/2) / 4 = [48]
            elif loc1[ctx]==1: ctx0_g1.append(acc)
            elif loc1[ctx]==2: ctx0_g2.append(acc)
            elif loc1[ctx]==3: ctx0_g3.append(acc)
        elif ctx==1:
            if loc1[ctx]==0: ctx1_g0.append(acc)
            elif loc1[ctx]==1: ctx1_g1.append(acc)
            elif loc1[ctx]==2: ctx1_g2.append(acc)
            elif loc1[ctx]==3: ctx1_g3.append(acc)
    ctx0_accs = [np.mean(ctx0_g0), np.mean(ctx0_g1), 
                np.mean(ctx0_g2), np.mean(ctx0_g3) ]
    ctx1_accs = [np.mean(ctx1_g0), np.mean(ctx1_g1), 
                np.mean(ctx1_g2), np.mean(ctx1_g3) ]
         
    # print('Accuracy at t=%s (face%s) contex 0:' %(hidd_t_idx,hidd_t_idx), ctx0_accs)
    # print('Accuracy at t=%s (face%s) contex 1:' %(hidd_t_idx,hidd_t_idx), ctx1_accs)
    return ctx0_accs, ctx1_accs

def get_analyses(args, final_step):
    if not final_step: # analyses to conduct at every checkpoint
        if args.no_base_analyses:
            analysis_dict = {}
        else:
            analysis_dict = {'distance_ratio': distance_ratio,
                             'ttest': ttest,
                             'correlation': correlation,
                             'regression': regression,
                             'regression_with_1D': regression_with_1D,
                             'regression_exclusion': regression_exclusion,
                             'analyze_test_seq': analyze_test_seq}
    else: # analyses to conduct only at the final step
        analysis_dict = {'dimensionality_reduction': dimensionality_reduction}

    return analysis_dict

def analyze(model, analyze_loader, args, final_step):
    # Gather representations
    reps = collect_representations(model, analyze_loader, args)

    # Compute distances
    dists = compute_distances(reps)

    # Get dictionary with analysis functions
    analysis_dict = get_analyses(args, final_step)

    # Analyze representations
    analysis_results = {}
    for analysis_name, analysis_func in analysis_dict.items():
        if args.verbose:
            print("Performing analysis: ", analysis_name)
        analysis_results[analysis_name] = analysis_func(reps, dists, args)
    
    return analysis_results