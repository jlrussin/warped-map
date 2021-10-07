from itertools import combinations
import torch 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm

from utils import get_congruency

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
    reps = {}
    reps_ctx0 = {}
    reps_ctx1 = {}
    for rep_name in model.rep_names:
        reps[rep_name] = {idx:[] for idx in range(n_states)}
        reps_ctx0[rep_name] = {idx:[] for idx in range(n_states)}
        reps_ctx1[rep_name] = {idx:[] for idx in range(n_states)}

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
    def average_reps(reps):
        ave_reps = {}
        # Average representations over trials
        for name, rep_dict in reps.items():
            ave_rep = {}
            for idx, sample_list in rep_dict.items():
                samples = [s.unsqueeze(0) for s in sample_list]
                samples = torch.cat(samples, dim=0)
                ave = torch.mean(samples, dim=0)
                ave_rep[idx] = ave
            ave_rep = [ave_rep[idx].unsqueeze(0) for idx in range(n_states)]
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
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    distance_data = {'rep_dists': {n:[] for n in rep_names},
                     'rep_dists_cong': {n:[] for n in rep_names},
                     'rep_dists_incong': {n:[] for n in rep_names},
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
        for rep_name in rep_names:
            rep1 = reps[rep_name][idx1]
            rep2 = reps[rep_name][idx2]
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
    n_states = args.n_states_a
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    results = {}
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
    n_states = args.n_states_a
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    results = {}
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
    n_states = args.n_states_a
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    results = {}
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
    n_states = args.n_states_a
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    grid_2d = np.expand_dims(dists['grid_dists'], axis=1) # [n_pairs, 1]
    # Categorical measure of congruency
    cong = np.expand_dims(dists['cong'], axis=1) # [n_pairs, 1]
    x_cat = np.concatenate([grid_2d, cong], axis=1) # categorical measure
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
    n_states = args.n_states_a
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
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
    rep_names = [rep_name for rep_name, r in reps.items() if len(r) == n_states]
    grid_2d = np.expand_dims(dists['grid_dists'], axis=1) # [n_pairs, 1]
    # Categorical measure of congruency
    cong = np.expand_dims(dists['cong'], axis=1) # [n_pairs, 1]
    idxs1 = dists['idx1']
    idxs2 = dists['idx2']
    locs1 = dists['loc1']
    locs2 = dists['loc2']

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
            rep_dists = np.array(dists['rep_dists'][rep_name])[s_idxs]
            cat_results = ols(x_cat, rep_dists, grid_2d)
            results_each[rep_name].append(cat_results)
        
    # Regression after removing both "major" corners - (0,0) and (3,3)
    bl, tl, br, tr = args.corners # bot-left, top-left, bot-right, top-right
    major = [bl, tr]   # locations of "major" corners
    results_ex_bl_tr = {} # excluding bottom-left and top-right
    # Get indices for pairs that do not include either of the major corners
    s_idxs = []
    for i, (loc1, loc2) in enumerate(zip(locs1, locs2)):
        if not any(c in [loc1, loc2] for c in major):
            s_idxs.append(i)
    # Perform regression, excluding pairs with either major corner
    grid_2d_ex = grid_2d[s_idxs]
    cong_ex = cong[s_idxs]
    x_cat = np.concatenate([grid_2d_ex, cong_ex], axis=1)
    x_cat = sm.add_constant(x_cat)
    for rep_name in rep_names:
        rep_dists = np.array(dists['rep_dists'][rep_name])[s_idxs]
        cat_results = ols(x_cat, rep_dists, grid_2d)
        results_ex_bl_tr[rep_name] = cat_results
    
    # Regression after removing all corners - (0,0), (0,3), (3,0), (3,3)
    results_ex_corners = {}
    # Get indices for pairs that do not include either of the major corners
    s_idxs = []
    for i, (loc1, loc2) in enumerate(zip(locs1, locs2)):
        if not any(c in [loc1, loc2] for c in args.corners):
            s_idxs.append(i)
    # Perform regression, excluding pairs with either major corner
    grid_2d_ex = grid_2d[s_idxs]
    cong_ex = cong[s_idxs]
    x_cat = np.concatenate([grid_2d_ex, cong_ex], axis=1)
    x_cat = sm.add_constant(x_cat)
    for rep_name in rep_names:
        rep_dists = np.array(dists['rep_dists'][rep_name])[s_idxs]
        cat_results = ols(x_cat, rep_dists, grid_2d)
        results_ex_corners[rep_name] = cat_results
    
    results = {'excluding_each': results_each,
               'excluding_bl_tr': results_ex_bl_tr,
               'excluding_corners': results_ex_corners}                   
    return results

def measure_grad_norms(model):
    """
    Measure the L2 norm of the gradient of the loss w.r.t. each embedding
    This takes place after loss.backward() has been called but before 
    optimizer.step() has been taken.
    This needs to be called in train, so does not appear in analysis_dict
    """
    grd_ctx = torch.linalg.norm(model.ctx_embed.grad, dim=1)
    grd_f1 = torch.linalg.norm(model.f1_embed.grad, dim=1)
    grd_f2 = torch.linalg.norm(model.f2_embed.grad, dim=1)

    results = {'grd_ctx': grd_ctx.cpu().numpy(),
               'grd_f1': grd_f1.cpu().numpy(),
               'grd_f2': grd_f2.cpu().numpy()}
    return results

def estimate_vis_params(reps, dists, args):
    """
    Estimate parameters for visualizing representations over time. 
    The parameterization works as follows:
        1. Two groups ("G" and "H") are defined based on locations with the same
           rank along the "congruent" (bottom-left to top-right) and 
           "incongruent" (bottom-right to top-left) diagonals
        2. Distances between adjacent groups are estimated by measuring the 
           Euclidean distance between the average of all vectors in each group.
             -alpha: distances between adjacent G ("congruent") groups
             -beta: distances between adjacent H ("incongruent") groups
        3. These distances are used to reconstruct the grid in 2 dimensions 
           (this is done in a jupyter notebook, not in this function)
    """
    # Helpful variables
    n_states = args.n_states_a # total number of faces in grid
    grid_size = np.sqrt(n_states) # length of one side of grid
    max_r = grid_size - 1 # maximum rank (starts from 0)
    loc2idx = args.loc2idx_a # dict mapping (x,y) tuples to indices
    idx2loc = {idx:loc for loc, idx in loc2idx.items()} # reverse mapping
    locs = [idx2loc[idx] for idx in range(n_states)] # (x,y) tuples in idx order

    # Construct same-rank groups for "congruent" and "incongruent" diagonals
    c_rank = np.array([l[0]+l[1] for l in locs]) # ranks ("congruent")
    i_rank = np.array([max_r+l[0]-l[1] for l in locs]) # ranks ("incongruent")
    n_ranks = len(set(c_rank)) # number of same-rank groups
    msg = "G and H groups have different sizes"
    assert len(set(c_rank)) == len(set(i_rank)), msg
    G_idxs = [] # same-rank groups for "congruent" diagonal
    H_idxs = [] # same-rank groups for "incongruent" diagonal
    for i in range(n_ranks):
        G_set = [j for j in range(n_states) if c_rank[j] == i] # indices in G[i]
        H_set = [j for j in range(n_states) if i_rank[j] == i] # indices in H[i]
        G_idxs.append(G_set)
        H_idxs.append(H_set)
    
    # Estimate alpha and beta parameters from averaged hidden vectors
    results = {}
    for rep_name, rep in reps.items():
        if len(rep) != n_states:
            continue # don't compute distances for _ctx
        M = rep # [n_states, hidden_dim]
        alpha = []
        beta = []
        n_params = n_ranks - 1 # 1 parameter for each adjacent pair of groups 
        for i in range(n_params):
            # Estimate alpha_{i, i+1}
            x_bar_i = np.mean(M[G_idxs[i],:], axis=0) # ave vec in G_i
            x_bar_ip1 = np.mean(M[G_idxs[i+1],:], axis=0) # ave vec in G_{i+1}
            x_dist = np.linalg.norm(x_bar_i - x_bar_ip1) # distance between aves
            alpha.append(x_dist)
            
            # Estimate beta_{i, i+1}
            y_bar_i = np.mean(M[H_idxs[i],:], axis=0) # ave vec in H_i
            y_bar_ip1 = np.mean(M[H_idxs[i+1],:], axis=0) # ave vec in H_{i+1}
            y_dist = np.linalg.norm(y_bar_i - y_bar_ip1) # distance between aves
            beta.append(y_dist)

            # Save results
            results[rep_name] = {'alpha': alpha, 
                                 'beta': beta,
                                 'n_states': n_states,
                                 'locs': locs,
                                 'idx2loc': idx2loc,
                                 'G_idxs': G_idxs,
                                 'H_idxs': H_idxs}
    return results

def get_analyses(args, final_step):
    if not final_step: # analyses to conduct at every checkpoint
        analysis_dict = {'distance_ratio': distance_ratio,
                         'ttest': ttest,
                         'correlation': correlation,
                         'regression': regression,
                         'regression_with_1D': regression_with_1D,
                         'regression_exclusion': regression_exclusion,
                         'estimate_vis_params': estimate_vis_params}
    else: # analyses to conduct only at the final step
        analysis_dict = {'dimensionality_reduction': dimensionality_reduction}

    return analysis_dict

def analyze(model, analyze_loader, args, final_step):
    # Gather representations
    reps = collect_representations(model, analyze_loader, args)

    # Compute distances
    dists = compute_distances(reps, args)

    # Get dictionary with analysis functions
    analysis_dict = get_analyses(args, final_step)

    # Analyze representations
    analysis_results = {}
    for analysis_name, analysis_func in analysis_dict.items():
        if args.verbose:
            print("Performing analysis: ", analysis_name)
        analysis_results[analysis_name] = analysis_func(reps, dists, args)
    
    # Include distance results for final step
    if final_step:
        analysis_results['dists'] = dists
    
    return analysis_results