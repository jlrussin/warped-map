import torch
import random
from itertools import permutations 
from torch.utils.data import Dataset, DataLoader 
from torchvision.datasets import ImageFolder 
from torchvision.transforms import Compose, Grayscale, ToTensor

from utils import get_congruency

class GridDataGenerator:
    def __init__(self, training_regime='grouped', size=4,  
                 use_images=True, image_dir=None, inner_4x4=False):
        self.training_regime = training_regime # structure of training set
        self.size = size             # length of one side of grid
        self.use_images = use_images # use images rather than one-hot vectors
        self.image_dir = image_dir   # directory with images
        self.inner_4x4 = inner_4x4   # only analyze inner 4x4 grid

        # Generate locations and indices for each face
        self.locs = [(i,j) for i in range(self.size) for j in range(self.size)]
        self.idxs = [idx for idx in range(len(self.locs))]
        self.loc2idx = {loc:idx for loc, idx in zip(self.locs, self.idxs)}
        self.idx2loc = {idx:loc for idx, loc in zip(self.idxs, self.locs)}
        self.n_states = len(self.idxs)

        # Prepare tensors for each idx
        idx2tensor = {}
        if self.use_images:
            # Tensors are images
            transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
            face_images = ImageFolder(self.image_dir, transform)
            for idx in self.idxs:
                idx2tensor[idx] = face_images[idx][0] # [1, 64, 64]
        else:
            # Tensors are integers (indexes of nn.Embedding)
            for idx in self.idxs:
                idx2tensor[idx] = torch.tensor(idx).type(torch.long) # [16]
        self.idx2tensor = idx2tensor

        # Useful variables for doing analyses on inner 4x4 grid
        if self.inner_4x4:
            l = (self.size - 4) // 2 # lower bound of inner 4x4 grid
            u = l + 3 # upper bound of inner 4x4 grid
            self.locs_map = {} # map old locs to new locs
            for (x, y) in self.locs:
                # Subtract lower bound to get new locations
                new_x = x - l 
                new_y = y - l
                if (l <= x <= u) and (l <= y <= u):
                    self.locs_map[(x,y)] = (new_x, new_y)
                else:
                    self.locs_map[(x,y)] = 'outer'
            locs_4x4 = [(x1,x2) for x1 in range(4) for x2 in range(4)]
            self.loc2idx_4x4 = {loc:idx for loc,idx in zip(locs_4x4, range(16))}
            self.idxs_map = {} # map old idxs to new idxs
            for idx in self.idxs:
                loc = self.idx2loc[idx]
                new_loc = self.locs_map[loc]
                if new_loc == 'outer':
                    new_idx = 'outer'
                else:
                    new_idx = self.loc2idx_4x4[new_loc]
                self.idxs_map[idx] = new_idx
            idxs_unmap = {v:k for k,v in self.idxs_map.items()}
            self.idx2tensor_4x4 = {}
            for idx in range(16):
                old_idx = idxs_unmap[idx]
                tensor = self.idx2tensor[old_idx]
                self.idx2tensor_4x4[idx] = tensor
        else:
            l = 0
            u = self.size - 1
        self.corners = ((l,l), (l,u), (u,l), (u,u)) # bl, tl, br, tr
        self.lower = l
        self.upper = u

        # Generate samples according to arguments
        if self.training_regime == 'grouped':
            assert self.size == 4, "size must be 4 if faces are grouped"
            train, test = self.generate_grouped_samples()
            _, _, analyze = self.generate_ungrouped_samples()
        elif self.training_regime in ['ungrouped', 'balanced', 'train_all']:
            train, test, analyze = self.generate_ungrouped_samples()

        # Add info to each sample (dict: loc1, loc2, idx1, idx2, info)
        train_info = self.append_info(train)
        test_info = self.append_info(test)
        analyze_info = self.append_info(analyze, inner_4x4=self.inner_4x4)

        # Save samples
        self.train = train_info
        self.test = test_info
        self.analyze = analyze_info

    def generate_grouped_samples(self):
        # Define groups
        group1 = [(1,0),(2,0),(0,1),(3,1),(1,2),(2,2),(0,3),(3,3)]
        group2 = [loc for loc in self.locs if loc not in group1]

        # Generate all within-group pairs
        all_within1 = [pair for pair in permutations(group1, 2)]
        all_within2 = [pair for pair in permutations(group2, 2)]

        # Get rank distances for both contexts in both groups
        d_within1_ctx1 = [pair[0][0] - pair[1][0] for pair in all_within1]
        d_within1_ctx2 = [pair[0][1] - pair[1][1] for pair in all_within1]
        d_within2_ctx1 = [pair[0][0] - pair[1][0] for pair in all_within2]
        d_within2_ctx2 = [pair[0][1] - pair[1][1] for pair in all_within2]
        
        # Create within-group samples, excluding pairs with distance > 1
        within = []
        # group 1
        for pair, d1, d2 in zip(all_within1, d_within1_ctx1, d_within1_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((0, f1, f2, y)) # (ctx, F1, F2, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((1, f1, f2, y)) # (ctx, F1, F2, y)
        # group 2
        for pair, d1, d2 in zip(all_within2, d_within2_ctx1, d_within2_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((0, f1, f2, y)) # (ctx, F1, F2, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((1, f1, f2, y)) # (ctx, F1, F2, y)

        # Between-group "hub" pairs
        hubs_ctx1 = [(1,0), (2,2), (1,1), (2,3)]
        hubs_ctx2 = [(1,2), (3,1), (0,2), (2,1)]
        hub_pairs_ctx1 = [((2,2),(1,1)),((2,2),(1,3)),
                         ((2,2),(3,0)),((2,2),(3,2)),
                         ((1,0),(0,0)),((1,0),(0,2)),
                         ((1,0),(2,1)),((1,0),(2,3)),
                         ((2,3),(1,0)),((2,3),(1,2)),
                         ((2,3),(3,1)),((2,3),(3,3)),
                         ((1,1),(0,1)),((1,1),(0,3)),
                         ((1,1),(2,0)),((1,1),(2,2))]
        hub_pairs_ctx2 = [((1,2),(1,1)),((1,2),(1,3)),
                         ((1,2),(2,1)),((1,2),(2,3)),
                         ((3,1),(0,0)),((3,1),(0,2)),
                         ((3,1),(3,0)),((3,1),(3,2)),
                         ((0,2),(0,1)),((0,2),(0,3)),
                         ((0,2),(3,1)),((0,2),(3,3)),
                         ((2,1),(1,0)),((2,1),(1,2)),
                         ((2,1),(2,0)),((2,1),(2,2))]
        
        # Add in reversals of each pair
        between_ctx1 = []
        for pair in hub_pairs_ctx1:
            between_ctx1.append((pair[0],pair[1]))
            between_ctx1.append((pair[1],pair[0]))
        between_ctx2 = []
        for pair in hub_pairs_ctx2:
            between_ctx2.append((pair[0],pair[1]))
            between_ctx2.append((pair[1],pair[0]))

        # Get rank distances for both contexts in both groups
        d_between_ctx1 = [pair[0][0] - pair[1][0] for pair in between_ctx1]
        d_between_ctx2 = [pair[0][1] - pair[1][1] for pair in between_ctx2]
        not_one_over_ctx1 = [d for d in d_between_ctx1 if not abs(d) == 1]
        not_one_over_ctx2 = [d for d in d_between_ctx2 if not abs(d) == 1]
        msg1 = "{} are not one-over!".format(not_one_over_ctx1)
        assert len(not_one_over_ctx1) == 0, msg1
        msg2 = "{} are not one-over!".format(not_one_over_ctx2)
        assert len(not_one_over_ctx2) == 0, msg2
        
        # Create samples from between-group "hub" pairs
        between = []
        for pair, d1 in zip(between_ctx1, d_between_ctx1):
            f1 = pair[0]
            f2 = pair[1]
            y = int(d1 > 0)
            between.append((0, f1, f2, y)) # (ctx, F1, F2, y)
        for pair, d2 in zip(between_ctx2, d_between_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            y = int(d2 > 0)
            between.append((1, f1, f2, y)) # (ctx, F1, F2, y)

        # Compile training set
        train = within + between
        random.shuffle(train)
        
        # Get all test pairs separated by a hub
        test_pairs_ctx1 = []
        for pair in between_ctx1:
            locb1 = pair[0]
            locb2 = pair[1]
            for sample in within:
                ctx, locw1, locw2, y = sample
                if ctx != 0:
                    continue
                if locb1 == locw1:
                    test_pairs_ctx1.append((locb2, locw2))
                    test_pairs_ctx1.append((locw2, locb2))
                if locb1 == locw2:
                    test_pairs_ctx1.append((locb2, locw1))
                    test_pairs_ctx1.append((locw1, locb2))
                if locb2 == locw1:
                    test_pairs_ctx1.append((locb1, locw2))
                    test_pairs_ctx1.append((locw2, locb1))
                if locb2 == locw2:
                    test_pairs_ctx1.append((locb1, locw1))
                    test_pairs_ctx1.append((locw1, locb1))            
        test_pairs_ctx1 = list(set(test_pairs_ctx1))
        
        test_pairs_ctx2 = []
        for pair in between_ctx2:
            locb1 = pair[0]
            locb2 = pair[1]
            for sample in within:
                ctx, locw1, locw2, y = sample
                if ctx != 1:
                    continue
                if locb1 == locw1:
                    test_pairs_ctx2.append((locb2, locw2))
                    test_pairs_ctx2.append((locw2, locb2))
                if locb1 == locw2:
                    test_pairs_ctx2.append((locb2, locw1))
                    test_pairs_ctx2.append((locw1, locb2))
                if locb2 == locw1:
                    test_pairs_ctx2.append((locb1, locw2))
                    test_pairs_ctx2.append((locw2, locb1))
                if locb2 == locw2:
                    test_pairs_ctx2.append((locb1, locw1))
                    test_pairs_ctx2.append((locw1, locb1))            
        test_pairs_ctx2 = list(set(test_pairs_ctx2))
        
        # Remove pairs that include a context 1 hub
        test_pairs_nohub_ctx1 = []
        for loc1, loc2 in test_pairs_ctx1:
            if loc1 not in hubs_ctx1 and loc2 not in hubs_ctx1:
                test_pairs_nohub_ctx1.append((loc1, loc2))

        # Remove pairs that include a context 2 hub
        test_pairs_nohub_ctx2 = []
        for loc1, loc2 in test_pairs_ctx2:
            if loc1 not in hubs_ctx2 and loc2 not in hubs_ctx2:
                test_pairs_nohub_ctx2.append((loc1, loc2))
                
        # Create test set, removing 0s
        test = []
        for pair in test_pairs_nohub_ctx1:
            loc1 = pair[0]
            loc2 = pair[1]
            d = loc1[0] - loc2[0]
            if d != 0:
                f1 = loc1
                f2 = loc2
                y = int(d > 0)
                test.append((0, f1, f2, y)) # (ctx, F1, F2, y)
        for pair in test_pairs_nohub_ctx2:
            loc1 = pair[0]
            loc2 = pair[1]
            d = loc1[1] - loc2[1]
            if d != 0:
                f1 = loc1
                f2 = loc2
                y = int(d > 0)
                test.append((1, f1, f2, y)) # (ctx, F1, F2, y)

        return train, test
    
    def generate_ungrouped_samples(self):
        train, test, analyze = [], [], []
        for idx1, idx2 in permutations(self.idxs, 2):
            f1, f2 = self.idx2loc[idx1], self.idx2loc[idx2]
            # Context 0
            for ctx in range(2):
                r1, r2 = f1[ctx], f2[ctx] # ranks in appropriate context
                d = r1 - r2 # rank difference
                if d!= 0:
                    y = int(d > 0)
                    sample = (ctx, f1, f2, y) # (ctx, F1, F2, y)
                    if not self.inner_4x4:
                        analyze.append(sample)
                    else:
                        if self.is_inner_4x4(f1, f2):
                            analyze.append(sample)
                    if self.training_regime == 'train_all':
                        train.append(sample)
                        test.append(sample)
                    else:
                        if abs(d) > 1: # test samples have rank diff > 1
                            test.append(sample)
                        elif abs(d) == 1: # train samples have rank diff of 1
                            train.append(sample)

        # Balance wins/losses of each face by adding pairings with corners
        if self.training_regime == 'balanced':
            assert self.size == 4, "balancing implemented for size of 4"
            # Extra wins
            wins1 = [(1,(0,0),(0,1),0)] * 4 + [(1,(0,1),(0,0),1)] * 4
            wins2 = [(1,(0,0),(0,2),0)] * 4 + [(1,(0,2),(0,0),1)] * 4
            wins3 = [(0,(0,0),(1,0),0)] * 4 + [(0,(1,0),(0,0),1)] * 4
            wins4 = [(0,(0,0),(2,0),0)] * 4 + [(0,(2,0),(0,0),1)] * 4
            wins = wins1 + wins2 + wins3 + wins4

            # Extra losses
            losses1 = [(1,(3,3),(3,2),1)] * 4 + [(1,(3,2),(3,3),0)] * 4
            losses2 = [(1,(3,3),(3,1),1)] * 4 + [(1,(3,1),(3,3),0)] * 4
            losses3 = [(0,(3,3),(2,3),1)] * 4 + [(0,(2,3),(3,3),0)] * 4
            losses4 = [(0,(3,3),(1,3),1)] * 4 + [(0,(1,3),(3,3),0)] * 4
            losses = losses1 + losses2 + losses3 + losses4

            # Extras
            extra = wins + losses
            train += extra

        return train, test, analyze

    def append_info(self, samples, inner_4x4=False):
        info_samples = []
        for sample in samples:
            ctx, loc1, loc2, y = sample
            idx1, idx2 = self.loc2idx[loc1], self.loc2idx[loc2]
            # Get congruency: 1=congruent, -1=incongruent, 0=none
            cong = get_congruency(loc1, loc2)
            if not inner_4x4:
                info = {'loc1': loc1,
                        'loc2': loc2,
                        'idx1': idx1,
                        'idx2': idx2,
                        'cong': cong}
            else:
                # HACK (very dangerous but also kind of elegant):
                # change info to align with 4x4 grid for analyses       
                info = {'loc1': self.locs_map[loc1],
                        'loc2': self.locs_map[loc2],
                        'idx1': self.idxs_map[idx1],
                        'idx2': self.idxs_map[idx2],
                        'cong': cong}
            info_samples.append((ctx, loc1, loc2, y, info))
        return info_samples
    
    def is_inner_4x4(self, loc1, loc2):
        assert (self.size % 2) == 0, "Cant analyze inner 4x4 with odd grid size"
        coords = [loc1[0], loc1[1], loc2[0], loc2[1]]
        return all(self.lower <= coord <= self.upper for coord in coords)

class GridDataset(Dataset):
    """
    Dataset used for experiments. Each sample is a tuple (ctx, f1, f2,  y):
        ctx  : context/axis variable, always an int to be embedded
        f1   : face 1, either int to be embedded or an image (1, 64, 64)
        f2   : face 2, either int to be embedded or an image (1, 64, 64) 
        y    : correct answer, always either 0 or 1
    """
    def __init__(self, samples, loc2idx, idx2tensor):
        self.samples = samples # list of all samples
        self.loc2idx = loc2idx # map from locations to indexes
        self.idx2tensor = idx2tensor # map from indexes to tensors

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        sample = self.samples[i]
        ctx, loc1, loc2, y, info = sample
        ctx = torch.tensor(ctx).unsqueeze(0).type(torch.long) # [1]
        idx1 = self.loc2idx[loc1]
        idx2 = self.loc2idx[loc2]
        f1 = self.idx2tensor[idx1].unsqueeze(0) # [1, 1, 64, 64]
        f2 = self.idx2tensor[idx2].unsqueeze(0) # [1, 1, 64, 64]
        y = torch.tensor(y).unsqueeze(0).type(torch.long) # [1]

        return ctx, f1, f2, y, info

# Collate function
def grid_collate(samples):
    # Tensors
    ctx_batch = torch.cat([s[0] for s in samples], dim=0) 
    f1_batch = torch.cat([s[1] for s in samples], dim=0)
    f2_batch = torch.cat([s[2] for s in samples], dim=0)
    y_batch = torch.cat([s[3] for s in samples], dim=0)

    # Info
    info_batch = {k:[] for k in samples[0][4].keys()}
    for s in samples:
        info = s[4]
        for k,v in info.items():
            info_batch[k].append(v)

    return ctx_batch, f1_batch, f2_batch, y_batch, info_batch

# General function for getting data loaders from args
def get_loaders(args):
    # Relevant arguments
    grid_size = args.grid_size
    use_images = args.use_images
    image_dir = args.image_dir
    batch_size = args.bs
    training_regime = args.training_regime
    inner_4x4 = args.inner_4x4

    # Generate data
    grid = GridDataGenerator(training_regime=training_regime, size=grid_size,
                             use_images=use_images, image_dir=image_dir, 
                             inner_4x4=inner_4x4)
    
    # Add useful variables to args for analyses
    args.n_states_a = 16 if inner_4x4 else int(grid_size**2)
    args.loc2idx_a = grid.loc2idx_4x4 if inner_4x4 else grid.loc2idx
    args.idx2tensor_a = grid.idx2tensor_4x4 if inner_4x4 else grid.idx2tensor
    args.corners = grid.corners

    # Create datasets
    train_set = GridDataset(grid.train, grid.loc2idx, grid.idx2tensor)
    test_set = GridDataset(grid.test, grid.loc2idx, grid.idx2tensor)
    analyze_set = GridDataset(grid.analyze, grid.loc2idx, grid.idx2tensor)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=True, collate_fn=grid_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                             shuffle=True, collate_fn=grid_collate)
    analyze_loader = DataLoader(analyze_set, batch_size=batch_size,
                                shuffle=True, collate_fn=grid_collate)

    return train_loader, test_loader, analyze_loader