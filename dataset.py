import torch
import random
import numpy as np
from itertools import permutations 
from torch.utils.data import Dataset, DataLoader 
from torchvision.datasets import ImageFolder 
from torchvision.transforms import Compose, Grayscale, ToTensor

class Grid:
    def __init__(self):
        self.size = 4 # 4x4 grid (fixed)
        
        # self.training_day = 'day3'
        # Generate locations (tuples) for each state in grid
        locs = [(i,j) for i in range(self.size) for j in range(self.size)]
        
        # Define groups
        group1 = [(1,0),(2,0),(0,1),(3,1),(1,2),(2,2),(0,3),(3,3)]
        group2 = [loc for loc in locs if loc not in group1]
        
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
        day1 = []
        day2 = []
        # group 1
        for pair, d1, d2 in zip(all_within1, d_within1_ctx1, d_within1_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((f1, f2, 0, y)) # (F1, F2, ctx, y)
                day1.append((f1, f2, 0, y)) # (F1, F2, ctx, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((f1, f2, 1, y)) # (F1, F2, ctx, y)
                day1.append((f1, f2, 0, y)) # (F1, F2, ctx, y)
        # group 2
        for pair, d1, d2 in zip(all_within2, d_within2_ctx1, d_within2_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            if abs(d1) == 1:
                y = int(d1 > 0)
                within.append((f1, f2, 0, y)) # (F1, F2, context, y)
                day2.append((f1, f2, 0, y)) # (F1, F2, ctx, y)
            if abs(d2) == 1:
                y = int(d2 > 0)
                within.append((f1, f2, 1, y)) # (F1, F2, context, y)
                day2.append((f1, f2, 0, y)) # (F1, F2, ctx, y)
        
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
            between.append((f1, f2, 0, y))
        for pair, d2 in zip(between_ctx2, d_between_ctx2):
            f1 = pair[0]
            f2 = pair[1]
            y = int(d2 > 0)
            between.append((f1, f2, 1, y))
        
        # Compile training set
        train = within + between
        random.shuffle(train)
        
        # Get all test pairs separated by a hub
        test_pairs_ctx1 = []
        for pair in between_ctx1:
            locb1 = pair[0]
            locb2 = pair[1]
            for sample in within:
                if sample[2] != 0:
                    continue
                locw1 = sample[0]
                locw2 = sample[1]
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
                if sample[2] != 1:
                    continue
                locw1 = sample[0]
                locw2 = sample[1]
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
        
        # Remove pairs that include an axis/context 1 hub
        test_pairs_nohub_ctx1 = []
        for loc1, loc2 in test_pairs_ctx1:
            if loc1 not in hubs_ctx1 and loc2 not in hubs_ctx1:
                test_pairs_nohub_ctx1.append((loc1, loc2))
        # Remove pairs that include an axis/context 2 hub
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
                test.append((f1, f2, 0, y))
        for pair in test_pairs_nohub_ctx2:
            loc1 = pair[0]
            loc2 = pair[1]
            d = loc1[1] - loc2[1]
            if d != 0:
                f1 = loc1
                f2 = loc2
                y = int(d > 0)
                test.append((f1, f2, 1, y))
                
        # Get relevant hub pairs for each sample in test set
        hub_sample_ids = [] # ids of relevant samples
        for sample in test:
            loc1 = sample[0]
            loc2 = sample[1]
            ctx = sample[2]
            possible_hubs_f1 = []
            possible_hubs_f2 = []
            for i, s in enumerate(train):
                if s[2] == ctx:
                    if loc1 == s[0]:
                        possible_hubs_f1.append((s[1], i))
                    elif loc1 == s[1]:
                        possible_hubs_f1.append((s[0], i))
                    if loc2 == s[0]:
                        possible_hubs_f2.append((s[1], i))
                    elif loc2 == s[1]:
                        possible_hubs_f2.append((s[0], i))
            sample_ids = []
            for loc, i in possible_hubs_f1:
                if loc in [p[0] for p in possible_hubs_f2]:
                    sample_ids.append(i)
            for loc, i in possible_hubs_f2:
                if loc in [p[0] for p in possible_hubs_f1]:
                    sample_ids.append(i)
            hub_sample_ids.append(list(set(sample_ids)))
        
        # Generate all the permutations of the pairs
        all_perms = []
        idxs = [idx for idx in range(len(locs))]
        # loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}
        idx2loc = {idx:loc for idx, loc in zip(idxs, locs)}

        for idx1, idx2 in permutations(idxs, 2):
            loc1, loc2 = idx2loc[idx1], idx2loc[idx2]
            d1 = loc1[0] - loc2[0]
            d2 = loc1[1] - loc2[1]
            f1 = loc1
            f2 = loc2
            for ctx in range(2):
                # axis/context = 0 is x-axis and ctx=1 is y-axis
                # removing ties
                # ties: pairs with the same rank on the relevant axis
                if ((d1!=0) & (ctx==0)):
                    y = int(d1 > 0)
                    all_perms.append((f1, f2, ctx, y))
                elif ((d2!=0) & (ctx==1)):
                    y = int(d2 > 0)
                    all_perms.append((f1, f2, ctx, y))
                
        # Save variables
        self.locs = locs
        self.group1 = group1
        self.group2 = group2
        self.within = within
        self.day1 = day1
        self.day2 = day2
        self.between = between
        # if self.training_day == 'day1':
        #     self.train = day1
        # elif self.training_day == 'day1_day2':
        #     self.train = day1+day2
        # elif self.training_day == 'day3':
        #     self.train = train
        self.train = train
        self.test = test
        self.all_perms = all_perms
        self.hub_sample_ids = hub_sample_ids
        
        self.n_train = len(self.train)
        self.n_test = len(self.test)
        self.n_all_perms = len(self.all_perms)

class GridDataset(Dataset):
    """
    Dataset used for cortical system. Each sample is a tuple (f1, f2, context, y):
        f1   : face 1, either int to be embedded or an image (1, 64, 64)
        f2   : face 2, either int to be embedded or an image (1, 64, 64)
        ctx  : context/axis variable, always an int to be embedded 
        y    : correct answer, always either 0 or 1
    """
    def __init__(self, testing, analyzing, use_images, image_dir = None):
        self.testing = testing       # use test set
        self.analyzing = analyzing
        self.use_images = use_images # use images rather than one-hot vectors
        self.image_dir = image_dir   # directory with images
        self.grid = Grid()
        # self.grid.training_day = training_day

        # Create 1 fixed mapping from locs to idxs
        locs = self.grid.locs 
        idxs = [idx for idx in range(len(locs))]
        self.loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}
        self.n_states = len(idxs)

        # Prepare tensors for each idx
        idx2tensor = {}
        if self.use_images:
            # Tensors are images
            transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
            face_images = ImageFolder(self.image_dir, transform)
            for idx in idxs:
                idx2tensor[idx] = face_images[idx][0] # [1, 64, 64]
        else:
            # Tensors are one-hot vectors
            for idx in idxs:
                idx2tensor[idx] = torch.tensor(idx).type(torch.long) # [16]
        self.idx2tensor = idx2tensor
    
    def __len__(self):
        if self.testing:
            return len(self.grid.test)
        elif self.analyzing:
            return len(self.grid.all_perms)
        else:
            return len(self.grid.train)
    
    def __getitem__(self, i):
        if self.testing:
            s = self.grid.test[i]
        elif self.analyzing:
            s = self.grid.all_perms[i]    
        else:
            s = self.grid.train[i]
        loc1 = s[0]
        loc2 = s[1]
        idx1 = self.loc2idx[loc1]
        idx2 = self.loc2idx[loc2]
        f1 = self.idx2tensor[idx1].unsqueeze(0) # [1, 1, 64, 64]
        f2 = self.idx2tensor[idx2].unsqueeze(0) # [1, 1, 64, 64]
        ctx = torch.tensor(s[2]).type(torch.long).unsqueeze(0) # [1]
        y = torch.tensor(s[3]).unsqueeze(0).unsqueeze(0) # [1]
        y = y.type(torch.long)
        return f1, f2, ctx, y, idx1, idx2 # ToDo: maybe change this to context, f1, f2, ...

class WineGrid:
    def __init__(self, N_responses, N_contexts, balanced = False):
        self.size = 4
        self.N_responses = N_responses
        self.N_contexts = N_contexts
        if balanced:
            msg = 'balancing only implemented for one response, two contexts'
            assert N_responses == 'one' and N_contexts == 2, msg
        self.balanced = balanced

        locs = [(i,j) for i in range(self.size) for j in range(self.size)]
        # Generate all the permutations of the pairs
        all_perms = []
        train = []
        test = []
        idxs = [idx for idx in range(len(locs))]
        # loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}
        idx2loc = {idx:loc for idx, loc in zip(idxs, locs)}
        for idx1, idx2 in permutations(idxs, 2):
            loc1, loc2 = idx2loc[idx1], idx2loc[idx2]
            f1 = loc1
            f2 = loc2
            # Competence on x-axis; Populatiry on y-axis
            # ties: d==0, pairs with the same rank (r1==r2) on the both axis
            if self.N_responses == 'two':
                ctx = -1
                # M:C and V:P - so dM comes from x-axis, dV from y-axis
                d1 = loc1[0] - loc2[0] # competence rank
                d2 = loc1[1] - loc2[1] # popularity rank
                if ((d1!=0) & (d2!=0)):
                    y1 = int(d1 > 0)
                    y2 = int(d2 > 0)
                    all_perms.append((f1, f2, ctx, y1, y2))
                    if (abs(d1)>1) & (abs(d2)>1):
                        test.append((f1, f2, ctx, y1, y2))
                    elif (abs(d1)==1) & (abs(d2)==1):
                        train.append((f1, f2, ctx, y1, y2)) 
            elif self.N_responses == 'one':
                y2 = -1
                ctx=0
                r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                self.add_sample(all_perms, test, train, r1, r2, loc1, loc2, ctx)
                ctx=1
                r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                self.add_sample(all_perms, test, train, r1, r2, loc1, loc2, ctx)
                if ((self.N_contexts==4) or (self.N_contexts==8)):     
                    ctx=2
                    r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                    self.add_sample(all_perms, test, train, 
                                    r1, r2, loc1, loc2, ctx)
                    ctx=3
                    r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                    self.add_sample(all_perms, test, train, 
                                    r1, r2, loc1, loc2, ctx)
                    if self.N_contexts==8:
                        ctx=4
                        r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                        self.add_sample(all_perms, test, train, 
                                        r1, r2, loc1, loc2, ctx)
                        ctx=5
                        r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                        self.add_sample(all_perms, test, train,
                                        r1, r2, loc1, loc2, ctx)
                        ctx=6
                        r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                        self.add_sample(all_perms, test, train, 
                                        r1, r2, loc1, loc2, ctx)
                        ctx=7
                        r1, r2 = self.ctx_to_r(ctx, loc1, loc2) 
                        self.add_sample(all_perms, test, train, 
                                        r1, r2, loc1, loc2, ctx)
                    
        # Save variables
        self.locs = locs
        self.all_perms = all_perms
        self.train = train
        self.test = test
        self.n_train = len(self.train)
        self.n_test = len(self.test)
        self.n_all_perms = len(self.all_perms)

        if balanced:
            self.balance_train()

    def ctx_to_r(self, ctx, loc1, loc2):
        loc1 = (loc1[0]+1, loc1[1]+1)
        loc2 = (loc2[0]+1, loc2[1]+1)
        if ctx==0:
            r1 = loc1[0]
            r2 = loc2[0]
        elif ctx==1:
            r1 = loc1[1]
            r2 = loc2[1]
        elif ctx==2:
            r1 = 5-loc1[0]
            r2 = 5-loc2[0]
        elif ctx==3:
            r1 = 5-loc1[1]
            r2 = 5-loc2[1]
        elif ctx==4:
            r1 = loc1[0]*loc1[1]
            r2 = loc2[0]*loc2[1]
        elif ctx==5:
            r1 = loc1[0]*(5-loc1[1])
            r2 = loc2[0]*(5-loc2[1])
        elif ctx==6:
            r1 = (5-loc1[0])*(5-loc1[1])
            r2 = (5-loc2[0])*(5-loc2[1])
        elif ctx==7:
            r1 = (5-loc1[0])*loc1[1]
            r2 = (5-loc2[0])*loc2[1]
        return r1, r2

    def add_sample(self, all_perms, test, train, r1, r2, f1, f2, ctx, y2=-1):
        d = r1 - r2
        if d!=0:
            y1 = int(d > 0)
            all_perms.append((f1, f2, ctx, y1, y2))
            if abs(d)>1:
                test.append((f1, f2, ctx, y1, y2))
            elif abs(d)==1:
                train.append((f1, f2, ctx, y1, y2))
    
    def balance_train(self):
        # Extra wins
        wins1 = [((0,0),(0,1),1,0,-1)] * 4 + [((0,1),(0,0),1,1,-1)] * 4
        wins2 = [((0,0),(0,2),1,0,-1)] * 4 + [((0,2),(0,0),1,1,-1)] * 4
        wins3 = [((0,0),(1,0),0,0,-1)] * 4 + [((1,0),(0,0),0,1,-1)] * 4
        wins4 = [((0,0),(2,0),0,0,-1)] * 4 + [((2,0),(0,0),0,1,-1)] * 4
        wins = wins1 + wins2 + wins3 + wins4

        # Extra losses
        losses1 = [((3,3),(3,2),1,1,-1)] * 4 + [((3,2),(3,3),1,0,-1)] * 4
        losses2 = [((3,3),(3,1),1,1,-1)] * 4 + [((3,1),(3,3),1,0,-1)] * 4
        losses3 = [((3,3),(2,3),0,1,-1)] * 4 + [((2,3),(3,3),0,0,-1)] * 4
        losses4 = [((3,3),(1,3),0,1,-1)] * 4 + [((1,3),(3,3),0,0,-1)] * 4
        losses = losses1 + losses2 + losses3 + losses4

        # Extras
        extra = wins + losses
        self.train += extra


class WineGridDataset(Dataset):
    """
    Dataset used for cortical system. Each sample is a tuple (f1, f2, ctx, y):
        f1   : face 1, either int to be embedded or an image (1, 64, 64)
        f2   : face 2, either int to be embedded or an image (1, 64, 64)
        ctx  : context/axis variable, always an int to be embedded 
        y1   : correct answer for response 1, always either 0 or 1
        y2   : correct answer for response 2, always either 0 or 1
    """
    def __init__(self, testing, analyzing, N_contexts, N_responses,
                 use_images, image_dir = None, balanced = False):
        self.testing = testing       # use test set
        self.analyzing = analyzing   # use all the permutations - random split 
        self.use_images = use_images # use images rather than one-hot vectors
        self.image_dir = image_dir   # directory with images
        self.balanced = balanced     # faces get same wins/losses
        # msg = 'N_responses is a string (e.g., "one") and N_contexts is an int'
        # assert (N_responses is not None) and (N_contexts is not None) and isinstance(N_responses, str) and isinstance(N_contexts, int), msg
        self.grid = WineGrid(N_responses, N_contexts, balanced)
        

        # Create 1 fixed mapping from locs to idxs
        locs = self.grid.locs 
        idxs = [idx for idx in range(len(locs))]
        self.loc2idx = {loc:idx for loc, idx in zip(locs, idxs)}
        self.n_states = len(idxs)

        # Prepare tensors for each idx
        idx2tensor = {}
        if self.use_images:
            # Tensors are images
            transform = Compose([Grayscale(num_output_channels=1), ToTensor()])
            face_images = ImageFolder(self.image_dir, transform)
            for idx in idxs:
                idx2tensor[idx] = face_images[idx][0] # [1, 64, 64]
        else:
            # Tensors are one-hot vectors
            for idx in idxs:
                idx2tensor[idx] = torch.tensor(idx).type(torch.long) # [16]
        self.idx2tensor = idx2tensor
    
    def __len__(self):
        if self.testing:
            return len(self.grid.test)
        elif self.analyzing:
            return len(self.grid.all_perms)
        else:
            return len(self.grid.train)
    
    def __getitem__(self, i):
        if self.testing:
            s = self.grid.test[i]
        elif self.analyzing:
            s = self.grid.all_perms[i]    
        else:
            s = self.grid.train[i]
        loc1 = s[0]
        loc2 = s[1]
        idx1 = self.loc2idx[loc1]
        idx2 = self.loc2idx[loc2]
        f1 = self.idx2tensor[idx1].unsqueeze(0) # [1, 1, 64, 64]
        f2 = self.idx2tensor[idx2].unsqueeze(0) # [1, 1, 64, 64]
        ctx = torch.tensor(s[2]).type(torch.long).unsqueeze(0) # [1]
        y1 = torch.tensor(s[3]).unsqueeze(0).unsqueeze(0) # [1]
        y1 = y1.type(torch.long)
        y2 = torch.tensor(s[4]).unsqueeze(0).unsqueeze(0) # [1]
        y2 = y2.type(torch.long)
        
        return f1, f2, ctx, y1, y2, idx1, idx2

# Collate functions

def grid_collate(samples):
    f1_batch = torch.cat([s[0] for s in samples], dim=0)
    f2_batch = torch.cat([s[1] for s in samples], dim=0)
    ctx_batch = torch.cat([s[2] for s in samples], dim=0)
    y_batch = torch.cat([s[3] for s in samples], dim=0)
    idx1_batch = [s[4] for s in samples]
    idx2_batch = [s[5] for s in samples]
    return f1_batch, f2_batch, ctx_batch, y_batch, idx1_batch, idx2_batch

def wine_grid_collate(samples):
    f1_batch = torch.cat([s[0] for s in samples], dim=0)
    f2_batch = torch.cat([s[1] for s in samples], dim=0)
    ctx_batch = torch.cat([s[2] for s in samples], dim=0)
    y1_batch = torch.cat([s[3] for s in samples], dim=0)
    y2_batch = torch.cat([s[4] for s in samples], dim=0)
    idx1_batch = [s[5] for s in samples]
    idx2_batch = [s[6] for s in samples]
    return f1_batch, f2_batch, ctx_batch, y1_batch, y2_batch, idx1_batch, idx2_batch

def get_loaders(args):
    batch_size = args.bs
    use_images = args.use_images
    image_dir = args.image_dir
    task = args.task
    balanced = args.balanced
    
    if task == 'face_task':
        # Train
        train_dataset = GridDataset(testing=False, analyzing=False, 
                                 use_images=use_images, image_dir=image_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, collate_fn=grid_collate)
        # Test
        test_dataset = GridDataset(testing=True, analyzing=False, 
                                use_images=use_images, image_dir=image_dir)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=True, collate_fn=grid_collate)
        # Analyze
        analyze_dataset = GridDataset(testing=False, analyzing=True, 
                                   use_images=use_images, image_dir=image_dir)
        analyze_loader = DataLoader(analyze_dataset, batch_size=1, 
                                    shuffle=False, collate_fn=grid_collate)
    elif task == 'wine_task':
    # ToDo: check if I need to check the N_responses == 'two' here
        # Train
        train_dataset = WineGridDataset(testing=False, analyzing=False, 
                                     N_contexts=N_contexts, N_responses=N_responses,
                                     use_images=use_images, image_dir=image_dir,
                                     balanced=balanced)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=wine_grid_collate)
        # Test
        test_dataset = WineGridDataset(testing=True, analyzing=False,
                                    N_contexts=N_contexts, N_responses=N_responses,
                                    use_images=use_images, image_dir=image_dir)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=wine_grid_collate)
        # Analyze
        analyze_dataset = WineGridDataset(testing=False, analyzing=True, 
                                       N_contexts=N_contexts, N_responses=N_responses,
                                       use_images=use_images, image_dir=image_dir)
        analyze_loader = DataLoader(analyze_dataset, batch_size=1, 
                                    shuffle=False, collate_fn=wine_grid_collate)        
    
    train_data = (train_dataset, train_loader)
    test_data = (test_dataset, test_loader)
    analyze_data = (analyze_dataset, analyze_loader)
    return train_data, test_data, analyze_data
