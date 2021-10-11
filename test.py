import torch 
import numpy as np

def test(model, loader, args):
    model.eval()
    with torch.no_grad():
        correct = []        # list of booleans indicating correct answers
        cong_correct = []   # correct for congruent trials only
        incong_correct = [] # correct for incongruent trials only
        loc1_ctx0 = [[]]*args.grid_size # correct for each level of loc1 in ctx0
        loc1_ctx1 = [[]]*args.grid_size # correct for each level of loc1 in ctx1
        for batch in loader:
            # Data
            ctx, f1, f2,  y, info = batch # context, face1, face2, y, info
            ctx = ctx.to(args.device)
            f1 = f1.to(args.device)
            f2 = f2.to(args.device)
            y = y.to(args.device)
            locs1 = info['loc1']
            congs = info['cong']

            # Run model
            y_hat, out = model(ctx, f1, f2) 
            # y_hat: [batch, output_dim] or [batch, seq_len, output_dim]

            # Get predictions with argmax
            preds = torch.argmax(y_hat, dim=1) # [batch]

            # Compute whether predictions are correct
            c = (preds == y) # [batch] or [batch, seq_len]
            c = c.cpu().numpy() # [batch] or [batch, seq_len]
            c = [c_i for c_i in c] # len = batch: scalars or seq_len arrays
            correct += c

            # Seperate into congruent and incongruent
            for c_i, cong in zip(c, congs):
                if cong==1:
                    cong_correct.append(c_i)
                elif cong==-1:
                    incong_correct.append(c_i)
            
            # Separate into levels of loc1
            for c_i, ctx_i, loc1 in zip(c, ctx, locs1):
                if ctx_i == 0:
                    loc1_ctx0[loc1[0]].append(c_i)
                elif ctx_i == 1:
                    loc1_ctx1[loc1[1]].append(c_i)

    # Compute accuracy    
    acc = np.mean(correct, axis=0)
    cong_acc = np.mean(cong_correct, axis=0)
    incong_acc = np.mean(incong_correct, axis=0)
    loc1_ctx0_acc = []
    loc1_ctx1_acc = []
    for c0, c1 in zip(loc1_ctx0, loc1_ctx1):
        loc1_ctx0_acc.append(np.mean(c0))
        loc1_ctx1_acc.append(np.mean(c1))

    results = {'acc': acc,
               'cong_acc': cong_acc,
               'incong_acc': incong_acc,
               'loc1_ctx0_acc': loc1_ctx0_acc,
               'loc1_ctx1_acc': loc1_ctx1_acc}

    model.train()
    return results
