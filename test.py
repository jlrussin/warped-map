import numpy as np
import torch 

def test(model, loader, args):
    model.eval()
    with torch.no_grad():
        correct = []        # list of booleans indicating correct answers
        cong_correct = []   # correct for congruent trials only
        incong_correct = [] # correct for incongruent trials only
        for batch in loader:
            # Data
            ctx, f1, f2,  y, info = batch # context, face1, face2, y, info
            ctx = ctx.to(args.device)
            f1 = f1.to(args.device)
            f2 = f2.to(args.device)
            y = y.to(args.device)
            congs = info['cong']

            # Run model
            y_hat, out = model(f1, f2, ctx) 
            # y_hat: [batch, output_dim] or [batch, seq_len, output_dim]

            # Get predictions with argmax
            if model.output_seq: # model returns predictions for each time step
                seq_len = y_hat.shape(1)      
                y = y.repeat(1, seq_len) # [batch, seq_len]
                preds = torch.argmax(y_hat, dim=2) # [batch, seq_len]
            else:
                preds = torch.argmax(y_hat, dim=1) # [batch]

            # Compute whether predictions are correct
            c = (preds == y) # [batch] or [batch, seq_len]
            c = c.cpu().numpy() # [batch] or [batch, seq_len]
            c = [c_i for c_i in c] # len = batch: scalars or seq_len arrays
            correct += c

            # Seperate into congruent and incongruent
            for i, cong in enumerate(congs):
                if cong==1:
                    cong_correct.append(c[i])
                elif cong==-1:
                    incong_correct.append(c[i])

    # Compute accuracy    
    acc = np.mean(correct, axis=0)
    cong_acc = np.mean(cong_correct, axis=0)
    incong_acc = np.mean(incong_correct, axis=0)

    model.train()
    return acc, cong_acc, incong_acc
