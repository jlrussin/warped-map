import torch
import torch.nn as nn
import numpy as np

from test import test
from analyze import *
from utils import log

def train(run_i, model, data, args):
    # Data
    train_loader, test_loader, analyze_loader = data

    # Model
    model.train()

    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Setup for recording loss, testing results, and analyses
    train_losses = [] # for recording all train losses
    ave_loss = [] # running average loss for printing
    train_accs = [] # accuracies on training set
    test_accs = [] # accuracies on test set
    analyze_accs = [] # accuracies on analyze set
    analyses = [] # results from all analyses
    congruencies = [] # for tracking congruency of each sample
    grad_norms = [] # for recording norms of gradients w.r.t. embeddings

    # Training loop
    step_i = 0 # current gradient step
    done = False # done when step_i >= n_steps
    while not done:
        for batch in train_loader:
            optimizer.zero_grad()

            # Data
            ctx, f1, f2, y, info = batch # context, face1, face2, y, info
            ctx = ctx.to(args.device)
            f1 = f1.to(args.device)
            f2 = f2.to(args.device)
            y = y.to(args.device)

            # Model
            y_hat, out = model(ctx, f1, f2)

            # Loss
            loss = loss_fn(y_hat, y)

            # Retain gradients to be measured later
            if args.measure_grad_norm:
                model.ctx_embed.retain_grad()
                model.f1_embed.retain_grad()
                model.f2_embed.retain_grad()
            
            # Backward
            loss.backward()

            # Record loss and cong
            ave_loss.append(loss.data.item())
            cong = info['cong'] # 1: congruent, -1: incongruent, 0: neutral
            congruencies.append(cong)
            
            # Norm of gradients w.r.t. context and face embeddings
            if args.measure_grad_norm:
                grad_norm = measure_grad_norms(model)
                grad_norms.append(grad_norm)
                    
            # Take optimizer step
            optimizer.step()
            
            # Log
            if step_i % args.print_every == 0:
                l = np.mean(ave_loss)
                train_losses.append(l)
                print("Run: {}, Step: {}, Loss: {}".format(run_i, step_i, l))
                ave_loss = []

            # Test
            final_step = step_i >= (args.n_steps - 1)
            if (step_i % args.test_every == 0) or final_step: 
                # Test on training set
                train_acc = test(model, train_loader, args)
                train_accs.append(train_acc)
                # Test on testing set
                test_acc = test(model, test_loader, args)
                test_accs.append(test_acc)
                # Test on analysis set
                analyze_acc = test(model, analyze_loader, args)
                analyze_accs.append(analyze_acc)
                # Log
                log(train_acc['acc'], test_acc['acc'], analyze_acc['acc'])
            
            # Analyze
            if (step_i % args.analyze_every == 0) or final_step:
                # Gather representations and analyze
                analysis = analyze(model, analyze_loader, args, final_step)
                analyses.append(analysis)

            # Break after n_steps
            if final_step:
                done = True 
                break
            step_i += 1

    results = {'train_losses': train_losses,
               'train_accs': train_accs,
               'test_accs': test_accs,
               'analyze_accs': analyze_accs}

    return results, analyses