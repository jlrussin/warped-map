import torch
import torch.nn as nn
import numpy as np

from test import test
from analyze import *

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
    train_accs_cong = [] # accuracies on congruent trial of training set
    train_accs_incong = [] # accuracies on incongruent trials of training set
    test_accs = [] # accuracies on test set
    test_accs_cong = [] # accuracies on congruent trial of test set
    test_accs_incong = [] # accuracies on incongruent trials of test set
    analyze_accs = [] # accuracies on analyze set
    analyze_accs_cong = [] # accuracies on congruent trial of test set
    analyze_accs_incong = [] # accuracies on incongruent trials of analyze set
    analyses = [] # results from all analyses


    # Setup for recording gradient norms and ratio diffs
    congruencies = [] # for tracking congruency of each sample
    grad_ctx = [] # norms of gradients w.r.t. context embedding
    grad_f1 = [] # norms of gradients w.r.t. face1 embedding
    grad_f2 = [] # norms of gradients w.r.t. face2 embedding
    grad_ctx_cong = [] # context, congruent trials
    grad_f1_cong = [] # face1, congruent trials
    grad_f2_cong = [] # face2, congruent trials
    grad_ctx_incong = [] # context, incongruent trials
    grad_f1_incong = [] # face1, incongruent trials
    grad_f2_incong = [] # face2, incongruent trials
    ratio_diffs = [] # distance ratio after taking a step minus before

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

            # Record loss
            train_losses.append(loss.data.item())
            ave_loss.append(loss.data.item())
            
            # Norm of gradients w.r.t. context and face embeddings
            if args.measure_grad_norm:
                grd_ctx = torch.linalg.norm(model.ctx_embed.grad, dim=1)
                grd_f1 = torch.linalg.norm(model.f1_embed.grad, dim=1)
                grd_f2 = torch.linalg.norm(model.f2_embed.grad, dim=1)

                grad_ctx.append(grd_ctx.numpy())
                grad_f1.append(grd_f1.numpy())
                grad_f2.append(grd_f2.numpy())
                
                # Separate into congruent and incongruent
                cong = info['cong'] # 1: congruent, -1: incongruent, 0: neutral
                for ii, c in enumerate(cong):
                    congruencies.append(c)
                    if c==1:
                        grad_ctx_cong.append(grd_ctx[ii].numpy())
                        grad_f1_cong.append(grd_f1[ii].numpy())
                        grad_f2_cong.append(grd_f2[ii].numpy())
                    if c==-1:
                        grad_ctx_incong.append(grd_ctx[ii].numpy())
                        grad_f1_incong.append(grd_f1[ii].numpy())
                        grad_f2_incong.append(grd_f2[ii].numpy())

            # Get measures of warping before taking a step
            if args.sbs_analysis and step_i % args.sbs_every == 0:
                reps_b = collect_representations(model, analyze_loader, args)
                dists_b = calc_dist(args, reps_b)    
                ratio_b = calc_ratio(args, reps_b, dists_b)
                    
            # Take optimizer step
            optimizer.step()

            # Get measures of warping after taking a step
            if args.sbs_analysis and step_i % args.sbs_every == 0:
                cong = info['cong']
                reps_a = collect_representations(model, analyze_loader, args)
                dists_a = calc_dist(args, reps_a)    
                ratio_a = calc_ratio(args, reps_a, dists_a)
                diff_ratio = ratio_a['ratio_hidd'] - ratio_b['ratio_hidd']
                ratio_diffs.append((diff_ratio, cong))
            
            # Log
            if step_i % args.print_every == 0:
                l = np.mean(ave_loss)
                print("Run: {}, Step: {}, Loss: {}".format(run_i, step_i, l))
                ave_loss = []

            # Test and analyze
            final_step = step_i >= args.n_steps
            if step_i % args.analyze_every == 0 or final_step: 
                model.output_seq = True
                # Test on training set
                train_acc_ = test(model, train_loader, args)
                train_acc, train_acc_cong, train_acc_incong = train_acc_
                train_accs.append(train_acc)
                train_accs_cong.append(train_acc_cong)
                train_accs_incong.append(train_acc_incong)
                # Test on testing set
                test_acc_ = test(model, test_loader, args)
                test_acc, test_acc_cong, test_acc_incong = test_acc_
                test_accs.append(test_acc)
                test_accs_cong.append(test_acc_cong)
                test_accs_incong.append(test_acc_incong)
                # Test on analysis set
                analyze_acc_ = test(model, analyze_loader, args)
                analyze_acc, analyze_acc_cong, analyze_acc_incong = analyze_acc_
                analyze_accs.append(analyze_acc)
                analyze_accs_cong.append(analyze_acc_cong)
                analyze_accs_incong.append(analyze_acc_incong)
                model.output_seq = False

                print("Cortical system training accuracy:", train_acc)
                print("Cortical system testing accuracy:", test_acc)
                print("Cortical system analyzing accuracy:", analyze_acc)

                # Gather representations and analyze
                analysis = analyze(model, analyze_loader, args, final_step)

                # Record norms of gradients
                # TODO: not recording correctly - need outer loop lists to append to
                if args.measure_grad_norm:
                    analysis['grad_ctx'] = np.mean(grad_ctx)
                    analysis['grad_f1'] = np.mean(grad_f1)
                    analysis['grad_f2'] = np.mean(grad_f2)

                    analysis['grad_ctx_cong'] = np.mean(grad_ctx_cong)
                    analysis['grad_f1_cong'] = np.mean(grad_f1_cong)
                    analysis['grad_f2_cong'] = np.mean(grad_f2_cong)

                    analysis['grad_ctx_incong'] = np.mean(grad_ctx_incong)
                    analysis['grad_f1_incong'] = np.mean(grad_f1_incong)
                    analysis['grad_f2_incong'] = np.mean(grad_f2_incong)
                    
                    grad_ctx, grad_f1, grad_f2 = [], [], []
                    grad_ctx_cong, grad_f1_cong, grad_f2_cong = [], [], []
                    grad_ctx_incong, grad_f1_incong, grad_f2_incong = [], [], []
                
                analyses.append(analysis)

            # Break after n_steps
            if final_step:
                done = True 
                break
            step_i+= 1

    results = {'train_losses': train_losses,
               'train_acc': train_acc,
               'train_acc_cong': train_acc_cong,
               'train_acc_incong': train_acc_incong,
               'test_acc': test_acc,
               'test_acc_cong': test_acc_cong,
               'test_acc_incong': test_acc_incong,
               'analyze_acc': analyze_acc,
               'analyze_acc_cong': analyze_acc_cong,
               'analyze_acc_incong': analyze_acc_incong}

    return results, analyses