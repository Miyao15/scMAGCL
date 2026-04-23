import argparse
import warnings
import importlib
import torch
import numpy as np
import pandas as pd
import os
import gc
import time
from utils import setup_seed, loader_construction, evaluate, device
from scSimGCL import Model
from sklearn.cluster import KMeans
from config import config

try:
    import psutil
except Exception:
    psutil = None

# Thread control and KMeans warning suppression for Windows/MKL environments
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)

def train(train_loader,
          test_loader,
          input_dim,
          graph_head,
          phi,
          gcn_dim,
          mlp_dim,
          prob_feature,
          prob_edge,
          tau,
          alpha,
          beta,
          lambda_cl,
          dropout,
          lr,
          seed,
          epochs,
          device,
          knn_k=15,
          phi1=None,
          lambda_byol=1.0):
    """
    Core training procedure for the scSimGCL model.
    Reconstruction loss (MAE) is utilized for optimization but external 
    imputation metrics are not explicitly calculated here.
    """
    model = Model(input_dim=input_dim, graph_head=graph_head, phi=phi, gcn_dim=gcn_dim,
                  mlp_dim=mlp_dim, prob_feature=prob_feature, prob_edge=prob_edge, tau=tau,
                  alpha=alpha, beta=beta, dropout=dropout,
                  phi1=(config['phi1'] if phi1 is None else phi1),
                  use_byol=config['use_byol'],
                  byol_hidden_dim=config['byol_hidden_dim'],
                  byol_output_dim=config['byol_output_dim'],
                  momentum_tau=config['momentum_tau']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    setup_seed(seed)
    
    best_epoch = 0
    min_loss = float('inf')
    best_state = None

    print(f"Initializing training: {epochs} epochs, learning rate {lr}")
    
    for epoch in range(epochs):
        model.train()
        for step, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            _, x_imp, loss_cl, loss_byol = model(batch_x)
            
            # Reconstruction loss (MAE) - Essential for training
            mask = (batch_x != 0).float()
            loss_mae = torch.nn.L1Loss()(mask * x_imp, mask * batch_x)
            
            # Total optimization objective
            loss = loss_mae + lambda_cl * loss_cl + lambda_byol * loss_byol
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_network()

        # Validation phase to monitor loss and save the best latent embeddings
        model.eval()
        epoch_losses = []
        epoch_z, epoch_y = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                bz, xi, lcl, lby = model(batch_x)
                
                mask = (batch_x != 0).float()
                loss_mae = torch.nn.L1Loss()(mask * xi, mask * batch_x)
                loss = loss_mae + lambda_cl * lcl + lambda_byol * lby
                
                epoch_losses.append(loss.item())
                epoch_z.append(bz.cpu().numpy())
                epoch_y.append(batch_y.numpy())

        avg_loss = np.mean(epoch_losses)
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_epoch = epoch
            best_state = (epoch_z, epoch_y)
            print(f"Epoch {epoch:03d}: Validation Loss = {avg_loss:.4f} (Best)")
        
        if (epoch + 1) % 10 == 0:
            gc.collect()

    print(f"Training finalized. Optimal Epoch: {best_epoch}, Best Loss: {min_loss:.4f}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return best_epoch, min_loss, best_state[0], best_state[1]

def test(z_test_epoch,
         y_test_epoch,
         n_clusters,
         seed):
    """
    Performs clustering evaluation on the latent embeddings.
    """
    z_test = np.vstack(z_test_epoch)
    y_test = np.hstack(y_test_epoch)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_pred = kmeans.labels_

    adata, method = None, None
    try:
        import anndata
        adata = anndata.AnnData(X=z_test)
        adata.obs['cell_type'] = y_test.astype(str)
        adata.obsm['X'] = adata.X
        method = 'X'
    except ImportError:
        pass

    acc, f1, nmi, ari, homo, comp, clisi, casw = evaluate(y_test, y_pred, adata=adata, method=method)
    results = {'CA': acc, 'NMI': nmi, 'ARI': ari, 'clisi': clisi, 'casw': casw}

    del kmeans, adata, z_test, y_test, y_pred
    gc.collect()
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scSimGCL: Graph Contrastive Learning for Single-Cell Analysis")
    parser.add_argument("--data_path", type=str, default="Zeisel.h5")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=config['epochs'])
    parser.add_argument("--sweep_param", type=str, default=None)
    parser.add_argument("--sweep_values", type=str, default=None)
    args = parser.parse_args()

    sweep_values = [float(v) for v in args.sweep_values.split(",")] if args.sweep_param and args.sweep_values else [None]

    for sweep_val in sweep_values:
        params = {
            'phi1': config['phi1'],
            'tau': config['tau'],
            'lambda_cl': config['lambda_cl'],
            'lambda_byol': config['lambda_byol']
        }
        if args.sweep_param in params:
            params[args.sweep_param] = sweep_val

        print(f"\nConfiguration: {args.sweep_param if args.sweep_param else 'Default'} = {sweep_val}")
        run_metrics = []
        runtime_stats = []

        for i in range(args.n_runs):
            print(f"Run {i+1}/{args.n_runs} (Seed: {config['seed'] + i})")
            
            start_run = time.perf_counter()
            cur_seed = config['seed'] + i
            
            train_loader, test_loader, input_dim = loader_construction(args.data_path)

            best_epoch, min_loss, z_test, y_test = train(
                train_loader, test_loader, input_dim, config['graph_head'], config['phi'], config['gcn_dim'], 
                config['mlp_dim'], config['prob_feature'], config['prob_edge'], params['tau'], config['alpha'], 
                config['beta'], params['lambda_cl'], config['dropout'], config['lr'], cur_seed, 
                args.epochs, device, phi1=params['phi1'], lambda_byol=params['lambda_byol'])

            metrics = test(z_test, y_test, args.n_clusters, cur_seed)
            print(f"Metrics: CA={metrics['CA']:.4f}, NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}")
            
            run_metrics.append(metrics)
            runtime_stats.append(time.perf_counter() - start_run)

            del z_test, y_test, train_loader, test_loader
            gc.collect()

        # Final Summary
        print("\n" + "-"*40)
        print(f"Summary Statistics ({args.n_runs} runs):")
        for m in ['CA', 'NMI', 'ARI']:
            vals = [r[m] for r in run_metrics]
            print(f"{m:4}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"Mean Runtime: {np.mean(runtime_stats):.2f}s")
        print("-"*40)

    print("Experiment workflow finished successfully.")