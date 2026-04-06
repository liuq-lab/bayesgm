import yaml
import argparse
import numpy as np
import sys
import json
from pathlib import Path

import pandas as pd
from bayesgm.models import CausalBGM, CausalBGM_IV, BGM, MNISTBGM, BGM_MNAR
from bayesgm.datasets import (
    simulate_mnar_factor_data,
    simulate_z_hetero,
    Sim_Hirano_Imbens_sampler, 
    Semi_acic_sampler,
    simulate_demand_design_iv,
    make_demand_design_grid
)
from bayesgm.utils import benchmark_mnar_imputers, rmse_on_missing_entries
from sklearn.model_selection import train_test_split
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices([], 'GPU')  # 显式禁止使用 GPU

def _fit_standardizer(data):
    mean = np.mean(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(data, axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return {"mean": mean, "scale": scale}


def _transform(data, stats):
    return ((data - stats["mean"]) / stats["scale"]).astype(np.float32)


def _inverse_transform(data, stats):
    return (data * stats["scale"] + stats["mean"]).astype(np.float32)


def _summarize_ranges(train):
    print("Observed data ranges before normalization:")
    for key in ("x", "y", "v", "w"):
        data = np.asarray(train[key], dtype=np.float32)
        print(
            f"  {key}: min={float(np.min(data)):.4f}, max={float(np.max(data)):.4f}, "
            f"mean={float(np.mean(data)):.4f}, std={float(np.std(data)):.4f}"
        )

def _standardize_demand_design_data(train, grid):
    stats = {key: _fit_standardizer(train[key]) for key in ("x", "y", "v", "w")}
    train_std = {
        "x": _transform(train["x"], stats["x"]),
        "y": _transform(train["y"], stats["y"]),
        "v": _transform(train["v"], stats["v"]),
        "w": _transform(train["w"], stats["w"]),
        "y_struct": train["y_struct"],
    }
    grid_std = {
        "x": _transform(grid["x"], stats["x"]),
        "v": _transform(grid["v"], stats["v"]),
        "y_struct": grid["y_struct"],
    }
    return train_std, grid_std, stats


def _make_structural_monitor_callback(
    grid_x,
    grid_v,
    y_true,
    latent_method="encoder",
    y_stats=None,
    additional_methods=None,
):
    methods = [latent_method]
    for method in additional_methods or ():
        if method not in methods:
            methods.append(method)

    def callback(model, stage, epoch, metrics):
        results = {"structural_latent_method": latent_method}
        for method in methods:
            data_y_pred = model.predict_structural(
                grid_x,
                grid_v,
                latent_method=method,
            )
            if y_stats is not None:
                data_y_pred = _inverse_transform(data_y_pred, y_stats)
            structural_mse = float(np.mean((y_true - data_y_pred) ** 2))    
            results[f"structural_mse_{method}"] = structural_mse
        results["structural_mse"] = results[f"structural_mse_{latent_method}"]
        return results

    return callback


def _print_training_history(history):
    if not history:
        return

    structural_keys = sorted(
        {
            key
            for record in history
            for key in record
            if key.startswith("structural_mse_")
        }
    )

    print("\nTraining metric history")
    header = f"{'stage':<12} {'epoch':>6} {'outcome':>8} {'mse_x':>12} {'mse_y':>12} {'mse_v':>12}"
    for key in structural_keys:
        method = key.removeprefix("structural_mse_")
        header += f" {method:>14}"
    print(header)
    print("-" * len(header))
    for record in history:
        epoch = "-" if record["epoch"] is None else str(record["epoch"])
        row = (
            f"{record['stage']:<12} {epoch:>6} {str(record['include_outcome']):>8} "
            f"{record['mse_x']:>12.6f} {record['mse_y']:>12.6f} {record['mse_v']:>12.6f}"
        )
        for key in structural_keys:
            value = record.get(key)
            row += f" {('-' if value is None else f'{value:.6f}'):>14}"
        print(row)

    if structural_keys:
        for key in structural_keys:
            method = key.removeprefix("structural_mse_")
            method_records = [record for record in history if key in record]
            best_record = min(method_records, key=lambda r: r[key])
            last_record = method_records[-1]
            print(
                "\nBest structural checkpoint "
                f"[{method}]: stage={best_record['stage']}, epoch={best_record['epoch']}, "
                f"structural_MSE={best_record[key]:.6f}"
            )
            print(
                "Last logged structural checkpoint "
                f"[{method}]: stage={last_record['stage']}, epoch={last_record['epoch']}, "
                f"structural_MSE={last_record[key]:.6f}"
            )


def _fit_demand_design_model(params, train, evaluation_callback=None):
    model = CausalBGM_IV(params=params, random_seed=None)
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=int(params.get("fit_epochs", 100)),
        epochs_per_eval=int(params.get("fit_epochs_per_eval", 10)),
        batch_size=int(params.get("fit_batch_size", 32)),
        use_egm_init=True,
        egm_n_iter=int(params.get("fit_egm_n_iter", 10000)),
        egm_batches_per_eval=int(params.get("fit_egm_batches_per_eval", 500)),
        verbose=1,
        first_stage_warmup_epochs=int(params.get("fit_first_stage_warmup_epochs", 30)),
        evaluation_callback=evaluation_callback,
    )
    return model


def _evaluate_structural_methods(
    model,
    grid_x,
    grid_v,
    y_true,
    methods,
    y_stats=None,
):
    results = {}
    for method in methods:
        data_y_pred = model.predict_structural(
            grid_x,
            grid_v,
            latent_method=method,
        )
        if y_stats is not None:
            data_y_pred = _inverse_transform(data_y_pred, y_stats)
        mse = float(np.mean((y_true - data_y_pred) ** 2))
        results[method] = mse
        print(f"Structural MSE [{method}] = {mse:.6f}")
    return results


def run_demand_design_iv(params):
    """Run demand-design IV experiments aligned with the DFIV benchmark."""
    train = simulate_demand_design_iv(
        n_samples=int(params.get("n_samples", 5000)),
        rho=float(params.get("rho", 0.5)),
        seed=int(params.get("seed", 0)),
    )
    grid = make_demand_design_grid(
        price_points=int(params.get("price_points", 20)),
        time_points=int(params.get("time_points", 20)),
    )
    _summarize_ranges(train)

    methods = tuple(
        params.get(
            "structural_methods",
            [params.get("structural_latent_method", "map")],
        )
    )
    normalize_before_training = bool(params.get("normalize_before_training", False))

    if normalize_before_training:
        print("\nNormalized-space experiment")
        train_std, grid_std, stats = _standardize_demand_design_data(train, grid)
        structural_monitor_method = params.get("training_structural_monitor_method", "encoder")
        additional_monitor_methods = [
            method
            for method in methods
            if method != structural_monitor_method
        ]
        evaluation_callback = _make_structural_monitor_callback(
            grid_std["x"],
            grid_std["v"],
            grid["y_struct"],
            latent_method=structural_monitor_method,
            y_stats=stats["y"],
            additional_methods=additional_monitor_methods,
        )
        model = _fit_demand_design_model(
            params,
            train_std,
            evaluation_callback=evaluation_callback,
        )
        _print_training_history(getattr(model, "training_history", []))
        causal_pre, mse_x, mse_y, mse_v = model.evaluate(
            data=(train_std["x"], train_std["y"], train_std["v"], train_std["w"]),
            data_z=None,
            nb_intervals=int(params.get("nb_intervals", 20)),
        )
        print(
            "Training evaluate:",
            causal_pre.shape,
            f"MSE_x={float(mse_x):.4f}",
            f"MSE_y={float(mse_y):.4f}",
            f"MSE_v={float(mse_v):.4f}",
        )
        results = _evaluate_structural_methods(
            model,
            grid_std["x"],
            grid_std["v"],
            grid["y_struct"],
            methods=methods,
            y_stats=stats["y"],
        )
        print("\nStructural MSE summary (DFIV-compatible original outcome space)")
        for method in methods:
            print(f"  normalized/{method}: {results[method]:.6f}")
        return
#k l z e b u
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='the path to config file')
    parser.add_argument('-k', '--kl_weight', type=float, help='KL divergence weight')
    parser.add_argument('-l', '--learning', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-a', '--alpha', type=float, help='coefficient in EGM init', default=0.0)
    parser.add_argument('-z','--Z_dim', type=int,help="Latent dimension Z",default=10)
    parser.add_argument('-x','--X_dim', type=int,help="Latent dimension X",default=100)
    parser.add_argument('-e','--epochs', type=int, default=500, help="Epoches for iterative updating")
    parser.add_argument('-b','--batches', type=int, default=100000, help="Batches for initialization")
    parser.add_argument('-r','--rank', type=int, default=2, help="rank of low-rank approximation")
    parser.add_argument('-u','--units', type=int, nargs='+', default=[64,64,64,64,64],
                        help='Number of units for covariates generative model (default: [64,64,64,64,64]).')
    args = parser.parse_args()
    config = args.config
    kl_weight = args.kl_weight
    lr = args.learning
    alpha = args.alpha
    z_dim = args.Z_dim
    x_dim = args.X_dim
    units = args.units
    E=args.epochs
    B=args.batches
    rank = args.rank

    with open(config, 'r') as f:
        params = yaml.safe_load(f)

    if params['dataset'] == 'Sim_Hirano_Imbens':
        x,y,v = Sim_Hirano_Imbens_sampler(N=20000, v_dim=200).load_all()

        # Instantiate a CausalBGM model
        model = CausalBGM(params=params, random_seed=None)

        # Train the CausalBGM model with an iterative updating algorithm
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10, use_egm_init=True, egm_n_iter=30000, egm_batches_per_eval=500, verbose=1)

        # Make predictions using the trained CausalBGM model
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, x_values=np.linspace(0,3,20), q_sd=1.0)

    elif params['dataset'] == 'Semi_acic':
        x,y,v = Semi_acic_sampler(ufid='629e3d2c63914e45b227cc913c09cebe').load_all()

        # Instantiate a CausalBGM model
        model = CausalBGM(params=params, random_seed=None)

        # Train the CausalBGM model with an iterative updating algorithm
        model.fit(data=(x,y,v), epochs=100, epochs_per_eval=10, use_egm_init=True, egm_n_iter=30000, egm_batches_per_eval=500, verbose=1)
        
        # Make predictions using the trained CausalBGM model
        causal_pre, pos_intervals = model.predict(data=(x,y,v), alpha=0.01, n_mcmc=3000, q_sd=1.0)

    elif params["dataset"] == "Sim_Demand_Design_IV":
        run_demand_design_iv(params)

    elif params['dataset'] == 'Sim_heteroskedastic':        
        X,Y = simulate_z_hetero(n=20000, k=params['z_dim'], d=params['x_dim']-1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
        data_train = np.c_[X_train, Y_train].astype('float32')
        X_test = X_test.astype('float32')
        Y_test = Y_test.astype('float32')
        model = BGM(params=params, random_seed=None)
        model.fit(data=data_train, epochs=200, epochs_per_eval=10, use_egm_init=True, egm_n_iter=50000, egm_batches_per_eval=500, verbose=1)
        data_test = np.hstack([X_test, np.full((X_test.shape[0], 1), np.nan)])
        data_x_pred, pred_interval = model.predict(data=data_test, 
                                                    alpha=0.05, 
                                                    bs=100,
                                                    n_mcmc=5000, 
                                                    burn_in=5000, 
                                                    step_size=0.01, 
                                                    num_leapfrog_steps=10, 
                                                    seed=42)
        print(data_x_pred.shape, pred_interval.shape)
        #np.savez(f'data_pred_test1.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))

    elif params['dataset'] == 'Synthetic_MNAR':
        missing_rate = float(params.get('missing_rate', 0.1))

        data = simulate_mnar_factor_data(
            n_samples=params['n_samples'],
            x_dim=params['x_dim'],
            z_dim=params['z_dim'],
            missing_rate=0.1,
            seed=123
        )

        model = BGM_MNAR(params, random_seed=None)
        model.fit(data=data['x_obs'], mask=data['mask'], x_true=data['x_full'], verbose=1)
        mcmc_imputed, intervals = model.predict(
            data=data['x_obs'],
            mask=data['mask'],
            x_true=data['x_full'],
            alpha=0.05,
            n_mcmc=500, 
            burn_in=500, 
            step_size=0.1, 
            num_leapfrog_steps=5, 
            seed=42,
            verbose=1
        )

        if model.last_prediction_ is None:
            raise RuntimeError('BGM_MNAR.predict() did not populate prediction diagnostics.')
        map_imputed = model.last_prediction_['map_imputed']

        metrics = {
            'method_name': 'bgm_mnar',
            'missingness_rate': missing_rate,
            'map_rmse': rmse_on_missing_entries(data['x_full'], map_imputed, data['mask']),
            'mcmc_rmse': rmse_on_missing_entries(data['x_full'], mcmc_imputed, data['mask']),
            'save_dir': model.save_dir,
        }

        save_dir = Path(model.save_dir)
        # save_dir.mkdir(parents=True, exist_ok=True)
        # with open(save_dir / 'mnar_single_rate_metrics.json', 'w', encoding='utf-8') as handle:
        #     json.dump(metrics, handle, indent=2)
        # np.savez(save_dir / 'mnar_single_rate_intervals.npz', intervals=np.array(intervals, dtype=object))

        baseline = benchmark_mnar_imputers(
            x_true=data['x_full'],
            x_obs=data['x_obs'],
            mask=data['mask'],
            params=params,
            missing_rate=missing_rate,
            seed=123,
            results_path=str(save_dir / 'baseline_results.csv'),
        )[['method_name', 'missingness_rate', 'rmse']].copy()
        bgm_rows = pd.DataFrame([
            {
                'method_name': 'bgm_mnar_mcmc',
                'missingness_rate': missing_rate,
                'rmse': metrics['mcmc_rmse'],
            },
            {
                'method_name': 'bgm_mnar_map',
                'missingness_rate': missing_rate,
                'rmse': metrics['map_rmse'],
            },
        ])
        comparison = pd.concat([bgm_rows, baseline], ignore_index=True)
        comparison = comparison.sort_values('rmse', kind='stable').reset_index(drop=True)
        comparison['delta_vs_bgm_mnar_map'] = comparison['rmse'] - metrics['map_rmse']
        comparison.to_csv(save_dir / f'comparison_missing_rate_{missing_rate:.1f}.csv', index=False)
        print(f'\nBGM_MNAR comparison at missing rate {missing_rate:.1f}')
        print(comparison.to_string(index=False))

        print('\nBGM_MNAR metrics')
        print(json.dumps(metrics, indent=2))

    elif params['dataset'] == 'MNIST':
        params['dataset'] = 'MNIST_%s_%s_%d_%d_%d'%(kl_weight, alpha, z_dim, E, B)
        params['kl_weight'] = kl_weight
        params['lr_theta'] = lr
        params['lr_z'] = lr
        params['alpha'] = alpha
        params['g_units'] = units
        params['e_units'] = units
        params['z_dim'] = z_dim
        params['use_bnn'] = False

        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        model = MNISTBGM(params=params, random_seed=None)
        if False:
            print('Initializing EGM...')
            model.egm_init(data=x_train, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
            print('Fitting...')
            model.fit(data=x_train, epochs=E, epochs_per_eval=20, verbose=1)
        else:
            from bayesgm.utils import mnist_mask_indices
            #ind_x1, ind_x2 = mnist_mask_indices(mode="holes",center=(14,14),num_holes=1, hole_size=5, seed=1)
            ind_x1, ind_x2 = mnist_mask_indices(mode="edge_stripe", orientation="horizontal", stripe_pos=14, stripe_width=5)
            #ind_x1, ind_x2 = mnist_mask_indices(mode="upper_half")
            #idx = [i for i,item in enumerate(y_test) if item in [1,3,5,7,6,8]]
            #x_test_selected = x_test[idx]
            
            #epoch = 1000
            #checkpoint_path = 'checkpoints/MNIST_0.001_0.0_10_1000_50000/20251025_173803' 

            epoch=2000
            checkpoint_path = 'checkpoints/MNIST_1e-05_0.0_10_3000_50000/20251104_004358' 
            
            print(f"Epoch {epoch}")
            base_path = checkpoint_path + f"/weights_at_{epoch}"
            model.g_net.load_weights(f"{base_path}_generator.weights.h5")
            # n_test = 100
            # data_x_pred, pred_interval = model.predict(data=x_test[:n_test].reshape((n_test, -1))[:,ind_x1], 
            #                                             ind_x1=ind_x1, 
            #                                             alpha=0.05, 
            #                                             bs=10, 
            #                                             n_mcmc=5000, 
            #                                             burn_in=5000, 
            #                                             step_size=0.01, 
            #                                             num_leapfrog_steps=10, 
            #                                             seed=42)
            # print(data_x_pred.shape, pred_interval.shape)
            # np.savez('eval_results/data_pred_mnist_upper_half_e2000.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
            #n_test = len(x_test)
            #n_test = len(x_train)
            #data_test = np.load(f'mnist_train_imgs_holes_{E}.npy')
            #data_test = np.load(f'mnist_test_imgs_holes_1_c_{E}_{B}.npy')
            data_test = np.load(f'mnist_test_imgs_5_stripe.npy')
            n_test = len(data_test)
            #n_test = 300
            batch_size = 300
            data_x_mean_pred_list = []
            pred_interval_list = []
            #(5000, 100, 28, 28, 1) (100, 28, 28, 1, 2)
            # Process in mini-batches
            for i in range(0, n_test, batch_size):
                end_idx = min(i + batch_size, n_test)
                batch_data = data_test[i:end_idx]
                batch_data_x_pred, batch_pred_interval = model.predict(
                    data=batch_data, 
                    alpha=0.05, 
                    bs=10, 
                    n_mcmc=5000, 
                    burn_in=5000, 
                    step_size=0.01, 
                    num_leapfrog_steps=10, 
                    seed=42
                )
                data_x_mean_pred_list.append(np.mean(batch_data_x_pred, axis=0))
                pred_interval_list.append(batch_pred_interval)
                print(f"Processed batch {i//batch_size + 1}/{(n_test + batch_size - 1)//batch_size}")

            # Concatenate results
            data_x_pred = np.concatenate(data_x_mean_pred_list, axis=0)
            if isinstance(pred_interval_list[0], np.ndarray):
                pred_interval = np.concatenate(pred_interval_list, axis=0)
            else:
                pred_interval = []
                for batch_interval in pred_interval_list:
                    pred_interval.extend(batch_interval)
                
            #np.savez(f'data_pred_mnist_holes_1_test_c_{E}_{B}.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))
            #np.savez(f'data_pred_mnist_random_hole_train_{E}.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))
            np.savez(f'data_pred_mnist_5_stripe_test.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))

            #np.savez('data_pred_mnist_edge_stripe_e2k_test_all_5.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
            #np.savez('data_pred_mnist_edge_stripe_e2k_train_all_5.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
