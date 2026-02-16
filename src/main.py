import yaml
import argparse
import numpy as np
import sys
from bayesgm.models import CausalBGM, BGM, MNISTBGM
from bayesgm.datasets import (
    simulate_z_hetero,
    Sim_Hirano_Imbens_sampler, 
    Semi_acic_sampler
)
from sklearn.model_selection import train_test_split
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
#tf.config.set_visible_devices([], 'GPU')  # 显式禁止使用 GPU

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
        np.savez(f'data_pred_test1.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))
        sys.exit()

        
        if True:
            model.egm_init(data=data, n_iter=B, batch_size=32, batches_per_eval=5000, verbose=1)
            model.fit(data=data, epochs=E, epochs_per_eval=20, verbose=1)
            data = np.hstack([X_test, np.full((X_test.shape[0], 1), np.nan)])
            data_x_pred, pred_interval = model.predict(data=data, 
                                                        alpha=0.05, 
                                                        bs=100,
                                                        n_mcmc=5000, 
                                                        burn_in=5000, 
                                                        step_size=0.01, 
                                                        num_leapfrog_steps=10, 
                                                        seed=42)

            X_test_pred = data_x_pred[:,:,-1]
            X_test_pred_mean = np.mean(X_test_pred, axis=0)
            X_test_pred_median = np.median(X_test_pred, axis=0)
            print(X_test_pred.shape, X_test_pred_mean.shape, X_test_pred_median.shape)
            from scipy.stats import pearsonr, spearmanr
            corr_pred_mean, _ = pearsonr(Y_test, X_test_pred_mean)
            print(f"Pearson's correlation coefficient mean: {corr_pred_mean}")
            corr_pred_median, _ = pearsonr(Y_test, X_test_pred_median)
            print(f"Pearson's correlation coefficient median: {corr_pred_median}")
            np.savez(f'data_pred_heter_{z_dim}_{x_dim}_{E}_{B}_{params["use_bnn"]}_{units}.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
            # length_gt = np.load('heter_length_gt_10_100.npy')
            # length_bgm = pred_interval[:,0,1]-pred_interval[:,0,0]
            # print('Average interval length:', np.mean(length_bgm), np.mean(length_gt))
            # print('interval length PCC:', pearsonr(length_bgm, length_gt)[0])
            # print('interval length Spearman:', spearmanr(length_bgm, length_gt)[0])
        else:
            checkpoint_path = 'checkpoints/Sim_heteroskedastic_5e-05_0.0_10_200_50000/20251030_214921'
            #for epoch in range(100,500+20,20):
            for epoch in [200]:
                print(f"Epoch {epoch}")
                base_path = checkpoint_path + f"/weights_at_{epoch}"
                model.g_net.load_weights(f"{base_path}_generator.weights.h5")
                data = np.hstack([X_test, np.full((X_test.shape[0], 1), np.nan)])
                data_x_pred, pred_interval = model.predict(data=data, 
                                                            alpha=0.05, 
                                                            bs=500, 
                                                            n_mcmc=5000, 
                                                            burn_in=5000, 
                                                            step_size=0.01, 
                                                            num_leapfrog_steps=10, 
                                                            seed=42)
                print(data_x_pred.shape, pred_interval.shape)
                np.savez(f'data_pred_test.npz', data_x_pred=data_x_pred, pred_interval=np.array(pred_interval, dtype=object))
                sys.exit()
                X_test_pred = data_x_pred[:,:,-1]
                X_test_pred_mean = np.mean(X_test_pred, axis=0)
                X_test_pred_median = np.median(X_test_pred, axis=0)
                print(X_test_pred.shape, X_test_pred_mean.shape, X_test_pred_median.shape)
                from scipy.stats import pearsonr, spearmanr
                corr_pred_mean, _ = pearsonr(Y_test, X_test_pred_mean)
                print(f"Pearson's correlation coefficient mean: {corr_pred_mean}")
                corr_pred_median, _ = pearsonr(Y_test, X_test_pred_median)
                print(f"Pearson's correlation coefficient median: {corr_pred_median}")
                np.savez('data_pred_heter_10_100.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)
                length_gt = np.load('heter_length_gt_10_100.npy')
                length_bgm = pred_interval[:,0,1]-pred_interval[:,0,0]
                print('Average interval length:', np.mean(length_bgm), np.mean(length_gt))
                print('interval length PCC:', pearsonr(length_bgm, length_gt)[0])
                print('interval length Spearman:', spearmanr(length_bgm, length_gt)[0])
                #np.savez('data_pred_heter_10_100.npz', data_x_pred=data_x_pred, pred_interval=pred_interval)

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
