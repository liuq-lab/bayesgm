from bayesgm.models import CausalBGM, BGM
from bayesgm.utils import parse_file, parse_file_triplet, save_data
import argparse
import numpy as np
from bayesgm import __version__


def _add_common_args(parser):
    """Add arguments shared by both CausalBGM and BGM subcommands."""
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="Output directory")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input data file must be in csv or txt or npz format")
    parser.add_argument('-t', '--delimiter', type=str, default='\t',
                        help="Delimiter for txt or csv files (default: tab '\\t').")
    parser.add_argument('-d', '--dataset', type=str, default='Mydata',
                        help="Dataset name")
    parser.add_argument('-F', '--save_format', type=str, default='txt',
                        help="Saving format (default: txt)")
    parser.add_argument('-save_model', default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to save model.")
    parser.add_argument('-save_res', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether to save intermediate results.")
    parser.add_argument('--use_bnn', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether use Bayesian neural nets.")
    parser.add_argument('--use_egm_init', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether use EGM initialization.")
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for reproduction (default: 123).")


def _build_causalbgm_parser(subparsers):
    """Build the CausalBGM subcommand parser."""
    parser = subparsers.add_parser(
        'causalbgm',
        help='Run CausalBGM for causal inference in observational studies',
        description='CausalBGM: An AI-powered Bayesian generative modeling approach for causal inference in observational studies'
    )
    _add_common_args(parser)

    # CausalBGM-specific arguments
    parser.add_argument('-B', '--binary_treatment', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether use binary treatment setting.")

    # Parameters for iterative updating algorithm
    parser.add_argument('-Z', '--z_dims', type=int, nargs='+', default=[3, 3, 6, 6],
                        help='Latent dimensions of Z (default: [3, 3, 6, 6]).')
    parser.add_argument('--lr_theta', type=float, default=0.0001,
                        help="Learning rate for updating model parameters (default: 0.0001).")
    parser.add_argument('--lr_z', type=float, default=0.0001,
                        help="Learning rate for updating latent variables (default: 0.0001).")
    parser.add_argument('--x_min', type=float, default=0.,
                        help="Lower bound for treatment interval (default: 0.0).")
    parser.add_argument('--x_max', type=float, default=3.,
                        help="Upper bound for treatment interval (default: 3.0).")
    parser.add_argument('--x_values', type=float, nargs='+',
                        help="List of treatment values to be predicted. Provide space-separated values. Example: --x_values 0.5 1.0 1.5")
    parser.add_argument('--g_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for covariates generative model (default: [64,64,64,64,64]).')
    parser.add_argument('--f_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for outcome generative model (default: [64,32,8]).')
    parser.add_argument('--h_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for treatment generative model (default: [64,32,8]).')

    # Parameters for EGM initialization
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help="Coefficient for KL divergence term in BNNs (default: 0.0001).")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate for EGM initialization (default: 0.0001).")
    parser.add_argument('--g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    parser.add_argument('--e_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('--dz_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for discriminator network in latent space (default: [64,32,8]).')
    parser.add_argument('--use-z-rec', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the reconstruction for latent features (default: True).")

    # Parameters for fitting && predicting
    parser.add_argument('-N', '--n_iter', type=int, default=30000,
                        help="Number of iterations (default: 30000).")
    parser.add_argument('--startoff', type=int, default=0,
                        help="Iteration for starting evaluation (default: 0).")
    parser.add_argument('--batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation (default: 500).")
    parser.add_argument('-E', '--epochs', type=int, default=100,
                        help="Number of epochs in iterative updating algorithm (default: 100).")
    parser.add_argument('-M', '--n_mcmc', type=int, default=3000,
                        help="MCMC sample size (default: 3000).")
    parser.add_argument('-q', '--q_sd', type=float, default=1.,
                        help="Standard deviation for proposal distribution in MCMC, a negative q_sd denotes adaptive MCMC (default: 1.0).")
    parser.add_argument('--epochs_per_eval', type=int, default=10,
                        help="Number of epochs per evaluation (default: 10).")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Significance level (default: 0.01).")

    parser.set_defaults(func=_run_causalbgm)
    return parser


def _build_bgm_parser(subparsers):
    """Build the BGM subcommand parser."""
    parser = subparsers.add_parser(
        'bgm',
        help='Run BGM for Bayesian generative modeling (data generation and imputation)',
        description='BGM: A Bayesian generative modeling approach for data generation and missing data imputation'
    )
    _add_common_args(parser)

    # BGM model architecture parameters
    parser.add_argument('--z_dim', type=int, default=10,
                        help='Latent dimension of Z (default: 10).')
    parser.add_argument('--g_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for generative model (default: [64,64,64,64,64]).')
    parser.add_argument('--e_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('--dz_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for latent discriminator (default: [64,32,8]).')
    parser.add_argument('--dx_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for data discriminator (default: [64,32,8]).')

    # Training parameters
    parser.add_argument('--lr_theta', type=float, default=0.0001,
                        help="Learning rate for updating model parameters (default: 0.0001).")
    parser.add_argument('--lr_z', type=float, default=0.0001,
                        help="Learning rate for updating latent variables (default: 0.0001).")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate for EGM initialization (default: 0.0001).")
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help="Coefficient for KL divergence term in BNNs (default: 0.0001).")
    parser.add_argument('--g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    parser.add_argument('--gamma', type=float, default=10.0,
                        help="Gradient penalty coefficient for EGM discriminator training (default: 10.0).")
    parser.add_argument('--egm_reg_alpha', type=float, default=0.01,
                        help="Regularization coefficient for variance in EGM generator training (default: 0.01).")

    # Fitting parameters
    parser.add_argument('-N', '--egm_n_iter', type=int, default=20000,
                        help="Number of iterations for EGM initialization (default: 20000).")
    parser.add_argument('--egm_batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation in EGM initialization (default: 500).")
    parser.add_argument('-E', '--epochs', type=int, default=100,
                        help="Number of epochs in iterative updating algorithm (default: 100).")
    parser.add_argument('--epochs_per_eval', type=int, default=5,
                        help="Number of epochs per evaluation (default: 5).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training (default: 32).")

    # Prediction parameters
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="Significance level for prediction intervals (default: 0.05).")
    parser.add_argument('-M', '--n_mcmc', type=int, default=5000,
                        help="Number of retained MCMC samples (default: 5000).")
    parser.add_argument('--burn_in', type=int, default=5000,
                        help="Number of burn-in iterations for MCMC (default: 5000).")
    parser.add_argument('--step_size', type=float, default=0.01,
                        help="HMC step size (default: 0.01).")
    parser.add_argument('--num_leapfrog_steps', type=int, default=10,
                        help="Number of leapfrog steps in HMC (default: 10).")

    parser.set_defaults(func=_run_bgm)
    return parser


def _run_causalbgm(args):
    """Execute the CausalBGM workflow."""
    params = vars(args)
    # Remove the 'func' key used for dispatch
    params.pop('func', None)
    data = parse_file_triplet(args.input, sep=params['delimiter'])
    params['v_dim'] = data[-1].shape[1]

    # Instantiate a CausalBGM model
    model = CausalBGM(params=params, random_seed=None)

    # Train the CausalBGM model (fit includes EGM init + iterative updating)
    model.fit(
        data=data,
        epochs=params['epochs'],
        epochs_per_eval=params['epochs_per_eval'],
        startoff=params['startoff'],
        use_egm_init=params['use_egm_init'],
        egm_n_iter=params['n_iter'],
        egm_batches_per_eval=params['batches_per_eval'],
        verbose=1
    )

    # Make predictions using the trained CausalBGM model
    if params['binary_treatment']:
        causal_pre, pos_intervals = model.predict(data=data, alpha=params['alpha'], n_mcmc=params['n_mcmc'], q_sd=params['q_sd'])
    else:
        causal_pre, pos_intervals = model.predict(data=data, alpha=params['alpha'], n_mcmc=params['n_mcmc'], x_values=params['x_values'], q_sd=params['q_sd'])

    # Save results
    save_data('{}/causal_effect_point_estimate.{}'.format(model.save_dir, params['save_format']), causal_pre)
    save_data('{}/causal_effect_posterior_interval.{}'.format(model.save_dir, params['save_format']), pos_intervals)


def _run_bgm(args):
    """Execute the BGM workflow."""
    params = vars(args)
    # Remove the 'func' key used for dispatch
    params.pop('func', None)
    data = parse_file(args.input, sep=params['delimiter'])

    # Set x_dim from data
    params['x_dim'] = data.shape[1]

    # Map egm_reg_alpha to the 'alpha' key expected by the BGM model for regularization
    # (distinct from the significance level 'alpha' used in predict)
    predict_alpha = params.pop('alpha')
    params['alpha'] = params.pop('egm_reg_alpha')

    # Instantiate a BGM model
    model = BGM(params=params, random_seed=params.get('seed'))

    # Train the BGM model (fit includes EGM init + iterative updating)
    model.fit(
        data=data,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        epochs_per_eval=params['epochs_per_eval'],
        use_egm_init=params['use_egm_init'],
        egm_n_iter=params['egm_n_iter'],
        egm_batches_per_eval=params['egm_batches_per_eval'],
        verbose=1
    )

    # Make predictions (imputation with uncertainty quantification)
    data_imputed, pred_interval = model.predict(
        data=data,
        alpha=predict_alpha,
        n_mcmc=params['n_mcmc'],
        burn_in=params['burn_in'],
        step_size=params['step_size'],
        num_leapfrog_steps=params['num_leapfrog_steps'],
        seed=params.get('seed', 42)
    )

    # Save results
    save_data('{}/imputed_data.{}'.format(model.save_dir, params['save_format']), data_imputed)
    np.savez('{}/prediction_intervals.npz'.format(model.save_dir), intervals=pred_interval)


def main(args=None):
    """Main entry point for the bayesgm CLI with subcommands."""
    parser = argparse.ArgumentParser(
        'bayesgm',
        description=f'BayesGM: An AI-powered Bayesian generative modeling framework - v{__version__}'
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(
        title='commands',
        description='Available model commands',
        dest='command'
    )

    _build_causalbgm_parser(subparsers)
    _build_bgm_parser(subparsers)

    args = parser.parse_args(args)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


def main_causalbgm(args=None):
    """Standalone entry point for the causalBGM command (backwards compatible)."""
    # Build a standalone parser that mimics the old flat-argument interface
    parser = argparse.ArgumentParser(
        'causalBGM',
        description=f'CausalBGM: An AI-powered Bayesian generative modeling approach for causal inference - v{__version__}'
    )

    _add_common_args(parser)

    # CausalBGM-specific arguments (same as the subcommand)
    parser.add_argument('-B', '--binary_treatment', default=True, action=argparse.BooleanOptionalAction,
                        help="Whether use binary treatment setting.")
    parser.add_argument('-Z', '--z_dims', type=int, nargs='+', default=[3, 3, 6, 6],
                        help='Latent dimensions of Z (default: [3, 3, 6, 6]).')
    parser.add_argument('--lr_theta', type=float, default=0.0001,
                        help="Learning rate for updating model parameters (default: 0.0001).")
    parser.add_argument('--lr_z', type=float, default=0.0001,
                        help="Learning rate for updating latent variables (default: 0.0001).")
    parser.add_argument('--x_min', type=float, default=0.,
                        help="Lower bound for treatment interval (default: 0.0).")
    parser.add_argument('--x_max', type=float, default=3.,
                        help="Upper bound for treatment interval (default: 3.0).")
    parser.add_argument('--x_values', type=float, nargs='+',
                        help="List of treatment values to be predicted.")
    parser.add_argument('--g_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for covariates generative model (default: [64,64,64,64,64]).')
    parser.add_argument('--f_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for outcome generative model (default: [64,32,8]).')
    parser.add_argument('--h_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for treatment generative model (default: [64,32,8]).')
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help="Coefficient for KL divergence term in BNNs (default: 0.0001).")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate for EGM initialization (default: 0.0001).")
    parser.add_argument('--g_d_freq', type=int, default=5,
                        help="Frequency for updating discriminators and generators (default: 5).")
    parser.add_argument('--e_units', type=int, nargs='+', default=[64, 64, 64, 64, 64],
                        help='Number of units for encoder network (default: [64,64,64,64,64]).')
    parser.add_argument('--dz_units', type=int, nargs='+', default=[64, 32, 8],
                        help='Number of units for discriminator network in latent space (default: [64,32,8]).')
    parser.add_argument('--use-z-rec', default=True, action=argparse.BooleanOptionalAction,
                        help="Use the reconstruction for latent features (default: True).")
    parser.add_argument('-N', '--n_iter', type=int, default=30000,
                        help="Number of iterations (default: 30000).")
    parser.add_argument('--startoff', type=int, default=0,
                        help="Iteration for starting evaluation (default: 0).")
    parser.add_argument('--batches_per_eval', type=int, default=500,
                        help="Number of iterations per evaluation (default: 500).")
    parser.add_argument('-E', '--epochs', type=int, default=100,
                        help="Number of epochs in iterative updating algorithm (default: 100).")
    parser.add_argument('-M', '--n_mcmc', type=int, default=3000,
                        help="MCMC sample size (default: 3000).")
    parser.add_argument('-q', '--q_sd', type=float, default=1.,
                        help="Standard deviation for proposal distribution in MCMC (default: 1.0).")
    parser.add_argument('--epochs_per_eval', type=int, default=10,
                        help="Number of epochs per evaluation (default: 10).")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Significance level (default: 0.01).")

    args = parser.parse_args()
    # Attach the func so _run_causalbgm can pop it
    args.func = _run_causalbgm
    _run_causalbgm(args)


if __name__ == "__main__":
    main()
