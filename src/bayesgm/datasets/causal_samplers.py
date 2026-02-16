from .base_sampler import Base_sampler
import numpy as np
import math
from scipy.sparse import diags
from scipy.stats import norm
import pandas as pd



class Semi_acic_sampler(Base_sampler):
    """ACIC 2018 competition dataset (binary treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    path
        Str object denoting the path to the original dataset.
    ufid
        Str object denoting the unique id of a specific semi-synthetic setting.
    Examples
    --------
    >>> from CausalEGM import Semi_acic_sampler
    >>> import numpy as np
    >>> x = np.random.normal(size=(2000,))
    >>> y = np.random.normal(size=(2000,))
    >>> v = np.random.normal(size=(2000,100))
    >>> ds = Semi_acic_sampler(path='../data/ACIC_2018',ufid='d5bd8e4814904c58a79d7cdcd7c2a1bb')
    """
    def __init__(self, batch_size=32, path='../data/ACIC_2018', 
                ufid='d5bd8e4814904c58a79d7cdcd7c2a1bb'):
        self.df_covariants = pd.read_csv('%s/x.csv'%path, index_col='sample_id',header=0, sep=',')
        self.df_sim = pd.read_csv('%s/scaling/factuals/%s.csv'%(path, ufid),index_col='sample_id',header=0, sep=',')
        dataset = self.df_covariants.join(self.df_sim, how='inner')
        x = dataset['z'].values.reshape(-1,1)
        y = dataset['y'].values.reshape(-1,1)
        v = dataset.values[:,:-2]
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Hirano_Imbens_sampler(Base_sampler):
    """Hirano Imbens simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Hirano_Imbens_sampler
    >>> ds = Sim_Hirano_Imbens_sampler(batch_size=32, N=20000, v_dim=200, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=200, seed=0):
        np.random.seed(seed)
        v = np.random.exponential(scale=1.0, size=(N, v_dim))
        rate = v[:,0] + v[:,1]
        scale = 1/rate
        x = np.random.exponential(scale=scale)
        y = np.random.normal(x + (v[:,0] + v[:,2]) * np.exp(-x * (v[:,0] + v[:,2])) , 1)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Sun_sampler(Base_sampler):
    """Sun simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Sun_sampler
    >>> ds = Sim_Sun_sampler(batch_size=32, N=20000, v_dim=200, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=200, seed=0):
        np.random.seed(seed)
        v = np.random.normal(0, 1, size=(N, v_dim))        
        x = np.random.normal(-2*(np.sin(2*v[:,0]))+ ((v[:,1])**2 - 1/3) + (v[:,2]-1/2) + np.cos(v[:,3]), 1)
        y = np.random.normal(((v[:,0] - 1/2)+ np.cos(v[:,1]) + (v[:,4])**2 + (v[:,5])) + x, 1)       
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Sim_Colangelo_sampler(Base_sampler):
    """Colangelo simulation dataset (continuous treatment) sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    N
        Sample size. Default: ``20000``.
    v_dim
        Int object denoting the dimension for covariates. Default: ``200``.
    seed
        Int object denoting the random seed. Default: ``0``.
    Examples
    --------
    >>> from CausalEGM import Sim_Colangelo_sampler
    >>> ds = Sim_Colangelo_sampler(batch_size=32, N=20000, v_dim=100, seed=0)
    """
    def __init__(self, batch_size=32, N=20000, v_dim=100, seed=0,
                rho=0.5, offset = [-1,0,1], d=1, a=3, b=0.75):
        np.random.seed(seed)
        k = np.array([rho*np.ones(v_dim-1),np.ones(v_dim),rho*np.ones(v_dim-1)],dtype=object)
        sigma = diags(k,offset).toarray()
        theta = np.array([(1/(l**2)) for l in list(range(1,(v_dim+1)))])
        epsilon = np.random.normal(0,1,N)
        nu = np.random.normal(0,1,N)
        v = np.random.multivariate_normal(np.zeros(v_dim),sigma,size=[N,])
        x = d*norm.cdf((a*v@theta)) + b*nu - 0.5
        y = 1.2*x + (x**3) + (x*v[:,0]) + 1.2*(v@theta) + epsilon
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

class Semi_Twins_sampler(Base_sampler):
    """Twins semi synthetic  dataset sampler (inherited from Base_sampler).

    Parameters
    ----------
    batch_size
        Int object denoting the batch size for mini-batch training. Default: ``32``.
    seed
        Int object denoting the random seed. Default: ``0``.
    path
        Str obejct denoting the path to the original data.
    Examples
    --------
    >>> from CausalEGM import Semi_Twins_sampler
    >>> ds = Semi_Twins_sampler(batch_size=32, path='../data/Twins')
    """
    def __init__(self, batch_size=32, seed=0,
                path='../data/Twins'):
        covariate_df = pd.read_csv('%s/twin_pairs_X_3years_samesex.csv'%path).iloc[:,2:].drop(['infant_id_0', 'infant_id_1'], axis=1)
        treatment_df_ = pd.read_csv('%s/twin_pairs_T_3years_samesex.csv'%path).iloc[:,1:]
        outcome_df = pd.read_csv('%s/twin_pairs_Y_3years_samesex.csv'%path).iloc[:,1:]
        #### discard NAN values
        rows_with_nan = [index for index, row in covariate_df.iterrows() if row.isnull().any()]
        covariate_df = covariate_df.drop(rows_with_nan)
        treatment_df_ = treatment_df_.drop(rows_with_nan)
        outcome_df = outcome_df.drop(rows_with_nan)
        #### select those below 2kg:
        rows_less2kg = [index for index, row in treatment_df_.iterrows() if (row['dbirwt_1']>=2000)]
        covariate_df = covariate_df.drop(rows_less2kg)
        treatment_df_ = treatment_df_.drop(rows_less2kg)
        outcome_df = outcome_df.drop(rows_less2kg)

        x = np.concatenate([treatment_df_.values[:,0], treatment_df_.values[:,1]])/1000
        v =  np.concatenate([covariate_df.values, covariate_df.values])
        np.random.seed(seed)
        eps = np.random.normal(0, 0.25, size=(v.shape[0],))
        gamma = np.random.normal(0, 0.025, size=(v.shape[1],))
        y = -2 * 1/(1 + np.exp(-3 * x)) + np.dot(v, gamma) + eps
        self.auxiliary_constant =  np.mean(np.dot(v, gamma))
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        super().__init__(x,y,v,batch_size=batch_size,normalize=True)

