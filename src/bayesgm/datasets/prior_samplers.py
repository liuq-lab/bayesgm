import numpy as np


class Gaussian_sampler(object):
    """Multivariate Gaussian sampler.

    Generates samples from :math:`\\mathcal{N}(\\mu, \\sigma^2 I)` and stores a
    pre-sampled dataset of size ``N`` for batch training.

    Parameters
    ----------
    mean : array-like
        Mean vector of length ``d``.
    sd : float, default=1
        Scalar standard deviation applied to every dimension.
    N : int, default=20000
        Size of the pre-sampled dataset.
    """

    def __init__(self, mean, sd=1, N=20000):
        self.total_size = N
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
        self.X = np.random.normal(self.mean, self.sd, (self.total_size,len(self.mean)))
        self.X = self.X.astype('float32')

    def train(self, batch_size, label = False):
        """Return a random batch from the pre-sampled dataset.

        Parameters
        ----------
        batch_size : int
            Number of samples to return.
        label : bool, default=False
            Unused.  Kept for API compatibility.

        Returns
        -------
        np.ndarray
            Batch with shape ``(batch_size, d)``.
        """
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_batch(self,batch_size):
        """Draw fresh samples from the Gaussian distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Samples with shape ``(batch_size, d)``, dtype float32.
        """
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean))).astype('float32')

    def load_all(self):
        """Return the full pre-sampled dataset.

        Returns
        -------
        np.ndarray
            Dataset with shape ``(N, d)``.
        """
        return self.X

class GMM_indep_sampler(object):
    """Independent Gaussian Mixture Model (GMM) sampler.

    Each dimension is sampled independently from a 1-D Gaussian mixture with
    ``n_components`` equally-spaced centres.

    Parameters
    ----------
    N : int
        Total number of pre-sampled data points.
    sd : float
        Standard deviation of each mixture component.
    dim : int
        Dimensionality of the data.
    n_components : int
        Number of mixture components per dimension.
    weights : array-like or None, optional
        Component weights (uniform if ``None``).
    bound : float, default=1
        Mixture centres are placed uniformly in ``[-bound, bound]``.
    """

    def __init__(self, N, sd, dim, n_components, weights=None, bound=1):
        np.random.seed(1024)
        self.total_size = N
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.bound = bound
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.nb_train = self.X_train.shape[0]
        self.Y=None
    def generate_gmm(self,weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.total_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
    
    def get_density(self, data):
        """Evaluate the exact GMM density at given points.

        Parameters
        ----------
        data : np.ndarray
            Query points with shape ``(m, dim)``.

        Returns
        -------
        np.ndarray
            Density values with shape ``(m,)``.
        """
        assert data.shape[1]==self.dim
        from scipy.stats import norm
        centers = np.linspace(-self.bound, self.bound, self.n_components)
        prob = []
        for i in range(self.dim):
            p_mat = np.zeros((self.n_components,len(data)))
            for j in range(len(data)):
                for k in range(self.n_components):
                    p_mat[k,j] = norm.pdf(data[j,i], loc=centers[k], scale=self.sd) 
            prob.append(np.mean(p_mat,axis=0))
        prob = np.stack(prob)        
        return np.prod(prob, axis=0)

    def train(self, batch_size):
        """Return a random batch from the training split.

        Parameters
        ----------
        batch_size : int
            Number of samples to return.

        Returns
        -------
        np.ndarray
            Batch with shape ``(batch_size, dim)``.
        """
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        return self.X_train[indx, :]

    def load_all(self):
        """Return the full pre-sampled dataset.

        Returns
        -------
        X : np.ndarray
            Dataset with shape ``(N, dim)``.
        Y : None
            Placeholder (always ``None``).
        """
        return self.X, self.Y

#Swiss roll (r*sin(scale*r),r*cos(scale*r)) + Gaussian noise
class Swiss_roll_sampler(object):
    """Swiss-roll distribution sampler.

    Generates 2-D data along the curve
    :math:`(r \\sin(s \\cdot r),\\, r \\cos(s \\cdot r))` plus isotropic
    Gaussian noise.

    Parameters
    ----------
    N : int
        Number of pre-sampled data points.
    theta : float, default=2*pi
        Maximum parameter value along the spiral.
    scale : float, default=2
        Frequency scaling of the spiral.
    sigma : float, default=0.4
        Standard deviation of the additive Gaussian noise.
    """

    def __init__(self, N, theta=2*np.pi, scale=2, sigma=0.4):
        np.random.seed(1024)
        self.total_size = N
        self.theta = theta
        self.scale = scale
        self.sigma = sigma
        params = np.linspace(0,self.theta,self.total_size)
        self.X_center = np.vstack((params*np.sin(scale*params),params*np.cos(scale*params)))
        self.X = self.X_center.T + np.random.normal(0,sigma,size=(self.total_size,2))
        np.random.shuffle(self.X)
        self.X_train, self.X_val,self.X_test = self.split(self.X)
        self.Y = None
        self.mean = 0
        self.sd = 0

    def split(self,data):
        N_test = int(0.1*data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1*data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]
        data = np.vstack((data_train, data_validate))
        return data_train, data_validate, data_test
        
    def train(self, batch_size, label = False):
        """Return a random batch from the pre-sampled dataset.

        Parameters
        ----------
        batch_size : int
            Number of samples to return.
        label : bool, default=False
            Unused.  Kept for API compatibility.

        Returns
        -------
        np.ndarray
            Batch with shape ``(batch_size, 2)``.
        """
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)
        return self.X[indx, :]

    def get_density(self,x_points):
        """Evaluate the (approximate) density via kernel density on the noiseless curve.

        Parameters
        ----------
        x_points : np.ndarray
            Query points with shape ``(m, 2)``.

        Returns
        -------
        np.ndarray
            Density values with shape ``(m,)``.
        """
        assert len(x_points.shape)==2
        c = 1./(2*np.pi*self.sigma)
        px = [c*np.mean(np.exp(-np.sum((np.tile(x,[self.total_size,1])-self.X_center.T)**2,axis=1)/(2*self.sigma))) for x in x_points]
        return np.array(px)

    def load_all(self):
        """Return the full pre-sampled dataset.

        Returns
        -------
        X : np.ndarray
            Dataset with shape ``(N, 2)``.
        Y : None
            Placeholder (always ``None``).
        """
        return self.X, self.Y
