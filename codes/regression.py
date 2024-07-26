#########################################################
# Least Squares Regression
# © 2024 Seho Jeong
#########################################################

# Import necessary packages
import numpy as np
from numpy.linalg import inv

# ------------------------------------ #
#     Ordinary Least Squares (OLS)     #
# ------------------------------------ #
class OLS:
    def __init__(self, Y, X, add_constant=True):
        self.Y = Y
        self.X = self.addX
        self.n, self.k = X.shape

    def add_constant(self):
        """
        Add a column of 1s
        """
        pass

    def coefficients(self):
        """
        Estimate β from the linear projection model Y = X'β + e
        """
        Y = self.Y
        X = self.X
        # Assume X is not ill-conditioned.
        return inv(X.T @ X) @ (X.T @ Y)

    def residuals(self):
        """
        Compute the residuals e_hat = Y - Xβ_ols
        """
        Y = self.Y
        X = self.X
        β_ols = self.coefficients()

        return Y - X @ β_ols

    def projection_matrix(self):
        """
        Return a projection matrix of given matrix form of regressors.
        """
        Y = self.Y
        X = self.X
        return X @ inv(X.T @ X) @ X.T

    def annihilator_matrix(self):
        """
        Return an annihilator matrix of given matrix form of regressors.
        """
        Y = self.Y
        X = self.X
        return np.eye(len(X)) - X @ inv(X.T @ X) @ X.T

    def r_squared(self, adjusted=False):
        """
        Compute the R-squared (coefficient of determination).
        """
        Y = self.Y
        X = self.X

        e_hat = self.residuals().reshape(-1, 1)
        Y_demeaned = Y - np.mean(Y)

        SSR = e_hat.T @ e_hat
        SST = Y.T @ Y

        return float(1 - SSR / SST)

    def leverage_values(self):
        """
        Return the leverage values of the model.
        """
        return np.diag(self.projection_matrix())

    def coefficients_loo(self):
        """
        Estimate the coefficients of leave-one-out regression
        """
        n = self.n
        X = self.X
        h = self.leverage_values()
        β_ols = self.coefficients()
        res = self.residuals()

        β_loo = []
        
        for i in range(n):
            hii = h[i]   # leverage value
            ei = res[i]  # residual of i-th observation
            Xi = X[i, :] # i-th observation

            prediction_error = ei / (1 - hii)
            loo_estimator = β_ols - inv(X.T @ X) @ Xi * prediction_error

            β_loo.append(loo_estimator)

        return np.array(β_loo)

    def influence(self):
        """
        Compute a simple diagnostic for influential observations.
        """
        h = self.leverage_values()
        res = self.residuals()

        prediction_errors = res / (1-h)

        return np.max(h * prediction_errors)

    def standard_errors(self, type='classic'):
        """
        Compute a vector of standard errors from the covariance estimate.
        """
        n = self.n
        k = self.k
        X = self.X
        Qinv = inv(X.T @ X)
        res = self.residuals()
        h = self.leverage_values()

        if type == 'classic':
            u = (1 / (n-k)) * (res.T @ res)
            V = Qinv * u
            
        elif type == 'HC0':
            u = X * res.reshape(-1, 1)
            V = Qinv @ (u.T @ u) @ Qinv
            
        elif type == 'HC1':
            u = X * res.reshape(-1, 1)
            V = (n / (n-k)) * (Qinv @ (u.T @ u) @ Qinv)

        elif type == 'HC2':
            u = X * (res / np.sqrt(1 - h)).reshape(-1, 1)
            V = Qinv @ (u.T @ u) @ Qinv

        elif type == 'HC3':
            u = X * (res / (1 - h)).reshape(-1, 1)
            V = Qinv @ (u.T @ u) @ Qinv

        else:
            print('Unsupported type of covariance estimate.')
            V = None

        return np.sqrt(np.diag(V))



# ------------------------------------- #
#    Generalized Least Squares (GLS)    #
# ------------------------------------- #
class GLS(OLS):
    def __init__(self):
        super().__init__(Y, X)