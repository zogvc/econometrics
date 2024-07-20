#########################################################
# Econometric Methods
# © 2024 Seho Jeong
#########################################################

# Ordinary Least Squares (OLS)
class OLS:
    def __init__(self, Y, X):
        self.Y = Y
        self.X = X

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
        Compute the residuals e_hat = Y - Xβ_hat
        """
        Y = self.Y
        X = self.X
        β_hat = self.coefficients()

        return Y - X @ β_hat

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

    def r_squared(self):
        """
        Compute the (unadjusted) R-squared (coefficient of determination).
        """
        Y = self.Y
        X = self.X
        
        e_hat = ols_residuals(Y, X).reshape(-1, 1)
        Y_demeaned = Y - np.mean(Y)

        num = e_hat.T @ e_hat
        den = Y.T @ Y

        return num / den