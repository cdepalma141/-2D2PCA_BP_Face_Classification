# This class is used to perform all calculations relating to 2D2PCA
# The algorithms follows the research from https://dl.acm.org/doi/10.1016/j.neucom.2005.06.004
import numpy as np

class PCA2D():

    def __init__(self, images, d=0, q=0, quality=0):
        # A structure of images must be passed in
        # d and q are used if the user wants to specify a number of row-wise 
        # and column-wise principle components, respectively
        # Otherwise, a quality value can be used to get quality*100 % representation of the original images
        
        self.images = images
        self.d = d
        self.q = q
        self.quality = quality
        self.A = np.mean(self.images,0) # Mean across all images
        self.G_e_vals = 0
        self.H_e_vals = 0

    def give_dq(self, eig_vals):
        # Uses quality parameter to generate d and q values
        s = np.sum(eig_vals)
        thresh = s * self.quality
        t = 0
        d = 0
        while t < thresh:
            t += eig_vals[d]
            d += 1

        return d
    
    def explained_variance_ratio(self):
        # Used to generate data for an explained variance ratio plot
        total_G_vals = np.sum(self.G_e_vals)
        total_H_vals = np.sum(self.H_e_vals)
       
        G_ratio = self.G_e_vals/total_G_vals 
        H_ratio = self.H_e_vals/total_H_vals

        G_ratio = np.cumsum(G_ratio)
        H_ratio = np.cumsum(H_ratio)

        return (G_ratio,np.arange(len(self.G_e_vals))),(H_ratio,np.arange(len(self.H_e_vals)))

    def reduce_dim_row(self):
        # Perform traditional 2DPCA among rows of image

        M = self.images.shape[0] # Total number of images
        height = self.images.shape[1]
        
        G = np.zeros((height, height)) # initialize row-wise covariance matrix
        
        for i in range(M):
            # Generation of row-wise covariance matrix per research linked at beginning of file
            G += (self.images[i]-self.A).T @ (self.images[i]-self.A)
        G /= M

        # Eigendecomposition is used to find vectors used to generate entire PC space
        G_e_vals, G_e_vec = np.linalg.eig(G)
        
        self.G_e_vals = G_e_vals

        # If quality is given, find corresponding d values
        if self.quality:
            self.d = self.give_dq(G_e_vals)

        # Create projection matrix using eigenvectors corresponding to the d largest eigenvalues
        # of the decomposed covariance matrix
        X = G_e_vec[:,0:self.d]

        # Generate principle component space
        Y = self.images @ X

        return Y, X, G

    def reduce_dim_col(self):
        # Perform alternative 2DPCA among columns of image
       
        M = self.images.shape[0] # Total number of images
        width = self.images.shape[2]
    
        H = np.zeros((width, width)) # initialize column-wise covariance matrix
        
        for i in range(M):
            # Generation of column-wise covariance matrix per research linked at beginning of file
            H += (self.images[i]-self.A) @ (self.images[i]-self.A).T
        H /= M

        # Eigendecomposition is used to find vectors used to generate PC space
        H_e_vals, H_e_vec = np.linalg.eig(H)
        
        self.H_e_vals = H_e_vals 

        # If quality is given, find corresponding q values
        if self.quality:
            self.q = self.give_dq(H_e_vals)

        # Create projection matrix using eigenvectors corresponding to the q largest eigenvalues
        # of the decomposed covariance matrix
        Z = H_e_vec[:,0:self.q]

        # Generate principle component space
        B = Z.T @ self.images

        return B, Z, H

    def reduce_dim_square(self):
        # 2-Directional 2D PCA is done by doing row-wise and column-wise 2DPCA 

        # Only need projection matrices to get PC space
        _,X,_ = self.reduce_dim_row()
        _,Z,_ = self.reduce_dim_col()

        C = Z.T @ self.images @ X

        return C, X, Z
