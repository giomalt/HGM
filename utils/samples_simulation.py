from math import *
from scipy.special.orthogonal import hermite
from scipy.special import eval_hermite

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def matrix_complex_to_real_imag(matrix):
    matrix_flat = matrix.reshape(matrix.shape[0]*matrix.shape[1])
    matrix_ri = np.zeros(matrix_flat.shape[0]*2)
    matrix_ri[0::2] = np.real(matrix_flat)
    matrix_ri[1::2] = np.imag(matrix_flat)
    
    return matrix_ri

def interp_near_pixels(image, interp_factor=4):
    """ Increse size of an image by taking neighbour pixel value.
    Used to increase size of (X,Y), so that PSF and HG modes are smoother."""
    img_interp = np.zeros((image.shape[0]*interp_factor,image.shape[1]*interp_factor))
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            img_interp[i*interp_factor:(i+1)*interp_factor,j*interp_factor:(j+1)*interp_factor] = image[i,j]
    return img_interp
    
def source_to_sim_size(source, x, up_scale): # up_scale depends on source image
    """ Interpolate and pad a source image to make it compatible with simulation (X,Y) sizes. """
    interp_img = interp_near_pixels(source, up_scale)
    
    padding = x.shape[0]-interp_img.shape[0]
    source_pad = np.pad(interp_img, (int(padding/2), int(padding/2)) , mode='constant', constant_values=0)
    return source_pad, padding

def HG_mode2d(X, Y, m, n, w):
    """
    Returns a 2D normalized HG mode of beam waist w, (m, n) order, in (x,y) plane
    """
    ux = X / (w * np.sqrt(2))
    uy = Y / (w * np.sqrt(2))
    numerator =  eval_hermite(m, ux) * np.exp(-X**2 / (4*(w**2)) ) * eval_hermite(n, uy) * np.exp(-Y**2 / (4*(w**2)) )
    denominator = (2*np.pi)**(1/4) * np.sqrt( (2**m) * factorial(m) * w ) * (2*np.pi)**(1/4) * np.sqrt( (2**n) * factorial(n) * w )
    mode = numerator / denominator 
    return mode / np.sqrt( np.sum (np.abs(mode)**2) )

def decomposition_basis2d(dim,X,Y,sigma):
    """
    Returns a 2D basis of (dim) HG modes.
    E.g. a = decomposition_basis2d(2) # HG basis: 00, 01, 10, 11
    """
    base_state = np.zeros((dim, dim, X.shape[0], X.shape[1]), dtype='float')
    for m in range(dim):
        for n in range(dim):
            base_state[m, n, :, :] = HG_mode2d(X, Y, m, n, sigma)
    return base_state
    
def propagate(source, PSF):
    """
    Calculates the field distribution in image plane for a given field distribution at the source and Point Spread Function
    """
    image = signal.fftconvolve(source, PSF, mode='same')
    image_norm = image / np.sqrt(np.sum(np.abs(image)**2) +1e-10) # normalize image
    return image_norm
    
def photocurrents2d(image, base_state):
    """
    Returns the photocurrents of heterodyne detections for a given HG basis
    
    Args:
        image -- matrix of complex elements
        base_state -- tensor of dimension [m,n,len(x),len(y)]
        
    Returns:
        J_n -- matrix [m,n]
    """
    dim = np.shape(base_state)[0]
    J_n = np.zeros((dim, dim), dtype='float')
    for i in range(dim):
        for j in range(dim):
            J_n[i,j] = np.sum( image[:,:] * np.conj(base_state[i,j,:,:]) )
    return J_n
    
def beta_coef2d(J_n, base_state):
    """
    Returns the beta coeff of the source electric field from the measured photocurrents
 
    Args:
        J_n -- matrix [m,n]
        base_state -- tensor of dimension [m,n,len(x),len(y)]
        
    Returns:
        beta -- matrix [m,n]
    """
    dim = len(base_state)
    beta = np.zeros((dim, dim), dtype='float')
    for i in range(dim):
        for j in range(dim):
            for m in range(i+1):
                for n in range(j+1):
                    beta[i, j] += (hermite(i).coef[::-1])[m] * (hermite(j).coef[::-1])[n] * sqrt(factorial(m)) * sqrt(factorial(n)) * J_n[m,n]
    return beta
    
    
    
class Simulation:
    def __init__(self,DMD_basis=210,HG_basis=21,wavelength=785e-9,DMD_pixel=7.56 * 1e-6,DMD_pixel_interp=10, NA=0.667e-3,simulation_size=210*3):
        self.DMD_basis = DMD_basis
        self.HG_basis = HG_basis
      
        DMD_logic_pixel = DMD_pixel_interp*DMD_pixel
        unit = wavelength / (2*NA)     
        rayleigh_distance =  1.22 * unit
        simulation_windows = DMD_logic_pixel*3*self.DMD_basis - 0.57e-4 # = 3*21 logical pixels
        x = np.linspace(-simulation_windows/2, simulation_windows/2, num=simulation_size, endpoint=True) 
        y = np.linspace(-simulation_windows/2, simulation_windows/2, num=simulation_size, endpoint=True)
        simulation_unit = x[1] - x[0]
        sigma = 0.21 * wavelength / NA       
        X, Y = np.meshgrid(x, y, sparse=False)
        
        self.x = x
        self.up_scale = int(np.round(DMD_logic_pixel*1.0/simulation_unit))
        self.PSF = (1/ ( ((2*np.pi)**(1/4)) * np.sqrt(sigma) )) * np.exp(-(X/(2*sigma))**2) * np.exp(-(Y/(2*sigma))**2)
        self.basis = decomposition_basis2d(self.HG_basis,X,Y,sigma)
        
        ################################## cache some stuff to speed up calculation #################################
        # hermite polynomials
        self.hermite_pol_coeff_inv = []

        for i in range(self.HG_basis):
            self.hermite_pol_coeff_inv.append(hermite(i).coef[::-1])
            
        # E_HG terms
        self.E_HG_constant_term = np.zeros((self.HG_basis, self.HG_basis, X.shape[0], Y.shape[0]))

        for i in range(self.HG_basis):
                for j in range(self.HG_basis):
                    x_factor = eval_hermite(i, X/(2*sigma)) / (2**(i+1) * factorial(i) * (np.sqrt(np.pi) * sigma))
                    y_factor = eval_hermite(j, Y/(2*sigma)) / (2**(j+1) * factorial(j) * (np.sqrt(np.pi) * sigma))
                    
                    self.E_HG_constant_term[i,j, :, :] = x_factor * y_factor * np.exp(-(X/(2*np.sqrt(2)*sigma))**2) * np.exp(-(Y/(2*np.sqrt(2)*sigma))**2)
        ###############################################################################################################

    def beta_coef2d_fast(self,J_n):
        """
        Returns the beta coeff of the source electric field from the measured photocurrents.
        Uses cached results to speed up computation time.
     
        Args:
            J_n -- matrix [m,n]
            
        Returns:
            beta -- matrix [m,n]

        """
        dim = len(self.basis)
        beta = np.zeros((dim, dim), dtype='float')
        for i in range(dim):
            for j in range(dim):
                for m in range(i+1):
                    for n in range(j+1):
                        beta[i, j] += self.hermite_pol_coeff_inv[i][m] * self.hermite_pol_coeff_inv[j][n] * sqrt(factorial(m)) * sqrt(factorial(n)) * J_n[m,n]
        return beta
        
    def field_image_plane(self,source):
    
        source_pad, _ = source_to_sim_size(source, self.x, self.up_scale)
        
        return propagate(source_pad,self.PSF)


    def HG_microscopy2d(self,image):
        """
        Calculates 2D reconstructed electric field from HG microscopy.

        Args:
            image -- matrix [m,n], electric field (complex array) at the image plane

        Returns:
            E_HG -- matrix [x,y], reconstructed source electric field
            beta -- matrix [m,n], reconstructed source electric field HG beta coefficients
            J_n -- matrix [m,n], heterodyne photocurrents
        """     
        J_n = photocurrents2d(image, self.basis)
        beta = self.beta_coef2d_fast(J_n)
        E_HG = 0
        for i in range(self.HG_basis):
            for j in range(self.HG_basis):                            
                E_HG += beta[i,j] * self.E_HG_constant_term[i,j, :, :]      
                
                
        return E_HG, beta, J_n 
        
    def simulate(self,source): # returns: camera image, image reconstruction, photocurrents (flat and with real and imaginary parts split)
    
        image = self.field_image_plane(source)
        E_HG, _ , J_n = self.HG_microscopy2d(image)
        
        J_n_processed = matrix_complex_to_real_imag(J_n)
        
        I_camera = np.abs(image)**2
        I_HG = np.abs(E_HG)**2
        
        return I_camera, I_HG, J_n_processed


if __name__ == '__main__': main()
