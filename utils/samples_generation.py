import os

import numpy as np
import random
from PIL import Image, ImageOps, ImageDraw, ImageFont
import string
from scipy.ndimage.interpolation import rotate # matrix rotation 
import random # random numbers for random matrix
from math import *

### Functions to generate train and test samples 
def rotate_and_norm(a_matrix, angle, axes=(0,1), threshold = 0.1, reshape=False):
    '''
    Rotate a matrix along axes=(0,1) and set its values to ones to avoid interpolation.
    
    Args:
        a_matrix (matrix) - matrix of dimensions (x, y)
        angle (int) - angle by which each matrix in tensor is rotated within xy plane around origin
        threshold (int) - scipy rotate sometimes interpolate values, so we reset them to one
        reshape (bool) - reshape arg in scipy rotate function. False by default
    '''
    rot_mat = rotate(a_matrix, axes=axes, angle=angle, reshape=reshape)
    return np.where(rot_mat > threshold, 1, 0)

def crop_210_210(a_matrix, out_size = 210):
    '''
    Select the central 210x210 pixels area of a matrix
    
    Args
        a_matrix (tensor) - input matrix of dim (size_x, size_y)
        out_size (int) - length of output matrix, out_size x out_size
    '''
    size_x, size_y = a_matrix.shape[0], a_matrix.shape[1]
    border_x, border_y = int((size_x - out_size)/2), int((size_y - out_size)/2)
    return a_matrix[border_x:border_x+out_size, border_y:border_y+out_size]

def rotate_and_norm_tensor(input_tensor, angle, axes=(1,2), threshold = 0.1, reshape=False):
    '''
    Rotate a tensor along (1,2) axes and set its values to ones to avoid interpolation.
    
    Args:
        input_tensor (tensor) - tensor of dimensions (idx_sample, x, y)
        angle (int) - angle by which each matrix in tensor is rotated within xy plane around origin
        threshold (int) - scipy rotate sometimes interpolate values, so we reset them to one
        reshape (bool) - reshape arg in scipy rotate function. False by default
    '''
    rot_tensor = rotate(input_tensor, axes=axes, angle=angle, reshape=reshape)
    return np.where(rot_tensor > threshold, 1, 0)

def crop_210_210_tensor(input_tensor, out_size = 210):
    '''
    Select the central 210x210 pixel areas from each matrix of a tensor.
    
    Args
        input_tensor (tensor) - input tensor of dim (n_samples, size_x, size_y)
        out_size (int) - length of output matrix, out_size x out_size
        
    Returns
        cropped_tensor (tensor) - output tensor of dim (n_samples, out_size, out_size)
    '''
    size_x, size_y = input_tensor.shape[1], input_tensor.shape[2]
    border_x, border_y = int((size_x - out_size)/2), int((size_y - out_size)/2)
    
    return input_tensor[:, border_x:border_x+out_size, border_y:border_y+out_size]

def generate_Rayleigh_lines(source_basis = 210, line_size = 10):
    ''' Generate pairs of vertical lines with increased spacing, used to estimate Rayleigh distance.'''

    ## array of half spacing distances
    lines_arr_rough = np.arange(0,100,5)
    lines_arr_fine = np.arange(15,60,1)
    lines_arr = np.sort( np.unique( np.concatenate((lines_arr_rough, lines_arr_fine)) ) )

    source = np.zeros((len(lines_arr), source_basis, source_basis))
    
    centre = int(source_basis/2)
    
    for idx, lines_half_spacing in enumerate(lines_arr):
        source[idx, :, centre-lines_half_spacing : centre-lines_half_spacing + line_size] = 1
        source[idx, :, centre+lines_half_spacing : centre+lines_half_spacing + line_size] = 1

    return source
	
def generate_lines_pairs(source_basis = 210):
    source = np.zeros((1764, source_basis, source_basis))
    horiz_lines = vert_lines = diag_lines = antidiag_lines = np.zeros((441, source_basis*3, source_basis*3))
    
    ## idea: genereate double lines along x, then apply rotation to generate them along y and diagonals xy, yx
    step = 10
    idx = 0
    for ii in np.arange(source_basis,source_basis*2,step):
        for jj in np.arange(source_basis,source_basis*2,step):
            horiz_lines[idx,ii:ii+step,:] = 1 
            horiz_lines[idx,jj:jj+step,:] = 1 
            idx+=1
            
    vert_lines = rotate_and_norm_tensor(horiz_lines, 90, threshold = 0.5, reshape=False)
    diag_lines = rotate_and_norm_tensor(horiz_lines, 45, threshold = 0.5, reshape=False)
    antidiag_lines = rotate_and_norm_tensor(horiz_lines, -45, threshold = 0.5, reshape=False)
    
    source = np.concatenate((horiz_lines, vert_lines, diag_lines, antidiag_lines))
    source = crop_210_210_tensor(source, out_size = 210)
    return source	
	
def generate_single_square(square_size, source_basis=210):
    ''' Generate a tensor of single squares, each of size (square_size x square_size)'''
    number_of_samples = int( (source_basis / square_size)**2 )

    single_square = np.zeros((number_of_samples, source_basis, source_basis))
    idx = 0
    for x in np.arange(0, source_basis, square_size):
        for y in np.arange(0, source_basis, square_size):
            single_square[idx, y:y+square_size, x:x+square_size] = 1
            idx+=1
    return single_square

def generate_squares(source_basis = 210):
    ''' Generate an ensamble of single squares matrices. '''
    source = np.zeros((60, source_basis, source_basis))
    
    source[0,:,:] = 1 # white matrix
    source[1,:,:] = 0 # black matrix

    
    source[2:51,:,:] = generate_single_square(30)
    source[51:60,:,:] = generate_single_square(70)

    return source
	
def generate_squares_parts(source_basis = 210):
    ''' Generate a set of squares with a missing piece. '''
    square_size = 30
    missing_part_size = 10

    source = np.zeros((441, source_basis, source_basis))
    idx = 0
    for x in np.arange(0, source_basis, square_size):
        for y in np.arange(0, source_basis, square_size):
            for xx in np.arange(x,x+square_size, missing_part_size):
                for yy in np.arange(y,y+square_size, missing_part_size):
                    source[idx, y:y+square_size, x:x+square_size] = 1
                    source[idx, yy:yy+missing_part_size, xx:xx+missing_part_size] = 0
                    idx+=1
                    
    return source
	
def interp_near_pixels_xy(image, interp_factor_x, interp_factor_y):
    """ 
    Increse size of an image differently along x and y by taking neighbour pixel value.
    Used to increase size of (X,Y), so that PSF and HG modes are smoother.
    """    
    img_interp = np.zeros((image.shape[0]*interp_factor_y,image.shape[1]*interp_factor_x))
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            img_interp[i*interp_factor_y:(i+1)*interp_factor_y,j*interp_factor_x:(j+1)*interp_factor_x] = image[i,j]
    return img_interp

def randmat_skewed(inter_factor_x, inter_factor_y, source_basis = 210, lower_probability = 0, upper_probability = 1):
    """    
    Generates random matrices of zeros or ones, of size (size_y,size-x)    
    The probability of the "ones" matrices is sampled from the interval [0, upper_probability] with equal probabilities 
    upper_probability should range from 0 to 1, with up to 2 decimal places
    """
    probability = np.random.randint(lower_probability*100, upper_probability*100,size=1)[0]/100
    image = np.random.choice(2, ( int(source_basis/10*2), int(source_basis/10*2) ), p=[1-probability, probability])
    image = interp_near_pixels_xy(image, inter_factor_x, inter_factor_y) 
    return image[0:source_basis, 0:source_basis]


def generate_random_matrices(n_samples,
                             block_prob_1 = (0.2, 0.8),
                             blocks_size = (10,50),
                             orientation = (0,361),
                             roll = (0,210),
                             source_basis = 210
                            ):
    """    
    Generates random matrices made of blocks of zeros and ones which are randomly positioned.
    All the blocks are oriented along the same, random, angle.
    
    Args:
        n_samples (int) - Number of random matrices returned
        block_prob_1 (tuple) - Min and max probability that a block of '1' is sampled (e.g. block_prob_1=(0,0.1) -> most matrix elements are zeros)
        blocks_size (tuple) - Min and max size of '0' and '1' blocks
        roll (tuple) - A matrix is randomly rolled along x and y by a shift sampled from this interval
        orientation (tuple) - All the blocks are oriented along an angle, randomly sampled within this interval
        source_basis (int) - Size of the produced random matrix

    Returns:
        rand_matrices (tensor) - tensor of random matrices of dimension (n_samples, source_basis, source_basis)
    """
    
    rand_matrices = np.zeros((n_samples, source_basis, source_basis))

    for idx in range(rand_matrices.shape[0]):

        inter_factor_x = np.random.randint(blocks_size[0], blocks_size[1])
        inter_factor_y = np.random.randint(blocks_size[0], blocks_size[1])

        random_angle = np.random.randint(orientation[0], orientation[1])

        random_roll_x = np.random.randint(roll[0],roll[1])
        random_roll_y = np.random.randint(roll[0],roll[1])

        r_mat = randmat_skewed(
                             inter_factor_x,
                             inter_factor_y,
                             source_basis = source_basis*2,
                             lower_probability = block_prob_1[0],
                             upper_probability = block_prob_1[1])

        r_mat = np.roll(r_mat, random_roll_x, axis=0)
        r_mat = np.roll(r_mat, random_roll_y, axis=1)

        r_mat = rotate_and_norm(r_mat, angle=random_angle)

        rand_matrices[idx,:,:] = crop_210_210(r_mat, out_size = 210)
        
    return rand_matrices
	
def generate_alphabet(source_basis = 210):
    '''
    Generate numbers, letters and a bunch of special characters.
    In total 96 characters.
    Used as part of the training set.
    
    Args:
        source_basis (int) - length of base and height of produced matrices
        
    Returns:
        source (tensor) - produces characters. Tensor of dimensions (n_samples, x, y) 
    '''
    
    alphabet = list(string.printable)
    font_path = "C:\Windows\Fonts\georgia.ttf"

    size = source_basis
    source = np.zeros((96, size, size))
    
    for idx, letter in enumerate(alphabet[:96]):
        font_size = 240

        img = Image.new('1', (size, size))
        fnt = ImageFont.truetype(font_path,font_size) #georgia

        d = ImageDraw.Draw(img)
        w, h = fnt.getsize(letter)

        d.text( ((size-w)/2,-20+(size-h)/2), letter, font=fnt, fill=1) 

    #     plt.pcolormesh(m)
    #     plt.show()
    #     img.save(r'D:\measurements\measurements\p20201209\test_generation_alphabet/'+str(idx)+'.png')

        source[idx,:,:] = np.flipud(np.array(img)).astype('float')

    return source
	
# lines part : from HGM, set it to rand 1-5 parts
def line(source_basis, width, angle, x_center, y_center, L):
    stage = np.zeros((L,L))
    stage[int(L/2)-floor(width/2):int(L/2)+ceil(width/2),:] = 1
    stage = rotate(stage, angle=angle, reshape=False, order=0, mode='constant')
    return stage[int(L/2)+y_center-floor(source_basis/2):int(L/2)+y_center+ceil(source_basis/2),int(L/2)+x_center-floor(source_basis/2):int(L/2)+x_center+ceil(source_basis/2)]
    
def line_random(source_basis, L, min_width, max_width):
    width = np.random.randint(min_width, max_width)
    angle = np.random.randint(0, 180)
    x_center = np.random.randint(-ceil(source_basis/2),floor(source_basis/2))
    y_center = np.random.randint(-ceil(source_basis/2),floor(source_basis/2))
    return line(source_basis,width,angle,x_center,y_center,L)

def lines_random(source_basis = 210,
                L = 200,
                min_width = 1,
                max_width = 8,
                min_lines = 1,
                max_lines = 3):

    number_of_lines = np.random.randint(min_lines, max_lines+1)  
    image = np.zeros(line_random(source_basis, L, min_width, max_width).shape)    
    for i in range(number_of_lines):
        image += line_random(source_basis, L, min_width, max_width)
    overlaps = np.where(image>1)
    image[overlaps]=1
    return image

def generate_lines(
                samples_number,
                source_basis = 210,
                L = 2000,
                min_width = 10,
                max_width = 70,
                min_lines = 1,
                max_lines = 1):
    
    source = np.zeros((samples_number, source_basis, source_basis))
    for s in range(source.shape[0]):
        source[s,:,:] = lines_random(
                source_basis = source_basis,
                L = L,
                min_width = min_width,
                max_width = max_width,
                min_lines = min_lines,
                max_lines = max_lines)
    return source
	
def ellipse_part(source_basis,A,B,L,inner_radius,outer_radius,start_x,start_y): 
    """ 
    Generates part of an ellipse:

    Generates an ellipse ring with principal axes A and B, in a plane of size L x L
    Makes a ring centered at that ellipse    
    Makes a cut of that ellipse, with size DMD_basis x DMD_basis and starting at position (start_y, start_x)
    """        
    ellipse = np.zeros((L,L))

    image = np.zeros((source_basis, source_basis))
    x = np.linspace(0,L-1,L)
    y = np.linspace(0,L-1,L)
    X,Y = np.meshgrid(x,y)
    X = X - L/2
    Y = Y - L/2

    ring = (X/A)**2+(Y/B)**2
    cut = np.where( np.logical_and(ring > inner_radius,ring < outer_radius)  )
    ellipse[cut] = 1
    
#     plt.pcolormesh(ellipse)
#     plt.colorbar()
#     plt.show()

    return ellipse[start_y:start_y+source_basis,start_x:start_x+source_basis]

def ellipse_random(source_basis=210):
    """
    Generates a cut of an ellipse ring, with random axes, radiuses and starting position    
    These parameters are randomly generated in a loop until the final image is not too empty or too full
    """        
    # METHOD PARAMETERS ---
    L = 1000
    axis_lower = 40
    axis_upper = 120
    radius_lower = 10
    radius_upper = 15
    min_thickness = 0.5
    emptyness = 0.1
    fullness = 1
    # ---------------------

    finish = False
    while finish == False:
        A = np.random.randint(axis_lower,axis_upper)
        B = np.random.randint(axis_lower,axis_upper)
        inner_radius = radius_lower
        outer_radius = np.random.randint(  int((radius_lower+min_thickness)*100)  ,radius_upper*100)/100
        start_x = np.random.randint(0,L-source_basis)
        start_y = np.random.randint(0,L-source_basis)

        source = ellipse_part(source_basis,A,B,L,inner_radius,outer_radius,start_x,start_y)

        if np.sum(source)>emptyness*source_basis**2 and np.sum(source)<fullness*source_basis**2:
            finish  = True 

    return source

def ellipse_random_multiple(min_number, max_number, source_basis=210):
    number = np.random.randint(min_number, max_number+1)
    a = np.zeros((source_basis,source_basis))
    for n in range(number):
        a+= ellipse_random(source_basis)
    overlap = np.where(a>1)
    a[overlap]=1
    return a

def generate_ellipses_parts(samples_number, min_number, max_number, source_basis = 210):
    source = np.zeros((samples_number, source_basis, source_basis))
    for s in range(source.shape[0]):
        source[s,:,:] = ellipse_random_multiple(min_number, max_number, source_basis=source_basis)
    return source
	
def load_and_create_sources(img_path):
    '''
    Load a bmp image of size (source_basis*a, source_basis*b) and return a tensor (a*b, source_basis, source_basis).
    '''
    im = Image.open(img_path).convert('1')
    im = np.array(im)
    im = im*-1+1 # invert color
    im = np.flipud(im) # flip upside down since Image.open flipped it
    return matrix_to_210x210_sources(im, source_basis=210)
    
def matrix_to_210x210_sources(matrix_full_img, source_basis=210):
    '''
    Convert a (source_basis*a, source_basis*b) matrix into a tensor (a*b, source_basis, source_basis)
    '''
    full_img_size_x, full_img_size_y = matrix_full_img.shape[0], matrix_full_img.shape[1]
    
    if (np.mod(full_img_size_x, source_basis)!=0) or (np.mod(full_img_size_y, source_basis)!=0):
        print('I cannot create the sources correctly: initial image must have dimensions multiple of "source_basis"')
        return _,_
    
    n_sources_x, n_sources_y = int(full_img_size_x/source_basis), int(full_img_size_y/source_basis)

    sources = np.zeros((n_sources_x * n_sources_y, source_basis, source_basis))
    
    idx = 0
    for x_idx in range(n_sources_x):
        for y_idx in range(n_sources_y):
            x_min, x_max = x_idx*source_basis, x_idx*source_basis + source_basis
            y_min, y_max = y_idx*source_basis, y_idx*source_basis + source_basis
            sources[idx,:,:] = matrix_full_img[x_min:x_max, y_min:y_max]
            idx += 1
            
    return sources

def recompose_img(sources, n_sources_x, n_sources_y, source_basis=210):
    """ Recompose an images from a set of images, like from an ordered puzzle.
    
    Args:
        sources - tensor (num_img_along_x*num_img_along_y, size_img_x, size_img, y)
        num_img_along_x (int) - number of images along x in the full image
        num_img_along_y (int) - number of images along y in the full image

    Outputs:
        reconstructed_source -- matrix (num_img_along_x*size_img_x, num_img_along_y*size_img, y)
    """
    
    size_x, size_y = n_sources_x*source_basis, n_sources_y*source_basis, 
    
    full_img = np.zeros(( size_x, size_y ))
    
    idx = 0
    for x_idx in range(n_sources_x):
        for y_idx in range(n_sources_y):
            x_min, x_max = x_idx*source_basis, x_idx*source_basis + source_basis
            y_min, y_max = y_idx*source_basis, y_idx*source_basis + source_basis
            full_img[x_min:x_max,y_min:y_max] = sources[idx,:,:]
            idx += 1
    return full_img
	

#### Functions to convert sources to DMD images


def interp_near_pixels_xy(image, interp_factor_x, interp_factor_y):
    """ 
    Increse size of an image differently along x and y by taking neighbour pixel value.
    Used to increase size of (X,Y), so that PSF and HG modes are smoother.
    """    
    img_interp = np.zeros((image.shape[0]*interp_factor_y,image.shape[1]*interp_factor_x))
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            img_interp[i*interp_factor_y:(i+1)*interp_factor_y,j*interp_factor_x:(j+1)*interp_factor_x] = image[i,j]
    return img_interp
    
def image_to_1bit_file(file_path, image):
        """
        Converts DMD image to binary bmp file
        """       
        rand_matrix_uint8 = (image*255).astype('uint8')
        im = Image.fromarray(rand_matrix_uint8)
        im = im.convert('1')
        im_flipped = ImageOps.mirror(im.rotate(180))
        im_flipped.save(file_path)
    

class DMD_samples_generation:

    def __init__(self,DMD_basis=210,DMD_size_x=1920, DMD_size_y=1080, DMD_x_min=855, DMD_y_min=435, DMD_pixel_interp=1):
        """
        DMD_basis: size of DMD source image
        DMD_size: size of DMD screen
        DMD_center: position of DMD source image on DMD screen
        DMD_pixel_interp: how many pixels in DMD screen correspond to a single pixel of DMD source image
        """        
        self.DMD_basis = DMD_basis
        self.DMD_size_x = DMD_size_x
        self.DMD_size_y = DMD_size_y
        self.DMD_x_min = DMD_x_min
        self.DMD_y_min = DMD_y_min
        self.DMD_pixel_interp = DMD_pixel_interp

    def source_to_image(self,source):
        """ 
        Converts a source (source_size_y x source_size_x) to DMD bitmap image (DMD_size_y x DMD_size_x)
        """        
        source_size_y, source_size_x = source.shape
        image = np.zeros((self.DMD_size_y, self.DMD_size_x)) 
        image[self.DMD_y_min:self.DMD_y_min+source_size_y*self.DMD_pixel_interp, self.DMD_x_min:self.DMD_x_min+source_size_x*self.DMD_pixel_interp] = interp_near_pixels_xy(source, self.DMD_pixel_interp,self.DMD_pixel_interp) 
        
        return image

    def image_to_source(self, image):
        """
        Converts DMD image (DMD_size_y x DMD_size_x) to DMD source (DMD_basis x DMD_basis)
        """   
        source = np.zeros((self.DMD_basis,self.DMD_basis))
        for i in range(self.DMD_basis):
            for j in range(self.DMD_basis):
                source[i,j] = image[self.DMD_y_min+i*self.DMD_pixel_interp,self.DMD_x_min+j*self.DMD_pixel_interp]
                
        return source

    def phase_ref(self,save_folder):
        dir_path = os.path.dirname(os.path.realpath(__file__))+'/files/'+'phase_ref.bmp'
        image = Image.open(dir_path)
        image.save(save_folder)

        
    def saved_source(self,name,max_number,number_of_samples):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if number_of_samples > max_number:
            print('Number of samples too large! Expect an endless amount of errors from now on')
            return None
        return np.load(dir_path+'/files/'+name+'.npy')


if __name__ == '__main__': main()