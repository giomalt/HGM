import numpy as np
import zipfile
import sys
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

from pathlib import Path
from cv2 import resize, INTER_AREA

## our custom libraries
sys.path.append('../utils/')
from utils.samples_simulation import *
from utils.samples_generation import *
# from utils.data_processing import * # <- old version, do not use
from utils.data_processing_parallel import *
    
print('Currently acceptable names for sources generation:\n ellipses, randmatNxM, randmatNxM_skewed (N,M = 1,...,9), black_image, white_image, lines, square, square_part, logo, alphabet, lines_random')

class Dataset:

    # MISSING: colab, create/erase/check folders, delete pretend Im not here

    def __init__(self, data_folder='',
                 
                 DMD_basis=210,
                 HG_basis=21,
                 
                 simulated_images_size=250,
                 
                 useful_frames=398,
                 scope_trace_points=55000,
                 ch0_vert_scale = 20,
                 ch1_vert_scale = 3,
                 
                 wavelength = 785e-9,
                 DMD_pixel = 7.56 * 1e-6,
                 DMD_pixel_interp = 1,
                 NA = 0.667e-3
                ):
           
        self.data_folder = data_folder
        self.DMD_basis = DMD_basis
        self.HG_basis = HG_basis
                
        self.simulated_images_size = simulated_images_size
        self.useful_frames = useful_frames
        self.scope_trace_points = scope_trace_points
        
        self.blueprint = list()    
        self.total_samples = 0
        self.sources = None
        self.sim_photocurrents = None
        self.sim_camera = None
        self.sim_HG_reconstruction = None
        self.exp_photocurrents = None
        
        self.wavelength = wavelength
        self.DMD_pixel = DMD_pixel
        self.DMD_pixel_interp = DMD_pixel_interp
        self.NA = NA

        self.DMD = DMD_samples_generation(DMD_basis = self.DMD_basis)
        
        self.Simulation = None
                
    def reset(self):
        """
        Resets everything
        """    
        self.blueprint = list()  
        self.total_samples = 0
        self.sources = None
        self.sim_photocurrents = None
        self.sim_camera = None
        self.sim_HG_reconstruction = None
        self.exp_photocurrents = None
        
        print('The dataset has been successfully reset')
        
    def blueprint_update_total(self):
        """
        Updates the total number of samples. Auxiliary method
        """
        self.total_samples = 0
        for line in self.blueprint: 
            self.total_samples+=int(line[0])

    def blueprint_print(self):
        """
        Prints blueprint
        """
        if self.blueprint is None:
            print('Blueprint is empty!')
        else:
            print('This is the current blueprint:')
            for line in self.blueprint:
                print('number of samples: '+str(line[0])+', name of samples: '+str(line[1]) )
            print('Total number of samples = '+str(self.total_samples))

    def blueprint_load(self,filename,reset=False):
        """
        Loads blueprint from a txt file. This file should consist of lines as: number_of_samples name_of_samples
        """  
        if reset==True:
            self.reset()
            
        file = open(filename, "r" )   
        for line in file:
            self.blueprint.append(line.split())
            
        self.blueprint_update_total()
            
    def blueprint_save(self,filename):
        """
        Saves blueprint to txt file
        """   
        if self.blueprint is None:
            return print('There is no blueprint to be saved!')
        
        with open(filename, 'w') as f:
            for line in self.blueprint:
                f.write("%s %s \n" % (line[0], line[1]))   
                    
    def blueprint_add_line(self,number_of_samples,name_of_samples):
        """
        Appends a line to the blueprint
        """
        self.blueprint.append([number_of_samples,name_of_samples])
        
        self.blueprint_update_total()
            
    def sources_generate(self):
        """
        Generates sources from blueprint instructions. Saves them in the dataset memory.
        A source is a matrix of dimension (DMD_basis, DMD_basis), used to create DMD bitmap files and for HGM simulation.
        """            
        if self.blueprint is None:
            return print('Blueprint is empty!')
        
        self.sources = np.zeros((self.total_samples,self.DMD_basis,self.DMD_basis))

        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            
            # ~ Test dataset ~ #
            if name_of_samples == 'logo':
                self.sources[i:i+number_of_samples,:,:] = load_and_create_sources(
                    os.path.join(self.data_folder, "sources/images/logo.bmp"))
                
            elif name_of_samples == 'logo_full':
                self.sources[i:i+number_of_samples,:,:] = load_and_create_sources(
                    os.path.join(self.data_folder, "sources/images/logo_full.bmp"))
                
            elif name_of_samples == 'lines_rayleigh':
                self.sources[i:i+number_of_samples,:,:] = generate_Rayleigh_lines(source_basis = self.DMD_basis, line_size = 10)
                
            elif name_of_samples == 'alphabet':
                self.sources[i:i+number_of_samples,:,:] = generate_alphabet()

            # ~ Train dataset ~ #
            elif name_of_samples == 'lines_pair':
                self.sources[i:i+number_of_samples,:,:] = generate_lines_pairs(source_basis = self.DMD_basis)
                
            elif name_of_samples == 'square':
                self.sources[i:i+number_of_samples,:,:] = generate_squares(source_basis = self.DMD_basis)
                
            elif name_of_samples == 'square_part':
                self.sources[i:i+number_of_samples,:,:] = generate_squares_parts(source_basis = self.DMD_basis)
             
            elif name_of_samples == 'rand_ellipse':
                self.sources[i:i+number_of_samples,:,:] = generate_ellipses_parts(
                    number_of_samples, 1, 1, source_basis=self.DMD_basis)
                
            elif name_of_samples == 'rand_ellipses':
                self.sources[i:i+number_of_samples,:,:] = generate_ellipses_parts(
                    number_of_samples, 1, 5, source_basis=self.DMD_basis)
                
            elif name_of_samples == 'rand_line':
                self.sources[i:i+number_of_samples,:,:] = generate_lines(
                            number_of_samples,
                            source_basis = self.DMD_basis,
                            L = 2000,
                            min_width = 10,
                            max_width = 70,
                            min_lines = 1,
                            max_lines = 1)

            elif name_of_samples == 'rand_lines':
                self.sources[i:i+number_of_samples,:,:] = generate_lines(
                            number_of_samples,
                            source_basis = self.DMD_basis,
                            L = 2000,
                            min_width = 10,
                            max_width = 70,
                            min_lines = 1,
                            max_lines = 5)

            elif name_of_samples == 'rand_matrix':
                self.sources[i:i+number_of_samples,:,:] = generate_random_matrices(number_of_samples,
                             block_prob_1 = (0.2, 0.8),
                             blocks_size = (10,50),
                             orientation = (0,361),
                             roll = (0,self.DMD_basis),
                             source_basis = self.DMD_basis)
                
            else:
                print('The name '+name_of_samples+' is not currently acceptable')
                return None
                
            i+=number_of_samples
            
    def sources_save(self):
        """
        Saves sources (stored in dataset memory) as numpy compressed tensors npz
        """   
        if self.sources is None:
            return print('There are no sources to be saved!')
               
        Path(self.data_folder+'/sources').mkdir(parents=True, exist_ok=True)
        
        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            np.savez_compressed(self.data_folder+'/sources/'+name_of_samples+'.npz',a=self.sources[i:i+number_of_samples])
            #torch.save(torch.from_numpy(self.sources[i:i+number_of_samples]), self.data_folder+'/sources/'+name_of_samples+'.pt')
            i+=number_of_samples
            
    def sources_load(self):
        """
        Loads sources from saved numpy tensors
        """            
        self.sources = np.zeros((self.total_samples,self.DMD_basis,self.DMD_basis))

        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            self.sources[i:i+number_of_samples,:,:] = np.load(self.data_folder+'/sources/'+name_of_samples+'.npz')['a'][:number_of_samples,:,:]
            i+=number_of_samples
            
    def sources_plot(self,name,number):
        """
        Plots a specific source from dataset memory
        """
        if self.sources is None:
            return print('No sources are loaded!')       
        success = False
        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            if str(name)==name_of_samples:
                if int(number) >= number_of_samples:
                    return print('Source number out of bounds!')
                else:
                    selected = self.sources[i+number,:,:]
                    success = True
                break
            i+=number_of_samples
        if success == False:
            return print('There is no such source name!')
        else:
            plt.pcolormesh(selected)
            plt.title('Source: '+name+' , number = '+str(number))
            plt.colorbar()
            plt.show()
            
    def sources_plot_random(self):
        """
        Plots a random source from dataset memory
        """
        if self.sources is None:
            return print('No sources are loaded!')
        i_selected = np.random.randint(0,self.total_samples)
        selected = self.sources[i_selected,:,:]
        
        i=0
        for line in self.blueprint:      
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            for n in range(number_of_samples):
                if i == i_selected:
                    name = name_of_samples
                    number = n
                    break
                i+=1
            else:
                continue
            break
        
        
        plt.pcolormesh(selected)
        plt.title('Random sample')
        plt.title('Source: '+name+' , number = '+str(number))
        plt.colorbar()
        plt.show()
            
    def DMD_frames_save(self):
        """
        Saves sources as DMD images, inside zipped folders
        """
        if self.sources is None:
            return print('No sources are loaded!')
        Path(self.data_folder+'/samples_zipped').mkdir(parents=True, exist_ok=True)
        useless_file = self.data_folder+'/samples_zipped/useless.bmp'
        
        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            
            with zipfile.ZipFile(self.data_folder+'/samples_zipped/'+name_of_samples+'.zip','w', zipfile.ZIP_DEFLATED) as zip:     
                for n in range(0,number_of_samples,1):
                    DMD_image = self.DMD.source_to_image(self.sources[i,:,:])
                    image_to_1bit_file(useless_file, DMD_image)
                    zip.write(useless_file,arcname=str(n)+'.bmp')
                    os.remove(useless_file)
                    i+=1 
   
    def DMD_frames_unzip(self):
        """
        Unzips the DMD images to the folder samples
        """
        Path(self.data_folder+'/samples').mkdir(parents=True, exist_ok=True)
        self.DMD.phase_ref(self.data_folder+'/samples/phase_ref.bmp')
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            
            with zipfile.ZipFile(self.data_folder+'/samples_zipped/'+name_of_samples+'.zip','r') as zip:     
                for n in range(0,number_of_samples,1):
                    zip.extract(str(n)+'.bmp',path=self.data_folder+'/samples/'+name_of_samples)
    
    def sim_init(self):
        self.Simulation = Simulation(
                DMD_basis = self.DMD_basis,
                HG_basis = self.HG_basis,
                wavelength = self.wavelength,
                DMD_pixel = self.DMD_pixel,
                DMD_pixel_interp = self.DMD_pixel_interp,
                NA = self.NA,
                simulation_size = self.DMD_basis*3)

    def sim_generate(self, save_temp=False, out_size=50):  
        """
        Generates the simulated data: photocurrents, camera images and HG reconstructed images
        """
        if self.sources is None:
            return print('There are no sources loaded!')
        
        Path(self.data_folder+'/sim_photocurrents').mkdir(parents=True, exist_ok=True)
        Path(self.data_folder+'/sim_camera').mkdir(parents=True, exist_ok=True)
        Path(self.data_folder+'/sim_HG_reconstruction').mkdir(parents=True, exist_ok=True)

        Path(self.data_folder+f'/sim_HG_reconstruction/{self.HG_basis}/temp').mkdir(parents=True, exist_ok=True)

        self.sim_photocurrents = np.zeros((self.total_samples,self.HG_basis*self.HG_basis*2))
        self.sim_camera = np.zeros((self.total_samples,self.simulated_images_size,self.simulated_images_size))
        self.sim_HG_reconstruction = np.zeros((self.total_samples,self.simulated_images_size,self.simulated_images_size))
        
        centre = int(self.DMD_basis*3/2)
        half = int(self.simulated_images_size/2)
        
        resize = transforms.Compose([transforms.Resize((out_size,out_size))])

        if save_temp:
            for i in range(0,self.total_samples):
#                 camera, HG_reconstruction, photocurrents = self.Simulation.simulate(self.sources[i])
                _, HG_reconstruction, _ = self.Simulation.simulate(self.sources[i])
    
                HGimg_low_res = resize(torch.from_numpy(HG_reconstruction[centre-half:centre+half,centre-
                                                                          half:centre+half]).unsqueeze(0)).squeeze()   
                torch.save(HGimg_low_res, 
                           self.data_folder+f'/sim_HG_reconstruction/{self.HG_basis}/temp/{i}.pt')
                del HG_reconstruction, HGimg_low_res
        else:
            for i in range(0,self.total_samples):
                camera, HG_reconstruction, self.sim_photocurrents[i] = self.Simulation.simulate(self.sources[i])

                self.sim_camera[i] = camera[centre-half:centre+half,centre-half:centre+half] 
                #resize(camera, (self.simulation_scale,self.simulation_scale), interpolation = INTER_AREA)
                self.sim_HG_reconstruction[i] = HG_reconstruction[centre-half:centre+half,centre-half:centre+half] 
                #resize(HG_reconstruction, (self.simulation_scale,self.simulation_scale), interpolation = INTER_AREA)
        print('sim over, go in peace')

    def sim_save(self, load_temp=False, out_size=50):
        """
        Saves the simulations as torch tensors
        """
#         if self.sim_camera is None:
#             return print('No camera images are loaded!')
#         if self.sim_HG_reconstruction is None:
#             return print('No HG reconstructions are loaded!')
#         if self.sim_photocurrents is None:
#             return print('No photocurrents are loaded!')
    
#         Path(self.data_folder+'/sim_photocurrents').mkdir(parents=True, exist_ok=True)
#         Path(self.data_folder+'/sim_camera').mkdir(parents=True, exist_ok=True)
#         Path(self.data_folder+'/sim_HG_reconstruction').mkdir(parents=True, exist_ok=True)

        i=0
    
        sample_idx=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            
            if load_temp:
#                 sim_photocurrents = np.zeros((number_of_samples,self.HG_basis*self.HG_basis*2))
                sim_HG_reconstruction = np.zeros((number_of_samples,out_size,out_size))

                for j in range(number_of_samples):
                    sim_HG_reconstruction[j] = \
                        torch.load(f'{self.data_folder}/sim_HG_reconstruction/{self.HG_basis}/temp/{sample_idx}.pt')
                    sample_idx+=1
                    
                torch.save(torch.from_numpy(sim_HG_reconstruction), 
                           f'{self.data_folder}/sim_HG_reconstruction/{self.HG_basis}/{name_of_samples}.pt')
                del sim_HG_reconstruction


            else:
                torch.save(torch.from_numpy(self.sim_photocurrents[i:i+number_of_samples]), 
                           self.data_folder+'/sim_photocurrents/'+name_of_samples+'.pt')
                torch.save(torch.from_numpy(self.sim_camera[i:i+number_of_samples]), 
                           self.data_folder+'/sim_camera/'+name_of_samples+'.pt')
                torch.save(torch.from_numpy(self.sim_HG_reconstruction[i:i+number_of_samples]), 
                           self.data_folder+'/sim_HG_reconstruction/'+name_of_samples+'.pt')
            
            i+=number_of_samples
            
        
    def sim_load(self):
        """
        Loads the simulations from torch tensors
        """
        self.sim_photocurrents = np.zeros((self.total_samples,self.HG_basis*self.HG_basis*2))
        self.sim_camera = np.zeros((self.total_samples,self.simulated_images_size,self.simulated_images_size),dtype=np.float32)
        self.sim_HG_reconstruction = np.zeros((self.total_samples,self.simulated_images_size,self.simulated_images_size),dtype=np.float32)

        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            self.sim_photocurrents[i:i+number_of_samples] = torch.load(self.data_folder+'/sim_photocurrents/'+name_of_samples+'.pt')[:number_of_samples]
            self.sim_camera[i:i+number_of_samples] = torch.load(self.data_folder+'/sim_camera/'+name_of_samples+'.pt')[:number_of_samples]
            self.sim_HG_reconstruction[i:i+number_of_samples] = torch.load(self.data_folder+'/sim_HG_reconstruction/'+name_of_samples+'.pt')[:number_of_samples]
            i+=number_of_samples
            
    def sim_plot(self,name,number):
        """
        Plots a specific simulated camera and HG reconstruction from dataset memory
        """
        if self.sim_camera is None:
            return print('No camera images are loaded!')
        if self.sim_HG_reconstruction is None:
            return print('No HG reconstructions are loaded!')
        
        success = False
        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            if str(name)==name_of_samples:
                if int(number) >= number_of_samples:
                    return print('Sample number out of bounds!')
                else:
                    selected_camera = self.sim_camera[i+number,:,:]
                    selected_HG_reconstruction = self.sim_HG_reconstruction[i+number,:,:]
                    success = True
                break
            i+=number_of_samples
        if success == False:
            return print('There is no such sample name!')
        else:
            fig = plt.figure(figsize=(13, 4))
            plt.subplot(1, 2, 1)
            plt.pcolormesh(selected_camera)
            plt.title('Camera image')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.pcolormesh(selected_HG_reconstruction)
            plt.title('HG reconstruction')
            plt.colorbar()
            plt.show()
            
    def sim_plot_random(self):
        """
        Plots random simulated camera and HG reconstruction from dataset memory
        """
        if self.sim_camera is None:
            return print('No camera images are loaded!')
        if self.sim_HG_reconstruction is None:
            return print('No HG reconstructions are loaded!')
        
        i_selected = np.random.randint(0,self.total_samples)
        selected_camera = self.sim_camera[i_selected]
        selected_HG_reconstruction = self.sim_HG_reconstruction[i_selected]
        
        i=0
        for line in self.blueprint:      
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            for n in range(number_of_samples):
                if i == i_selected:
                    name = name_of_samples
                    number = n
                    break
                i+=1
            else:
                continue
            break

        print('source: '+name+', number = '+str(number))

        fig = plt.figure(figsize=(13, 4))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(selected_camera)
        plt.title('Camera image')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.pcolormesh(selected_HG_reconstruction)
        plt.title('HG reconstruction')
        plt.colorbar()
        plt.show()
        
    def sequences_save(self):
        """
        Saves the DMD sequences for pattern on the fly mode
        """  
        
        number_of_sequences = np.int(np.ceil(self.total_samples / self.useful_frames))
    
        sequence = list()
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            for n in range(number_of_samples):
                sequence.append('../samples/'+name_of_samples+'/'+str(n)+'.bmp'+',1,105,0,1,0,1'+'\n')
        
        i = 0
        for seq_n in range(number_of_sequences):
            text = 'Normal Mode'+ '\n'
            for frame_index in range(self.useful_frames):
                if i >= self.total_samples:
                    break
                text += sequence[i]
                i+=1
            text += '../samples/point_source_centre960_100x100.bmp,1,105,1050,1,0,1' + '\n' +'../samples/phase_ref.bmp,1,105,0,1,0,1'
            with open(self.data_folder+'/sequences/'+str(seq_n)+'.txt', 'w') as f:
                f.write(text)
                
    def exp_preprocess(self,
                       source_folder,
                       save_folder,
                       scope_trace_points = 215000,
                       ch0_vert_scale = 20,
                       ch1_vert_scale = 3,
                       guess_fit_freq = 0.52*0.5,

                       repetition = 1,

                       moving_avg_kernel = None, 
                       
                       parallel_processing = False,
                       debug = True,
                       
                       load_fits = False,
                       power_norm = False): 
    
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        Path(save_folder+'seq/').mkdir(parents=True, exist_ok=True)

        self.exp_photocurrents = preprocess(
            source_folder,
            self.total_samples,
            self.useful_frames,
            self.HG_basis,
            
            save_folder = save_folder,
            
            scope_trace_points = scope_trace_points,
            ch0_vert_scale = ch0_vert_scale,
            ch1_vert_scale = ch1_vert_scale,
            guess_fit_freq = guess_fit_freq,
            
            repetition = repetition,
            
            moving_avg_kernel = moving_avg_kernel,
            
            parallel_processing = parallel_processing,
            debug = debug,
            
            load_fits = load_fits,
            
            power_norm = power_norm
        )
        
        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            torch.save(torch.from_numpy(self.exp_photocurrents[i:i+number_of_samples]), save_folder+name_of_samples+'.pt')
            i+=number_of_samples
            
            
            
    
    def exp_preprocess_repetition(self,
                                  number_of_repetitions,
                                  source_folder,
                                  save_folder): 
    
        
        for n in range(number_of_repetitions):
            print('Processing acquisition # {} \n'.format(n))
            
            repet_save_folder = save_folder+str(n)+'/'
            Path(repet_save_folder).mkdir(parents=True, exist_ok=True)
            Path(repet_save_folder+'/seq/').mkdir(parents=True, exist_ok=True) # processed amp and phase in each seq
            
            self.exp_photocurrents = preprocess_repetition(source_folder,self.total_samples,self.useful_frames,self.HG_basis,n,scope_trace_points=self.scope_trace_points,save_folder=repet_save_folder)
            
            i=0
            for line in self.blueprint:
                print('Processing # {} \n'.format(line))
                number_of_samples, name_of_samples = int(line[0]), str(line[1])
                torch.save(torch.from_numpy(self.exp_photocurrents[i:i+number_of_samples]), save_folder+str(n)+'/'+name_of_samples+'.pt')
                i+=number_of_samples
    
    def exp_load(self):
    
        self.exp_photocurrents = np.zeros((self.total_samples,self.HG_basis*self.HG_basis*2))

        i=0
        for line in self.blueprint:
            number_of_samples, name_of_samples = int(line[0]), str(line[1])
            self.exp_photocurrents[i:i+number_of_samples] = torch.load(save_folder+name_of_samples+'.pt')[:number_of_samples]
            i+=number_of_samples
            
        
if __name__ == '__main__': main()                

    