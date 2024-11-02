import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from jaxfit import CurveFit
import jax.numpy as jnp
import timeit
#from scipy.optimize import curve_fit

def normalize_image(image):
    # Subtract the minimum value
    image = image - np.min(image)
    # Normalize by the maximum value and scale to 65535 (16-bit image range)
    image = (image / np.max(image)) * 65535
    return image

# 1. Load multiple interference fringe images from .mat file
def load_mat_file(filename):
    mat_data = sio.loadmat(filename)
    fringe_images = mat_data['image']  # assuming the images are stored under this key
    return fringe_images

def wrapTo2Pi(phi):
    return jnp.remainder(2000 * jnp.pi + phi, 2 * jnp.pi)

# 2. Calculate Kx, Ky using FFT
def get_k_values(image, pixel_size, Fs):
    v1 = image[:, 0]
    u1 = image[0, :]

    L_v = len(v1)
    L_u = len(u1)
    
    f_v = Fs * jnp.arange(0, L_v // 2 + 1) / L_v
    f_u = Fs * jnp.arange(0, L_u // 2 + 1) / L_u
    
    F_v = jnp.fft.fft(v1 - jnp.mean(v1))
    F_u = jnp.fft.fft(u1 - jnp.mean(u1))
    
    Pv = 2 * jnp.abs(F_v[:L_v // 2 + 1])
    Pu = 2 * jnp.abs(F_u[:L_u // 2 + 1])
    
    index_kv = jnp.argmax(Pv)
    index_ku = jnp.argmax(Pu)
    
    Ky0 = 2 * jnp.pi * f_v[index_kv] if Pv[index_kv] > 0.05 else 0
    Kx0 = 2 * jnp.pi * f_u[index_ku] if Pu[index_ku] > 0.05 else 0
    
    Kx0 /= pixel_size
    Ky0 /= pixel_size
    
    return Kx0, Ky0

# 3. Define the spacial_fit function
def spacial_fit(param_time, param_space, upos, vpos):
    B = param_time[0]
    A = param_time[1]
    Kx = param_time[2]
    Ky = param_time[3]
    Phi = wrapTo2Pi(param_time[4])
    b = param_space[0] 
    a = param_space[1] 
    vpos_offset = param_space[2] * vpos
    upos_offset = param_space[3] * upos
    phase_offset = param_space[4]

    precomputed = Kx * upos_offset + Ky * vpos_offset
    fringe_image = B * b + A * a * jnp.sin(precomputed + Phi + phase_offset)
    
    return fringe_image

# 4. Generate coordinate grid for upos, vpos
def get_coordinates(width, height):
    upos = jnp.linspace(0, width - 1, width)
    vpos = jnp.linspace(0, height - 1, height)
    upos_grid, vpos_grid = jnp.meshgrid(upos, vpos)
    return upos_grid, vpos_grid

# 5. Define fitting function for jaxfit and scipy.optimize.curve_fit
def fit_fringe_image(fringedata, upos, vpos, pixel_size, Fs,initial_guess):
    def fit_func(flat_coords, B, A, Kx, Ky, Phi, b, a, vpos_offset, upos_offset, phase_offset):
        upos_flat, vpos_flat = flat_coords
        param_time = [B, A, Kx, Ky, Phi]
        param_space = [b, a, vpos_offset, upos_offset, phase_offset]

        return spacial_fit(param_time, param_space, upos_flat, vpos_flat).flatten()

    flat_fringedata = fringedata.flatten()
    flat_upos = upos.flatten()
    flat_vpos = vpos.flatten()
    flat_coords = np.vstack((flat_upos, flat_vpos))

    
    cf = CurveFit()
    start_time = timeit.default_timer()
    popt, pcov = cf.curve_fit(fit_func, flat_coords, flat_fringedata, p0=initial_guess)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Fitting function execution time: {execution_time:.2f} seconds")
    return popt, pcov

# 6. Fit multiple images and visualize
def fit_multiple_images(images, pixel_size, Fs):
    num_images = images.shape[2]
    
    for i in range(num_images):
        print(f"Processing image {i+1}/{num_images}")
        fringe_image = images[:, :, i]
        normalized_image = normalize_image(fringe_image)

        upos, vpos = get_coordinates(normalized_image.shape[1], normalized_image.shape[0])
        Kx0, Ky0 = get_k_values(fringe_image, pixel_size, Fs)
        print("Kx0:", Kx0)
        print("Ky0:", Ky0)
        initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, pixel_size, pixel_size, 0]

        popt, pcov = fit_fringe_image(normalized_image, upos, vpos, pixel_size, Fs,initial_guess)
        fitted_fringes = spacial_fit(popt[:5], popt[5:], upos, vpos)

        fitted_fringes_rescaled = (fitted_fringes - np.min(fitted_fringes)) / (np.max(fitted_fringes) - np.min(fitted_fringes)) * (np.max(normalized_image) - np.min(normalized_image)) + np.min(normalized_image)
        residuals = normalized_image - fitted_fringes

        plt.figure(figsize=(12, 4))
        
        # Display the original image
        plt.subplot(1, 2, 1)
        plt.title(f"Original Fringe Image {i+1}")
        plt.imshow(normalized_image, cmap='gray')
        plt.colorbar(label='Intensity')

        # Display the fitted image
        plt.subplot(1, 2, 2)
        plt.title(f"Fitted Fringe Image {i+1}")
        plt.imshow(fitted_fringes_rescaled, cmap='gray')
        plt.colorbar(label='Intensity')

        plt.show()

        print(f"Fitted Parameters for image {i+1}: {popt[:5]}")

# 7. Load data and fit
filename = 'D:\\Software\\matlab2024a\\相机标定\\new.mat'
fringe_images = load_mat_file(filename)

# Define pixel size and sampling frequency (Fs)
pixel_size = 4.6e-6
Fs = 1

# Fit all images in the loaded .mat file

fit_multiple_images(fringe_images, pixel_size, Fs)
