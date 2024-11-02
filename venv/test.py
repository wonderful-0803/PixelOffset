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
# 1. Load interference fringe image from .mat file
def load_mat_file(filename):
    mat_data = sio.loadmat(filename)
    fringe_image = mat_data['firstImage']  # assuming the image is stored under this key
    return fringe_image
def wrapTo2Pi(phi):
    # Wrap phi values to the range [0, 2*pi]
    return jnp.remainder(2000 * jnp.pi + phi, 2 * jnp.pi)

# 2. Calculate Kx, Ky using FFT
def get_k_values(image, pixel_size, Fs):
    """
    This function computes Kx and Ky using FFT, similar to the MATLAB code.
    :param image: Input fringe image
    :param pixel_size: Physical size of each pixel (for scaling)
    :param Fs: Sampling frequency
    :return: Kx, Ky (spatial frequencies)
    """
   # Extract a column (v1) and row (u1) from the image
    v1 = image[:, 0]
    u1 = image[0, :]
    
    L_v = len(v1)  # Length of the column
    L_u = len(u1)  # Length of the row
    
    # Frequency array
    f_v = Fs * jnp.arange(0, L_v // 2 + 1) / L_v
    f_u = Fs * jnp.arange(0, L_u // 2 + 1) / L_u
    
    # FFT of the vertical and horizontal line profiles
    F_v = jnp.fft.fft(v1 - jnp.mean(v1))
    F_u = jnp.fft.fft(u1 - jnp.mean(u1))
    
    # Get magnitude (absolute value)
    Pv = 2 * jnp.abs(F_v[:L_v // 2 + 1])
    Pu = 2 * jnp.abs(F_u[:L_u // 2 + 1])
    
    # Find the frequency corresponding to the peak of the spectrum
    index_kv = jnp.argmax(Pv)
    index_ku = jnp.argmax(Pu)
    
    # Check if the peak value is above a threshold to avoid noise
    Ky0 = 2 * jnp.pi * f_v[index_kv] if Pv[index_kv] > 0.05 else 0
    Kx0 = 2 * jnp.pi * f_u[index_ku] if Pu[index_ku] > 0.05 else 0
    
    # Scale by pixel size
    Kx0 /= pixel_size
    Ky0 /= pixel_size
    
    return Kx0, Ky0

# 3. Define the spacial_fit function
def spacial_fit(param_time, param_space, upos, vpos):
    B = param_time[0]
    A = param_time[1]
    Kx = param_time[2]
    Ky = param_time[3]
    #Phi = param_time[4]
    Phi = wrapTo2Pi(param_time[4])
    b = param_space[0] 
    a = param_space[1] 
    vpos_offset = param_space[2] * vpos
    upos_offset = param_space[3] * upos
    phase_offset = param_space[4]
    #phase_offset = wrapTo2Pi(param_space[4])  
    #precomputed = Kx * (upos - upos_offset) + Ky * (vpos - vpos_offset)
    precomputed = Kx * upos_offset + Ky * vpos_offset
    fringe_image = B * b + A * a * jnp.sin(precomputed + Phi + phase_offset)
    return fringe_image
def temporalfit(image, param_T):
    # Extract B, A, and Phi from param_T
    B = param_T[:, 0]  # Background term 
    A = param_T[:, 1]  # Amplitude term
    Phi = param_T[:, 4]  # Phase term
    
    # Get image dimensions (v = rows, u = cols, T = time samples)
    v, u, T = image.shape
    
    # Initialize parameters
    phase = np.zeros((v, u))
    b = np.zeros((v, u))  # Background
    a = np.zeros((v, u))  # Amplitude
    
    # Precompute sin and cos of Phi
    sPhi = np.sin(Phi)
    cPhi = np.cos(Phi)
    
    # Precompute matrix M
    M = np.array([
        [np.sum(B * B), np.sum(B * A * sPhi), np.sum(B * A * cPhi)],
        [np.sum(B * A * sPhi), np.sum(A * A * sPhi * sPhi), np.sum(A * A * sPhi * cPhi)],
        [np.sum(B * A * cPhi), np.sum(A * A * sPhi * cPhi), np.sum(A * A * cPhi * cPhi)]
    ])
    
    # Iterate over all pixels
    for m in range(v):
        for n in range(u):
            # Collect intensity over time for each pixel
            I = image[m, n, :]
            
            # Compute vector N
            N = np.array([
                np.sum(B * I),
                np.sum(A * sPhi * I),
                np.sum(A * cPhi * I)
            ])
            
            # Solve for X (b, a*sin(phase), a*cos(phase))
            X = np.linalg.solve(M, N)
            cc, aa, bb = X
            
            # Compute phase and amplitude for each pixel
            phase[m, n] = np.arctan2(bb, aa)  # Phase
            b[m, n] = cc  # Background
            a[m, n] = np.sqrt(aa**2 + bb**2)  # Amplitude
    
    # Now fit the image over time
    image_tem_fit = np.zeros((v, u, T))
    for i in range(T):
        image_tem_fit[:, :, i] = B[i] * b + A[i] * a * np.sin(Phi[i] + phase)
    
    # Wrap phase to [0, 2*pi]
    phase = wrapTo2Pi(phase)
    
    # Prepare output
    param_fit = {
        'b': b,
        'a': a,
        'phase': phase
    }
    
    return param_fit, image_tem_fit
# 4. Generate coordinate grid for upos, vpos
def get_coordinates(width, height):
    upos = np.linspace(0, width - 1, width)
    vpos = np.linspace(0, height - 1, height)
    upos_grid, vpos_grid = np.meshgrid(upos, vpos)
    return upos_grid, vpos_grid

# 5. Define fitting function for jaxfit and scipy.optimize.curve_fit
def fit_fringe_image(fringedata, upos, vpos, pixel_size, Fs):
    # Calculate Kx, Ky using FFT
    Kx0, Ky0 = get_k_values(fringedata, pixel_size, Fs)
    print("Kx0:", Kx0)
    print("Ky0:", Ky0)

    def fit_func(flat_coords, B, A, Kx, Ky, Phi, b, a, vpos_offset, upos_offset, phase_offset):
        upos_flat, vpos_flat = flat_coords
        param_time = [B, A, Kx, Ky, Phi]
        param_space = [b, a, vpos_offset, upos_offset, phase_offset]
        

        return spacial_fit(param_time, param_space, upos_flat, vpos_flat).flatten()

    # Flatten the fringe data and coordinates
    flat_fringedata = fringedata.flatten()
    flat_upos = upos.flatten()
    flat_vpos = vpos.flatten()
    flat_coords = np.vstack((flat_upos, flat_vpos))
    # Initial guess for fitting parameters, including Kx0 and Ky0
    initial_guess = [0, 0, -Kx0, Ky0, 0, 1, 1, pixel_size, pixel_size, 0]
    cf = CurveFit()
    # Fit using curve_fit
    start_time = timeit.default_timer()
    popt, pcov = cf.curve_fit(fit_func, flat_coords, flat_fringedata, p0=initial_guess)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Fitting function execution time: {execution_time:.2f} seconds")
    return popt, pcov
def getphaseoffset(param_time,param_space,param_temfit):
    
    return

def fit_fringe_row(rowdata, upos, pixel_size, Fs):
    """
    This function fits the 10th row of the fringe image.
    :param rowdata: 1D array (10th row of the image)
    :param upos: 1D array of horizontal positions (for the 10th row)
    :param pixel_size: Size of each pixel
    :param Fs: Sampling frequency
    :return: Optimized parameters (popt) and covariance (pcov)
    """
    # Calculate Kx using FFT (Ky is irrelevant since it's a row)
    Kx0, _ = get_k_values(rowdata.reshape(1, -1), pixel_size, Fs)  # Pass as 1D row
    print("Kx0:", Kx0)
    def fit_func(flat_upos, B, A, Kx, Ky, Phi, b, a, vpos_offset, upos_offset, phase_offset):
        param_time = [B, A, Kx, Ky, Phi]  # Ky is 0 because we're working with 1D
        param_space = [b, a, vpos_offset, upos_offset, phase_offset]  # vpos_offset is 0 for 1D row fitting
        return spacial_fit(param_time, param_space, flat_upos, 0).flatten()

    # Initial guess for fitting parameters
    initial_guess = [0, 0, Kx0, 0, 0, 1, 1, 0,pixel_size, 0]

    # Fit using curve_fit
    cf = CurveFit()
    popt, pcov = cf.curve_fit(fit_func, upos.flatten(), rowdata.flatten(), p0=initial_guess)
    return popt, pcov

# 6. Load data and fit
filename = 'D:\\Software\\matlab2024a\\相机标定\\firstImage.mat'  # Update with your actual .mat file path
#filename = 'D:\\Software\\Matlab2024a\\相机标定\\数据\\0903gaussfit1000p.mat'
fringedata = load_mat_file(filename)
normalized_image = normalize_image(fringedata)
# 取一行
tenth_row_data = normalized_image[9, :]

# Get the coordinates grid based on the fringe image size
upos, vpos = get_coordinates(1000, 1000)
# upos_row = np.linspace(0, normalized_image.shape[1] - 1, normalized_image.shape[1])
# Define pixel size and sampling frequency (Fs)
pixel_size = 4.6e-6  # adjust based on your image setup
Fs = 1   # spatial sampling frequency
# popt_row, pcov_row = fit_fringe_row(tenth_row_data, upos_row, pixel_size, Fs)
# fitted_tenth_row_data = spacial_fit(popt_row[:5], popt_row[5:], upos_row, 0)
# fitted_tenth_row_rescaled = (fitted_tenth_row_data - np.min(fitted_tenth_row_data)) / \
#                             (np.max(fitted_tenth_row_data) - np.min(fitted_tenth_row_data)) * \
#                             (np.max(tenth_row_data) - np.min(tenth_row_data)) + np.min(tenth_row_data)
# Fit the fringe image data

popt, pcov = fit_fringe_image(normalized_image, upos, vpos, pixel_size, Fs)

#popt111 = []
#popt1, pcov1 = fit_fringe_image(tenth_row_data, upos, vpos, pixel_size, Fs)
# 7. Visualize the fit result
fitted_fringes = spacial_fit(popt[:5], popt[5:], upos, vpos)
#fit_tenth_row_data = spacial_fit(popt1[:5], popt1[5:], upos, vpos)

# **调整拟合结果强度到原始图像范围**
fitted_fringes_rescaled = (fitted_fringes - np.min(fitted_fringes)) / (np.max(fitted_fringes) - np.min(fitted_fringes)) * (np.max(normalized_image) - np.min(normalized_image)) + np.min(normalized_image)

# 计算残差
residuals = normalized_image - fitted_fringes


plt.figure(figsize=(12, 4))  # 设置图像大小

# 显示原始图像
plt.subplot(1, 3, 1)
plt.title("Original Fringe Image")
plt.imshow(normalized_image, cmap='gray')
plt.colorbar(label='Intensity')  # 显示颜色条

# 显示拟合图像
plt.subplot(1, 3, 2)
plt.title("Fitted Fringe Image")
plt.imshow(fitted_fringes_rescaled, cmap='gray')
plt.colorbar(label='Intensity')  # 显示颜色条

# plt.subplot(1, 3, 3)
# plt.title("Fitted 10th Row Data")
# plt.plot(upos_row, fitted_tenth_row_rescaled, label='Fitted', color='orange')
# plt.legend()

# 显示残差图像
# plt.subplot(1, 3, 3)
# plt.title("Residuals")
# plt.imshow(residuals, cmap='RdBu_r')  # 使用红蓝颜色映射来突出正负残差
# plt.colorbar(label='Residual Intensity')  # 显示颜色条

# plt.tight_layout()  # 调整子图间距
plt.show()

#print("Fitted Parameters:", popt[5:])
print("pcov:", pcov)