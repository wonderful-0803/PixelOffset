import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from jaxfit import CurveFit
from jax import jit
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import vmap
import jax
import timeit

class FringeImageFitter:
    results = []
    error = []
    repeat_std = []
    def __init__(self, filename, pixel_size, Fs, sam_fre, drive_fre, win_length, v_temp, u_temp,sample_per_set):
        self.filename = filename
        self.pixel_size = pixel_size
        self.Fs = Fs
        self.sam_fre = sam_fre  # 采样频率
        self.drive_fre = drive_fre  # 驱动频率
        self.win_length = win_length  # 窗口尺寸
        self.v_temp = v_temp  # v 方向的起始点
        self.u_temp = u_temp  # u 方向的起始点
        self.sample_per_set = sample_per_set
        self.fringe_images = self.load_mat_file(filename)

    # 1. 加载数据
    # def load_mat_file(self, filename):
    #     #mat_data = sio.loadmat(filename)
    #     with h5py.File(filename, 'r') as file: 
    #         fringe_images = file['/image'][:] # 打开 HDF5 文件
    #     #fringe_images = mat_data['image']  # assuming the images are stored under this key
    #     return fringe_images
    def load_mat_file(self, filename, block_size=50):
        with h5py.File(filename, 'r') as file:
            dataset = file['/image'][:]
            
            total_slices = dataset.shape[0]
            
            # 初始化空列表以存储分块数据
            fringe_images = []
            
            # 分块读取数据
            for start in range(0, total_slices, block_size):
                end = min(start + block_size, total_slices)
                block = dataset[start:end, :, :]  # 读取块数据
                fringe_images.append(block)
            
            # 将块数据合并成一个 NumPy 数组
            fringe_images = np.concatenate(fringe_images, axis=0)
            
        return fringe_images
    # 归一化图片 of 16-bit 
    def normalize_image(self, image):
        image = image - jnp.min(image)
        image = (image / jnp.max(image)) * 65535
        return image
    
    # 窗口加载
    def winload(self, allimage, period_per_set, set_num):
        self.sample_per_period = int(self.sam_fre * 1 / self.drive_fre)
        self.sample_per_set = self.sample_per_period * period_per_set
        winsize = [self.win_length, self.win_length, self.sample_per_set]
        sample_start = self.sample_per_set * (set_num - 1)
        image = allimage[self.v_temp:self.v_temp + self.win_length,
                         self.u_temp:self.u_temp + self.win_length,
                         sample_start:sample_start + self.sample_per_set]
        DNmax = 65535
        image = image / DNmax
        return image
    # 2. 计算 Kx, Ky 
    def get_k_values(self, image):
        v1 = image[:, 0, 0]
        u1 = image[0, :, 0]

        L_v = len(v1)
        L_u = len(u1)

        f_v = self.Fs * jnp.arange(0, L_v // 2 + 1) / L_v
        f_u = self.Fs * jnp.arange(0, L_u // 2 + 1) / L_u

        F_v = jnp.fft.fft(v1 - jnp.mean(v1))
        F_u = jnp.fft.fft(u1 - jnp.mean(u1))

        Pv = 2 * jnp.abs(F_v[:L_v // 2 + 1])
        Pu = 2 * jnp.abs(F_u[:L_u // 2 + 1])

        index_kv = jnp.argmax(Pv)
        index_ku = jnp.argmax(Pu)

        Ky0 = 2 * jnp.pi * f_v[index_kv] if Pv[index_kv] > 0.05 else 0
        Kx0 = 2 * jnp.pi * f_u[index_ku] if Pu[index_ku] > 0.05 else 0

        Kx0 /= self.pixel_size
        Ky0 /= self.pixel_size

        return Kx0, Ky0

    # Wrap phase to 2*pi
    # def wrapTo2Pi(self, phi):
    #     return jnp.remainder(2000 * jnp.pi + phi, 2 * jnp.pi) 
    def wrapTo2Pi(self,lambda_):
    # 将角度归一化到 [0, 2*pi]
        lambda_ = jnp.mod(lambda_, 2 * jnp.pi)
        
        # 处理 lambda 为零的情况，正输入对应于 2*pi
        positive_input = lambda_ > 0
        lambda_ = jnp.where((lambda_ == 0) & positive_input, 2 * jnp.pi, lambda_)
        
        return lambda_  
    def get_phase_offset(self, param_T, param_space, param_fit):
        # 从 param_T 中提取 Kx 和 Ky 的均值
        Kx = jnp.mean(param_T[:, 2])  # Python 索引从 0 开始，列 3 在 param_T 的索引是 2
        Ky = jnp.mean(param_T[:, 3])  # 列 4 在 param_T 的索引是 3
        
        # 提取 param_space 中的 u 和 v 位置
        upos = param_space['u']
        vpos = param_space['v']
        
        # 计算理想相位 phase0
        phase0 = self.wrapTo2Pi(Kx * upos + Ky * vpos)
        
        # 获取拟合相位 phase
        phase = param_fit[2]
        
        # 计算相位偏移
        phase_offset = phase - phase0
        
        # 确保 phase_offset 在(-2*pi 到 2*pi)内 
        phase_offset = jnp.where(phase_offset > 4, phase_offset - 2 * jnp.pi, phase_offset)
        phase_offset = jnp.where(phase_offset < -4, phase_offset + 2 * jnp.pi, phase_offset)
        # print(phase_offset)
        return phase_offset, phase0, phase
    # 3. 拟合函数
    def spacial_fit(self, param_time, param_space, upos, vpos):

        B = param_time[0]
        A = param_time[1]
        Kx = param_time[2]
        Ky = param_time[3]
        Phi = self.wrapTo2Pi(param_time[4])
        b = param_space[0]
        a = param_space[1]
        vpos_offset = param_space[2] * vpos
        upos_offset = param_space[3] * upos
        phase_offset = param_space[4]

        precomputed = Kx * upos_offset + Ky * vpos_offset
        fringe_image = B * b + A * a * jnp.sin(precomputed + Phi + phase_offset)

        return fringe_image

    # 4. Generate coordinate grid for upos, vpos
    def get_coordinates(self, width, height):
        upos = jnp.linspace(0, width - 1, width)
        vpos = jnp.linspace(0, height - 1, height)
        upos_grid, vpos_grid = jnp.meshgrid(upos, vpos)
        return upos_grid, vpos_grid

    # 5. Define fitting function for jaxfit
    def fit_fringe_image(self, fringedata, upos, vpos, initial_guess):
        def fit_func(flat_coords, B, A, Kx, Ky, Phi, b, a, vpos_offset, upos_offset, phase_offset):
            upos_flat, vpos_flat = flat_coords
            param_time = [B, A, Kx, Ky, Phi]
            param_space = [b, a, vpos_offset, upos_offset, phase_offset]
            return self.spacial_fit(param_time, param_space, upos_flat, vpos_flat).flatten()
        # 原代码
        flat_fringedata = fringedata.flatten()
        flat_upos = upos.flatten()
        flat_vpos = vpos.flatten()
        flat_coords = np.vstack((flat_upos, flat_vpos))

        # 确保数据为 NumPy 数组

        # flat_fringedata = jnp.asarray(fringedata.flatten())
        # flat_upos = jnp.asarray(upos.flatten())
        # flat_vpos = jnp.asarray(vpos.flatten())
        # flat_coords = jnp.vstack((flat_upos, flat_vpos))
        cf = CurveFit()
        start_time = timeit.default_timer()
        popt, pcov = cf.curve_fit(fit_func, flat_coords, flat_fringedata, p0=initial_guess)
        end_time = timeit.default_timer()

        execution_time = end_time - start_time
        print(f"Fitting function execution time: {execution_time:.2f} seconds")
        return popt, pcov
    def temporalfit(self, image, param_T):
        print('param_T size', param_T.shape)

        # Extract B, A, and Phi from param_T
        B = param_T[:, 0]  # Background term
        A = param_T[:, 1]  # Amplitude term
        Phi = param_T[:, 4]  # Phase term

        # Get image dimensions (v = rows, u = cols, T = time samples)
        v, u, T = image.shape

        # Precompute sin and cos of Phi
        sPhi = jnp.sin(Phi)
        cPhi = jnp.cos(Phi)

        # Precompute matrix M
        M = jnp.array([
            [jnp.sum(B * B), jnp.sum(B * A * sPhi), jnp.sum(B * A * cPhi)],
            [jnp.sum(B * A * sPhi), jnp.sum(A * A * sPhi * sPhi), jnp.sum(A * A * sPhi * cPhi)],
            [jnp.sum(B * A * cPhi), jnp.sum(A * A * sPhi * cPhi), jnp.sum(A * A * cPhi * cPhi)]
        ])

        # Collect intensity over time for each pixel
        I = image.reshape(v * u, T)

        # Compute vector N
        N = jnp.empty((v * u, 3))  # Preallocate space for N
        N = N.at[:, 0].set(jnp.sum(B * I, axis=1))
        N = N.at[:, 1].set(jnp.sum(A * sPhi * I, axis=1))
        N = N.at[:, 2].set(jnp.sum(A * cPhi * I, axis=1))

        # Solve for X (b, a*sin(phase), a*cos(phase))
        X = jnp.linalg.solve(M, N.T)

        cc, aa, bb = X

        # Compute phase and amplitude for each pixel
        phase = jnp.arctan2(bb, aa).reshape(v, u)  # 相位
        b = cc.reshape(v, u)  # Background
        a = jnp.sqrt(aa**2 + bb**2).reshape(v, u)  # Amplitude

        # Now fit the image over time
        image_tem_fit = jnp.zeros((v, u, T))
        for i in range(T):
            image_tem_fit = image_tem_fit.at[:, :, i].set(B[i] * b + A[i] * a * jnp.sin(Phi[i] + phase))

        # Wrap phase to [0, 2*pi]
        phase = self.wrapTo2Pi(phase)

        # Prepare output
        param_fit = [b, a, phase]
        return param_fit, image_tem_fit

        # print('param_T size',param_T.shape)
        # # Extract B, A, and Phi from param_T
        # B = param_T[:,0]  # Background term
        # A = param_T[:,1]  # Amplitude term
        # Phi = param_T[:,4]  # Phase term
        # # print('B',B)
        # # print('A',A)
        # # print('Phi',Phi)
        # # Get image dimensions (v = rows, u = cols, T = time samples)
        # v, u, T = image.shape

        # # Initialize parameters
        # phase = jnp.zeros((v, u))
        # b = jnp.zeros((v, u))  # Background
        # a = jnp.zeros((v, u))  # Amplitude

        # # Precompute sin and cos of Phi
        # sPhi = jnp.sin(Phi)
        # cPhi = jnp.cos(Phi)

        # # Precompute matrix M
        # M = jnp.array([
        #     [jnp.sum(B * B), jnp.sum(B * A * sPhi), jnp.sum(B * A * cPhi)],
        #     [jnp.sum(B * A * sPhi), jnp.sum(A * A * sPhi * sPhi), jnp.sum(A * A * sPhi * cPhi)],
        #     [jnp.sum(B * A * cPhi), jnp.sum(A * A * sPhi * cPhi), jnp.sum(A * A * cPhi * cPhi)]
        # ])
        # # I = np.zeros((T,1))
        # # I = np.zeros(T)
        # # Iterate over all pixels
        # for m in range(v):
        #     for n in range(u):
        #         # Collect intensity over time for each pixel
        #         I = image[m, n, :]

        #         # Compute vector N
        #         N = jnp.array([
        #             jnp.sum(B * I),
        #             jnp.sum(A * sPhi * I),
        #             jnp.sum(A * cPhi * I)
        #         ])
        #         # Solve for X (b, a*sin(phase), a*cos(phase))
        #         # X = np.linalg.solve(M, N)
        #         X = jnp.linalg.solve(M,N)
        #         cc, aa, bb = X

        #         # Compute phase and amplitude for each pixel

        #         phase[m, n] = jnp.arctan2(bb, aa)  # 相位
        #         b[m, n] = cc  # Background
        #         a[m, n] = jnp.sqrt(aa**2 + bb**2)  # Amplitude
        #         # phase = phase.at[m, n].set(jnp.arctan2(bb, aa))
        #         # b = b.at[m, n].set(cc)  # Background
        #         # a = a.at[m, n].set(jnp.sqrt(aa**2 + bb**2))  # Amplitude
        # # Now fit the image over time
        # image_tem_fit = jnp.zeros((v, u, T))
        # # for i in range(T):
        # #     image_tem_fit[:, :, i] = B[i] * b + A[i] * a * jnp.sin(Phi[i] + phase)
        # for i in range(T):
        #     image_tem_fit = image_tem_fit.at[:, :, i].set(B[i] * b + A[i] * a * jnp.sin(Phi[i] + phase))
        # # Wrap phase to [0, 2*pi]
        # phase = self.wrapTo2Pi(phase)
        # # Prepare output
        # param_fit = [b,a,phase]
        # return param_fit, image_tem_fit
    # 6. Fit multiple images and visualize  ---修改前---
    # def fit_multiple_images(self):
    #     num_images = self.fringe_images.shape[2]
        
    #     for i in range(num_images):
    #         print(f"Processing image {i+1}/{num_images}")
    #         fringe_image = self.fringe_images[:, :, i]
    #         normalized_image = self.normalize_image(fringe_image)

    #         upos, vpos = self.get_coordinates(normalized_image.shape[1], normalized_image.shape[0])
    #         Kx0, Ky0 = self.get_k_values(fringe_image)
    #         print("Kx0:", Kx0)
    #         print("Ky0:", Ky0)
    #         initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]

    #         popt, pcov = self.fit_fringe_image(normalized_image, upos, vpos, initial_guess)
    #         fitted_fringes = self.spacial_fit(popt[:5], popt[5:], upos, vpos)

    #         fitted_fringes_rescaled = (fitted_fringes - np.min(fitted_fringes)) / (np.max(fitted_fringes) - np.min(fitted_fringes)) * (np.max(normalized_image) - np.min(normalized_image)) + np.min(normalized_image)
    #         residuals = normalized_image - fitted_fringes

    #         # plt.figure(figsize=(12, 4))
            
    #         # # Display the original image
    #         # plt.subplot(1, 2, 1)
    #         # plt.title(f"Original Fringe Image {i+1}")
    #         # plt.imshow(normalized_image, cmap='gray')
    #         # plt.colorbar(label='Intensity')

    #         # # Display the fitted image
    #         # plt.subplot(1, 2, 2)
    #         # plt.title(f"Fitted Fringe Image {i+1}")
    #         # plt.imshow(fitted_fringes_rescaled, cmap='gray')
    #         # plt.colorbar(label='Intensity')

    #         # plt.show()

    #         print(f"Fitted Parameters for image {i+1}: {popt[:5]}")
    
    # def fit_single_image(self, img, Kx0, Ky0):
    #     upos, vpos = self.get_coordinates(img.shape[1], img.shape[0])
    #     initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
    #     popt, _ = self.fit_fringe_image(img, upos, vpos, initial_guess)
    #     return popt[:5]
    
    # 非并行策略
    def fit_multiple_images(self, image,iternum):
        # num_images = self.fringe_images.shape[2]
        # fringe_image = self.fringe_images[:, :, 0] 
        u = self.win_length
        v = self.win_length
        upos = jnp.linspace(0, u - 1, u)
        vpos = jnp.linspace(0, v - 1, v)
        upos_grid, vpos_grid = jnp.meshgrid(upos, vpos)
        Kx0, Ky0 = self.get_k_values(image)
        print("Kx0:", Kx0,"Ky0:", Ky0)
        # normalized_image = self.normalize_image(self.fringe_images) 
        # upos, vpos = self.get_coordinates(normalized_image.shape[1], normalized_image.shape[0])

        # Ensure that 'fringe_image' is defined before being passed to 'get_k_values'
        # For the first image (or any image you wish to use for initial Kx0, Ky0):
         # Use the first image or any specific one

        # Now pass the properly defined 'fringe_image' to 'get_k_values'
        
        # initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
        param_space_t = {
                    'b' : jnp.ones((u,v)),
                    'a' : jnp.ones((u,v)),
                    'v' : vpos_grid * self.pixel_size,
                    'u' : upos_grid * self.pixel_size,
                    'phase_offset' : jnp.zeros((u,v))
                    }
        start_iter_time = timeit.default_timer()
        for iter in range(iternum):

            print(f"Beginning iteration {iter + 1}")
            param_T = jnp.array([], dtype=float).reshape(0, 5)
            result = {
                'pixel_offset': None,
                'param_T': None,
                'param_space': None,
                'std': None
            }
            # initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
            for t in range(self.sample_per_set):
                img = image[:, :, t]
                #fringe_image = self.fringe_images[:, :, t]
                ## 归一化----
                # normalized_image = self.normalize_image(img)
                
                upos, vpos = self.get_coordinates(img.shape[1], img.shape[0])
                # upos, vpos = self.get_coordinates(img.shape[1], img.shape[0])
                
                initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
                # fringe_image = self.fringe_images[:, :, t]
                # normalized_image = self.normalize_image(self.fringe_images) 
                # upos, vpos = self.get_coordinates(normalized_image.shape[1], normalized_image.shape[0])
                
                popt, pcov = self.fit_fringe_image(img, upos, vpos, initial_guess)

                param_time = popt[:5]
                param_T = jnp.vstack((param_T, param_time))
                # plt.figure(figsize=(12, 4))
                
                # # Display the original image
                # plt.subplot(1, 2, 1)
                # plt.title(f"Original Fringe Image {t+1}")
                # plt.imshow(normalized_image, cmap='gray')
                # plt.colorbar(label='Intensity')

                # # Display the fitted image
                # plt.subplot(1, 2, 2)
                # plt.title(f"Fitted Fringe Image {t+1}")
                # plt.imshow(fitted_fringes_rescaled, cmap='gray')
                # plt.colorbar(label='Intensity')

                # plt.show()
                #print(f"Fitted Parameters for image {t+1}: {popt[:5]}")
            start_time = timeit.default_timer()   
            fitted_fringes = self.spacial_fit(param_time, popt[5:], upos, vpos)
            end_time = timeit.default_timer()           
            execution_time = end_time - start_time
            print(f"Time taken by spacial_fit function: {execution_time:.8f} seconds")
            print("param_T:",param_T)
            param_fit, image_tem_fit = self.temporalfit(image, param_T)
            # 绘图
            # fig = plt.figure(61, figsize=(10, 12))
            # # Subplot 1: b
            # ax1 = fig.add_subplot(311, projection='3d')
            # U, V = np.meshgrid(np.arange(param_fit[0].shape[0]), np.arange(param_fit[0].shape[1]))
            # ax1.plot_surface(U, V, param_fit[0], cmap='viridis')
            # ax1.set_title('b')
            # ax1.set_xlabel('u')
            # ax1.set_ylabel('v')

            # # Subplot 2: a
            # ax2 = fig.add_subplot(312, projection='3d')
            # ax2.plot_surface(U, V, param_fit[1], cmap='viridis')
            # ax2.set_title('a')
            # ax2.set_xlabel('u')
            # ax2.set_ylabel('v')

            # # Subplot 3: Phase Offset
            # ax3 = fig.add_subplot(313, projection='3d')
            # ax3.plot_surface(U, V, param_fit[2], cmap='viridis')
            # ax3.set_title('Phase space')
            # ax3.set_xlabel('u')
            # ax3.set_ylabel('v')
            # # Show the plots
            # plt.tight_layout()
            # plt.show()

            # phase offset
            phase_offset, phase0, phase = self.get_phase_offset(param_T, param_space_t, param_fit)
            # 绘图
            # fig = plt.figure(61, figsize=(10, 12))
        
            # # Subplot 1: Phase
            # ax1 = fig.add_subplot(311, projection='3d')
            # U, V = np.meshgrid(np.arange(phase.shape[0]), np.arange(phase.shape[1]))
            # ax1.plot_surface(U, V, phase, cmap='viridis')
            # ax1.set_title('Phase')
            # ax1.set_xlabel('u')
            # ax1.set_ylabel('v')

            # # Subplot 2: Phase0
            # ax2 = fig.add_subplot(312, projection='3d')
            # ax2.plot_surface(U, V, phase0, cmap='viridis')
            # ax2.set_title('Phase0')
            # ax2.set_xlabel('u')
            # ax2.set_ylabel('v')

            # # Subplot 3: Phase Offset
            # ax3 = fig.add_subplot(313, projection='3d')
            # ax3.plot_surface(U, V, phase_offset, cmap='viridis')
            # ax3.set_title('Phase Offset')
            # ax3.set_xlabel('u')
            # ax3.set_ylabel('v')

            # # Show the plots
            # plt.tight_layout()
            # plt.show()
            kx = jnp.mean(param_T[:,2])
            ky = 1j * jnp.mean(param_T[:,3])
            k_space = jnp.abs(kx + ky)
            position_offset = phase_offset / k_space
            pixel_offset = position_offset / self.pixel_size
            std_pixeloffset = jnp.std(pixel_offset)
            print(f"Iteration {iter + 1}: The standard deviation of the pixel offset is {std_pixeloffset}")

            param_space_t['phase_offset'] = phase_offset
            param_space_t['b'] = param_fit[0]
            param_space_t['a'] = param_fit[1]
            
            result['pixel_offset'] = pixel_offset
            result['param_T'] = param_T
            result['param_space'] = param_fit[5:]
            result['std'] = std_pixeloffset
            self.results.append(result)
        end_iter_time = timeit.default_timer()
        iter_time = end_iter_time - start_time
        print(f"Time taken by iteration {iter + 1}: {iter_time:.8f} seconds")

            # fitted_fringes_rescaled = (fitted_fringes - np.min(fitted_fringes)) / (np.max(fitted_fringes) - np.min(fitted_fringes)) * (np.max(normalized_image) - np.min(normalized_image)) + np.min(normalized_image)

        return result
    # def fit_multiple_images(self, image, iternum):
    #     u = self.win_length
    #     v = self.win_length
    #     upos = jnp.linspace(0, u - 1, u)
    #     vpos = jnp.linspace(0, v - 1, v)
    #     upos_grid, vpos_grid = jnp.meshgrid(upos, vpos)
    #     Kx0, Ky0 = self.get_k_values(image)
    #     print("Kx0:", Kx0, "Ky0:", Ky0)
        
    #     param_space_t = {
    #         'b': jnp.ones((u, v)),
    #         'a': jnp.ones((u, v)),
    #         'v': vpos_grid * self.pixel_size,
    #         'u': upos_grid * self.pixel_size,
    #         'phase_offset': jnp.zeros((u, v))
    #     }

    #     for iter in range(iternum):
    #         print(f"Beginning iteration {iter + 1}")
    #         param_T = jnp.empty((0, 5))
    #         result = {
    #             'pixel_offset': None,
    #             'param_T': None,
    #             'param_space': None,
    #             'std': None
    #         }

    #         # 矢量化图像拟合过程
    #         def fit_single_image(img):
    #             img = jax.lax.stop_gradient(img)
    #             print("img type:", type(img))
    #             initial_guess = [0, 0, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
    #             upos, vpos = self.get_coordinates(img.shape[1], img.shape[0])
    #             flat_fringedata = img.flatten()  # 确保为 DeviceArray
    #             popt, pcov = self.fit_fringe_image(flat_fringedata, upos,vpos, initial_guess)
    #             return popt[:5]

    #         # 使用 vmap 批量处理图像
    #         param_time_results = vmap(lambda img: fit_single_image(jax.lax.stop_gradient(img)))(image)
    #         param_T = jnp.vstack((param_T, param_time_results))

    #         # 计算拟合结果
    #         fitted_fringes = self.spacial_fit(param_time_results[-1], param_space_t['a'], upos, vpos)

    #         param_fit, image_tem_fit = self.temporalfit(image, param_T)

    #         phase_offset, phase0, phase = self.get_phase_offset(param_T, param_space_t, param_fit)

    #         kx = jnp.mean(param_T[:, 2])
    #         ky = jnp.mean(param_T[:, 3])
    #         k_space = jnp.abs(kx + 1j * ky)
    #         position_offset = phase_offset / k_space
    #         pixel_offset = position_offset / self.pixel_size
    #         std_pixeloffset = jnp.std(pixel_offset)
    #         print(f"Iteration {iter + 1}: The standard deviation of the pixel offset is {std_pixeloffset}")

    #         param_space_t['phase_offset'] = phase_offset
    #         param_space_t['b'] = param_fit[0]
    #         param_space_t['a'] = param_fit[1]

    #         result['pixel_offset'] = pixel_offset
    #         result['param_T'] = param_T
    #         result['param_space'] = param_fit[5:]
    #         result['std'] = std_pixeloffset
    #         self.results.append(result)

    #     return result
    # def fit_multiple_images(self, image, iternum):
    #     u = self.win_length
    #     v = self.win_length
    #     upos = jnp.linspace(0, u - 1, u)
    #     vpos = jnp.linspace(0, v - 1, v)
    #     upos_grid, vpos_grid = jnp.meshgrid(upos, vpos)
    #     Kx0, Ky0 = self.get_k_values(image)

    #     param_space_t = {
    #         'b': jnp.ones((u, v)),
    #         'a': jnp.ones((u, v)),
    #         'v': vpos_grid * self.pixel_size,
    #         'u': upos_grid * self.pixel_size,
    #         'phase_offset': jnp.zeros((u, v))
    #     }
    #     for iter in range(iternum):
    #         print(f"Beginning iteration {iter + 1}")

    #         # 使用 vmap 进行并行拟合
    #         fit_function = vmap(lambda img: self.fit_single_image(img, Kx0, Ky0))
    #         popt_list = fit_function(image)  # 返回每幅图像的拟合参数

    #         # 取最后一幅图像的参数用于空间拟合
    #         last_popt = popt_list[-1]  

    #         # 进行空间拟合
    #         fitted_fringes = self.spacial_fit(last_popt, upos, vpos)  # 使用最后一组拟合参数

    #         # 进行时间拟合
    #         param_fit, image_tem_fit = self.temporalfit(image, popt_list)

    #         # 计算相位偏移
    #         phase_offset, phase0, phase = self.get_phase_offset(popt_list, param_space_t, param_fit)

    #         # 计算像素偏移
    #         kx = jnp.mean(popt_list[:, 2])
    #         ky = jnp.mean(popt_list[:, 3])
    #         k_space = jnp.abs(kx + 1j * ky)
    #         position_offset = phase_offset / k_space
    #         pixel_offset = position_offset / self.pixel_size
    #         std_pixeloffset = jnp.std(pixel_offset)

    #         print(f"Iteration {iter + 1}: The standard deviation of the pixel offset is {std_pixeloffset}")

    #         param_space_t['phase_offset'] = phase_offset
    #         param_space_t['b'] = param_fit[0]
    #         param_space_t['a'] = param_fit[1]

    #         result = {
    #             'pixel_offset': pixel_offset,
    #             'param_T': popt_list,  # 返回所有图像的参数
    #             'param_space': param_fit[5:],
    #             'std': std_pixeloffset
    #         }
    #         self.results.append(result)

    #     return result
    def repeat_error(self):
        set_num = len(self.results)
        self.error = []

        for i in range(set_num - 1):
            for j in range(i + 1, set_num):
                error_entry = {}
                error_entry['diff'] = self.results[i]['pixel_offset'] - self.results[j]['pixel_offset']
                error_entry['diff_std'] = jnp.std(error_entry['diff'])

                # Correlation coefficient between pixel offsets
                coef = jnp.corrcoef(self.results[i]['pixel_offset'].flatten(),
                                self.results[j]['pixel_offset'].flatten())
                error_entry['corr_coef'] = coef[0, 1]

                self.error.append(error_entry)
        # print("error diff",self.error[0]['diff'])
        # Set the repeat_std based on the first error entry
        self.repeat_std = jnp.std(self.error[0]['diff'])

    def pixel_offset_pipeline(self,set_num,allimage,period_per_set,iternum):
        result = []
        for i in range(set_num):
            image = self.winload(allimage,period_per_set,i+1)
            result_i = self.fit_multiple_images(image,iternum)
            result.append(result_i)
        self.results = result
        self.repeat_error()