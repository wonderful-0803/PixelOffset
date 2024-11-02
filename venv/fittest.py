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
        Phi = param_time[4]
        # Phi = self.wrapTo2Pi(param_time[4])
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
                
                initial_guess = [5, 5, Kx0, Ky0, 0, 1, 1, self.pixel_size, self.pixel_size, 0]
                # fringe_image = self.fringe_images[:, :, t]
                # normalized_image = self.normalize_image(self.fringe_images) 
                # upos, vpos = self.get_coordinates(normalized_image.shape[1], normalized_image.shape[0])
                
                popt, pcov = self.fit_fringe_image(img, upos, vpos, initial_guess)

                param_time = popt[:5]
                param_T = jnp.vstack((param_T, param_time))
            start_time = timeit.default_timer()   
            fitted_fringes = self.spacial_fit(param_time, popt[5:], upos, vpos)
            end_time = timeit.default_timer()           
            execution_time = end_time - start_time
            print(f"Time taken by spacial_fit function: {execution_time:.8f} seconds")
            # print("param_T:",param_T)
            param_fit, image_tem_fit = self.temporalfit(image, param_T)
            # 绘图

            # phase offset
            phase_offset, phase0, phase = self.get_phase_offset(param_T, param_space_t, param_fit)
            # 绘图
            
            kx = jnp.mean(param_T[:,2])
            ky = 0+1j * jnp.mean(param_T[:,3])
            # ky_python_scientific = "{:.4e}{}".format(ky.real, ky.imag * 1j)
            k_space = kx + ky
            k_space = jnp.abs(k_space)
            position_offset = phase_offset / k_space
            pixel_offset = position_offset / (self.pixel_size * 100)
            std_pixeloffset = jnp.std(pixel_offset)
            print(f"Iteration {iter + 1}: The kx is {kx}")
            print(f"Iteration {iter + 1}: The ky is {ky}")
            # print(f"Iteration {iter + 1}: The k_space is {k_space}")
            # print(f"Iteration {iter + 1}: The phase_offset offset is {phase_offset}")

            param_space_t['phase_offset'] = phase_offset
            param_space_t['b'] = param_fit[0]
            param_space_t['a'] = param_fit[1]
            
            result['pixel_offset'] = pixel_offset
            result['param_T'] = param_T
            result['param_space'] = param_fit[5:]
            result['std'] = std_pixeloffset
            self.results.append(result)
            print("pixel_offset:",self.results[iter]['pixel_offset'])

        end_iter_time = timeit.default_timer()
        iter_time = end_iter_time - start_time
        print(f"Time taken by iteration {iter + 1}: {iter_time:.8f} seconds")

            # fitted_fringes_rescaled = (fitted_fringes - np.min(fitted_fringes)) / (np.max(fitted_fringes) - np.min(fitted_fringes)) * (np.max(normalized_image) - np.min(normalized_image)) + np.min(normalized_image)

        return result
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