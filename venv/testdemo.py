
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from jaxfit import CurveFit
import jax.numpy as jnp
import h5py
import timeit
from fittest import FringeImageFitter 
# Initialize and fit images
# 0827allfit512p
filename = 'D:\\Software\\Matlab2024a\\程序整理 0603\\程序整理\\0508gaussfitsingle600p.mat'
pixel_size = 4.6e-6
Fs = 1
sam_fre = 60  # example value
drive_fre = 1  # example value
win_length = 300  # example window size
v_temp = 0  # example v start
u_temp = 0  # example u start
iternum = 1 
set_num = 2
sample_per_set = 1
area1 = {
    'v_temp': 130,
    'u_temp': 121,
    'intersec': {'v_1': 11, 'v_2': win_length, 'u_1': 1, 'u_2': win_length}
}
area2 = {
    'v_temp': 140,
    'u_temp': 121,
    'intersec': {'v_1': 1, 'v_2': win_length - 10, 'u_1': 1, 'u_2': win_length}
}
# Create an instance of the class
start_time_cmv = timeit.default_timer()
cmv = FringeImageFitter(filename, pixel_size, Fs, sam_fre, drive_fre, win_length, area1['v_temp'], area1['u_temp'],sample_per_set)
end_time_cmv = timeit.default_timer()
cmv_time = start_time_cmv - end_time_cmv
print(f"cmv init time: {cmv_time} seconds")
start_time = timeit.default_timer()
image = cmv.load_mat_file(filename)
end_time = timeit.default_timer()
load_time = end_time - start_time
print(f"Load time: {load_time} seconds")
allimage = image.astype(jnp.float64)
# allimage = jnp.array(image, dtype=jnp.float64)
u,v,T = image.shape
# u, v, T = allimage.shape
print("allimage type",type(allimage))
for i in range(T):
     img = image[:,:,i]
     img_min = jnp.min(img)
     img = img - img_min
     img_max = jnp.max(img)
     img = (img / img_max) * 65535
     allimage[:,:,i] = img
# 处理每一帧
# for i in range(T):
#     img = allimage[:, :, i]  # 直接从 allimage 获取当前图像
#     img_min = jnp.min(img)
#     img = img - img_min
#     img_max = jnp.max(img)
#     img = (img / img_max) * 65535
#     allimage = allimage.at[:, :, i].set(img)  # 更新 allimage
# print("allimage's type",type(allimage))
# allimage = jnp.array(allimage)
# print("after change allimage's type",type(allimage))
start_pipeline_time = timeit.default_timer()
cmv.pixel_offset_pipeline(set_num, allimage, sample_per_set, iternum)
end_pipeline_time = timeit.default_timer()
pipeline_time = end_pipeline_time - start_pipeline_time
print(f"Pipeline time: {pipeline_time} seconds")

cmv2 = FringeImageFitter(filename, pixel_size, Fs, sam_fre, drive_fre, win_length, area2['v_temp'], area2['u_temp'],sample_per_set)
cmv2.pixel_offset_pipeline(set_num, allimage, sample_per_set, iternum)
# print("cmv2 results",len(cmv2.results))
# print("cmv2 error",cmv2.error)
# # Display results for area 1
print(f'Area 1 - v_temp: {area1["v_temp"]}, u_temp: {area1["u_temp"]}, win_length: {win_length}')
print(f'Area 1 - Pixel Offset std: {cmv.results[1]["std"]}')
print(f'Area 1 - Repeatability: {cmv.repeat_std}')
print(f'Area 1 - Correlation: {cmv.error[0]["corr_coef"]}')
# Display results for area 2
print(f'Area 2 - v_temp: {area2["v_temp"]}, u_temp: {area2["u_temp"]}, win_length: {win_length}')
print(f'Area 2 - Pixel Offset std: {cmv2.results[1]["std"]}')
print(f'Area 2 - Repeatability: {cmv2.repeat_std}')
print(f'Area 2 - Correlation: {cmv2.error[0]["corr_coef"]}')

# Compare two tests in the same area
pixel_offset_area1 = cmv.results[1]['pixel_offset'][area1['intersec']['v_1']:area1['intersec']['v_2'],
                                                    area1['intersec']['u_1']:area1['intersec']['u_2']]
pixel_offset_area2 = cmv2.results[1]['pixel_offset'][area2['intersec']['v_1']:area2['intersec']['v_2'],
                                                    area2['intersec']['u_1']:area2['intersec']['u_2']]
coef_areas = jnp.corrcoef(pixel_offset_area1.flatten(), pixel_offset_area2.flatten())
pixel_offset_diff_area = pixel_offset_area1 - pixel_offset_area2
std_diff_area = jnp.std(pixel_offset_diff_area)

# Display cross-window results
print(f'Cross-window std difference: {std_diff_area}')
print(f'Cross-window correlation: {coef_areas[0, 1]}')

U, V = jnp.meshgrid(jnp.arange(area1['intersec']['u_1'], area1['intersec']['u_2']),
                   jnp.arange(area1['intersec']['v_1'], area1['intersec']['v_2']))
# Plot pixel offset difference
# plt.figure()
# plt.imshow(pixel_offset_diff_area, extent=[area1['intersec']['u_1'], area1['intersec']['u_2'], 
#                                             area1['intersec']['v_1'], area1['intersec']['v_2']],
#            aspect='auto', cmap='jet')
# plt.colorbar(label='Pixel Offset Difference')
# plt.title('Pixel Offset Difference Area')
# plt.xlabel('u')
# plt.ylabel('v')
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U, V, pixel_offset_diff_area, cmap='jet')

ax.set_title('3D Pixel Offset Difference')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('Pixel Offset Difference')
plt.colorbar(ax.plot_surface(U, V, pixel_offset_diff_area, cmap='jet'), ax=ax, label='Pixel Offset Difference')

plt.show()

# Plot in v direction
plt.figure()
plt.plot(pixel_offset_area1[0, :], 'b.-', label='Area 1')
plt.plot(pixel_offset_area2[0, :], 'ro-', label='Area 2')
plt.plot(pixel_offset_area1[0, :] - pixel_offset_area2[0, :], 'k-', label='Difference')
plt.title('V-direction Pixel Offset Difference')
plt.xlabel('u')
plt.ylabel('Pixel Offset')
plt.legend()
plt.grid(True)
plt.show()

# Plot in u direction
plt.figure()
plt.plot(pixel_offset_area1[:, 0], 'b.-', label='Area 1')
plt.plot(pixel_offset_area2[:, 0], 'ro-', label='Area 2')
plt.plot(pixel_offset_area1[:, 0] - pixel_offset_area2[:, 0], 'k-', label='Difference')
plt.title('U-direction Pixel Offset Difference')
plt.xlabel('v')
plt.ylabel('Pixel Offset')
plt.legend()
plt.grid(True)
plt.show()










