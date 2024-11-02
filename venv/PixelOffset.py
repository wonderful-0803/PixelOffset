from jaxfit import CurveFit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
def rotate_coordinates2D(coords, theta):
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                  [jnp.sin(theta), jnp.cos(theta)]])
  
    shape = coords[0].shape
    coords = jnp.stack([coord.flatten() for coord in coords])
    rcoords = R @ coords
    return [jnp.reshape(coord, shape) for coord in rcoords]


def gaussian2d(coords, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    coords = [coords[0] - x0, coords[1] - y0] #translate first
    X, Y = rotate_coordinates2D(coords, theta)
    density = n0 * jnp.exp(-.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2))
    return density + offset


def get_coordinates(width, height):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y


def get_gaussian_parameters(length):
  n0 = 1
  x0 = length / 2
  y0 = length / 2
  sigx = length / 6
  sigy = length / 8
  theta = np.pi / 3

  offset = .1 * n0
  params = [n0, x0, y0, sigx, sigy, theta, offset]
  return params

length = 500
XY_tuple = get_coordinates(length, length)

params = get_gaussian_parameters(length)
zdata = gaussian2d(XY_tuple, *params)
zdata += np.random.normal(0, .1, size=(length, length))

plt.imshow(zdata)
plt.show()

from scipy.optimize import curve_fit

def get_random_float(low, high):
    delta = high - low
    return low + delta * np.random.random()

flat_data = zdata.flatten()
flat_XY_tuple = [coord.flatten() for coord in XY_tuple]
jcf = CurveFit()

loop = 100
times = []
stimes = []
for i in range(loop):
    seed = [val * get_random_float(.9, 1.2) for val in params]
    st = time.time()
    popt, pcov = jcf.curve_fit(gaussian2d, flat_XY_tuple, flat_data, p0=seed)
    times.append(time.time() - st)

popt2, pcov2 = curve_fit(gaussian2d, flat_XY_tuple, flat_data, p0=seed)
print('Average fit time', np.mean(times[1:]))
print('JAXFit parameters', popt)
print('SciPy parameters', popt2)

plt.figure()
plt.plot(times[1:])
plt.xlabel('Fit Number')
plt.ylabel('Fit Speed (seconds)')
plt.show()