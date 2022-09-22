
# %matplotlib inline

# r = 2
# length = 20
# width = 10
# grid = Grid(r, length, width)
#
# rand = (random.uniform(0, length), random.uniform(0, width))
# data = grid.poisson(rand)
#
# def unzip(items):
#     return ([item[i] for item in items] for i in range(len(items[0])))
#
# points_data = unzip(data)
#
# print('points_data', *points_data)
# plt.scatter(*unzip(data))
# plt.axvline(ymin=2/14, ymax=12/14, color='red')
# plt.axvline(x=20, ymin=2/14, ymax=12/14, color='red')
# plt.axhline(y=10, xmin=5/30, xmax=25/30, color='red')
# plt.axhline(y=0, xmin=5/30, xmax=25/30, color='red')
# plt.show()

# Poisson disc sampling in arbitrary dimensions
# Implementation by Pavel Zun, pavel.zun@gmail.com
# BSD licence - https://github.com/diregoblin/poisson_disc_sampling

# -----------------------------------------------------------------------------
# Based on 2D sampling by Nicolas P. Rougier - https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------
import numpy as np
import poisson_disc as pd
# plotting tools
# %matplotlib widget
import time


def current_milli_time():
    return round(time.time() * 1000)


if __name__ == '__main__':
    # default: 2D points, classic Bridson algorithm as described here:
    # https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    # points = pd.Bridson_sampling()
    #
    # fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    # ax[0].scatter(points[:, 0], points[:, 1], s=10)
    # ax[0].set_xlim(0, 1)
    # ax[0].set_ylim(0, 1)
    # ax[0].set_aspect('equal')

    # alternative sampler, results in denser points:
    # based on the method proposed here by Martin Roberts: http://extremelearning.com.au/an-improved-version-of-bridsons-algorithm-n-for-poisson-disc-sampling/
    time
    dims2d = np.array([1.0, 1.0])
    points_surf = pd.Bridson_sampling(dims=dims2d, radius=0.05, k=30, hypersphere_sample=pd.hypersphere_surface_sample)
    print('points_surf', points_surf)
    #
    # ax[1].scatter(points_surf[:, 0], points_surf[:, 1], s=10)
    # ax[1].set_xlim(0, 1)
    # ax[1].set_ylim(0, 1)
    # ax[1].set_aspect('equal')
    #
    # plt.show()