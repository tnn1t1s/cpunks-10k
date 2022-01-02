import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def show(X, idx=[3307, 281, 510, 741], ncols=2):
    Xs = [X[i] for i in idx]
    nrows= int(len(idx) / ncols)

    fig = plt.figure(figsize=(3., 3.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
             nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
             axes_pad=0.1,  # pad between axes in inch.
             )

    for ax, im in zip(grid, Xs):
        ax.imshow(im)
    plt.show()
