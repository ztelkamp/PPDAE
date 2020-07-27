"""Utility functions

This file contains utility functions used during training and/or evaluation.

functions:

    * plot_wall_time_series - creates a wall plot of real an reconstrute LCs,
                              used during model training
    * count_parameters      - return number of model's trainable parameters
    * days_hours_minutes    - return number of days, hours, and minutes from a
                              date/time string
    * plot_latent_space     - creates a figure with latent distributions
                              during model training
    * str2bool              - convert Y/N string to bool
    * scatter_hue           - creates a color-codded scatter plot
"""

import os, re, glob
import socket
import numpy as np
import pandas as pd
import matplotlib
if socket.gethostname() == 'exalearn':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.cm as cm
from matplotlib import colors

path = os.path.dirname(os.getcwd())


def plot_recon_wall(xhat, x, epoch=0, log=True):
    """Light-curves wall plot, function used during VAE training phase.
    Figure designed and ready to be appended to W&B logger.

    Parameters
    ----------
    xhat : numpy array
        Array of generated light curves
    x    : numpy array
        List of real light curves.
    epoch: int, optional
        Epoch number

    Returns
    -------
    fig
        a matplotlib figure
    image
        an image version of the figure
    """

    plt.close('all')
    ncols = 10
    fig, axis = plt.subplots(nrows=3, ncols=ncols, figsize=(ncols, 4))
    v_min = np.min([np.min(x), np.min(xhat)])
    v_max = np.max([np.max(x), np.max(xhat)])
    if log:
        norm = colors.SymLogNorm(vmin=v_min, vmax=v_max, linthresh=1)
    else:
        norm = None
    
    for i in range(ncols):
        im1 = axis[0, i].imshow(x[i, 0, :, :], interpolation='bilinear',
                          cmap=cm.viridis, origin='upper', aspect='equal',
                         vmin=v_min, vmax=v_max, norm=norm)
        im2 = axis[1, i].imshow(xhat[i, 0, :, :], interpolation='bilinear',
                          cmap=cm.viridis, origin='upper', aspect='equal',
                         vmin=v_min, vmax=v_max, norm=norm)
        im3 = axis[2, i].imshow(x[i, 0, :, :] - xhat[i, 0, :, :],
                          interpolation='bilinear',
                          cmap=cm.viridis, origin='upper', aspect='equal',
                         vmin=v_min, vmax=v_max, norm=norm)

    for ax in axis.ravel():
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1)
    fig.suptitle('Reconstruction [Epoch %s]' % epoch,
                 fontsize=20, y=.95)

    ax_list = []
    for i in range(3):
      for j in range(ncols):
        ax_list.append(axis[i,j])
      
    fig.colorbar(im1, ax = ax_list)

    fig.canvas.draw()
    return fig


def count_parameters(model):
    """Calculate the number of trainable parameters of a Pytorch moel.

    Parameters
    ----------
    model : pytorh model
        Pytorch model

    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def days_hours_minutes(dt):
    """Convert ellapsed time to Days, hours, minutes, and seconds.

    Parameters
    ----------
    dt : value
        Ellapsed time

    Returns
    -------
    d
        Days
    h
        Hours
    m
        Min
    s
        Seconds
    """
    totsec = dt.total_seconds()
    d = dt.days
    h = totsec // 3600
    m = (totsec % 3600) // 60
    sec = (totsec % 3600) % 60 #just for reference
    return d, h, m, sec


def plot_latent_space(z, y=None):
    """Creates a joint plot of features, used during training, figures
    are W&B ready

    Parameters
    ----------
    z : numpy array
        fetures to be plotted
    y : list, optional
        axis for color code

    Returns
    -------
    fig
        matplotlib figure
    fig
        image of matplotlib figure
    """
    plt.close('all')
    if z.shape[1] > 8:
        z = z[:,:8]
    df = pd.DataFrame(z)
    if y is not None:
        df.loc[:, 'y'] = y
    pp = sb.pairplot(df,
                     hue='y' if y is not None else None,
                     hue_order=sorted(set(y)) if y is not None else None,
                     diag_kind="hist", markers=".", height=2,
                     plot_kws=dict(s=30, edgecolors='face', alpha=.8),
                     diag_kws=dict(histtype='step'))

    plt.tight_layout()
    pp.fig.canvas.draw()
    return pp.fig


def str2bool(v):
    """Convert strings (y,yes, true, t, 1,n, no,false, f,0) 
    to boolean values

    Parameters
    ----------
    v : numpy array
        string value to be converted to boolean

    Returns
    -------
    bool
        boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def scatter_hue(x, y, labels, disc=True, c_label=''):
    """Creates a wall of light curves plot with real and reconstruction
    sequences, paper-ready.

    Parameters
    ----------
    x      : array
        data to be plotted in horizontal axis
    y      : array
        data to be plotted in vertical axis
    labels : list, optional
        list with corresponding lables to be displayed as legends
    disc : bool, optional
        wheather the axis used for coloring is discrete or not
    c_label    : bool, optional
        name of color dimension

    Returns
    -------
        display figure
    """

    fig = plt.figure(figsize=(12, 9))
    if disc:
        c = cm.Dark2_r(np.linspace(0, 1, len(set(labels))))
        for i, cls in enumerate(set(labels)):
            idx = np.where(labels == cls)[0]
            plt.scatter(x[idx], y[idx], marker='.', s=20,
                        color=c[i], alpha=.7, label=cls)
    else:
        plt.scatter(x, y, marker='.', s=20,
                    c=labels, cmap='coolwarm_r', alpha=.7)
        plt.colorbar(label=c_label)

    plt.xlabel('embedding 1')
    plt.ylabel('embedding 2')
    plt.legend(loc='best', fontsize='x-large')
    plt.show()
