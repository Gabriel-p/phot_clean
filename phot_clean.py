
import os
from os.path import exists
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from astropy.table import Table
from functools import reduce


def main():
    """
    Filter photometry for color ranges given. Also remove stars with all
    'nan' values in their photometry.
    """

    # Generate output dir if it doesn't exist.
    if not exists('out'):
        makedirs('out')

    col_names, V_min, V_max, BV_min, BV_max, UB_min, UB_max, VI_min, VI_max =\
        params_input()

    for f_id in os.listdir('in/'):
        if f_id == 'dont_read':
            continue
        print('\n' + f_id)

        # Load final photometry file.
        f_id, phot = photLoad(f_id, col_names)
        print("Stars in file: {}".format(len(phot)))

        V_min = np.nanmin(phot['V']) if V_min == 'min' else float(V_min)
        V_max = np.nanmax(phot['V']) if V_max == 'max' else float(V_max)

        plotCMDs(
            f_id, phot, 'all', V_min, V_max, BV_min, BV_max, UB_min, UB_max,
            VI_min, VI_max)

        phot, phot_rjct = filterPhot(
            phot, V_min, V_max, BV_min, BV_max, UB_min, UB_max, VI_min, VI_max)
        print("Stars in cleaned file: {}".format(len(phot)))

        plotCMDs(
            f_id, phot_rjct, 'rjct', V_min, V_max, BV_min, BV_max, UB_min,
            UB_max, VI_min, VI_max)
        plotCMDs(
            f_id, phot, 'accpt', V_min, V_max, BV_min, BV_max, UB_min, UB_max,
            VI_min, VI_max)
        print("Plots created")

        fileClean(f_id, phot)


def params_input():
    """
    Read input parameters from 'params_input.dat' file.
    """
    with open('params_input.dat', "r") as f_dat:
        # Iterate through each line in the file.
        for line in f_dat:
            if not line.startswith("#") and line.strip() != '':
                reader = line.split()
                if reader[0] == 'CN':
                    col_names = reader[1:]
                if reader[0] == 'VV':
                    V_min, V_max = reader[1:]
                if reader[0] == 'BV':
                    BV_min, BV_max = list(map(float, reader[1:]))
                if reader[0] == 'VI':
                    VI_min, VI_max = list(map(float, reader[1:]))
                if reader[0] == 'UB':
                    UB_min, UB_max = list(map(float, reader[1:]))

    return col_names, V_min, V_max, BV_min, BV_max, UB_min, UB_max, VI_min,\
        VI_max


def photLoad(f_id, col_names):
    """
    """
    phot = ascii.read('in/' + f_id, fill_values=('INDEF', np.nan))
    phot = Table(phot, names=col_names)

    return f_id, phot


def filterPhot(
    phot, V_min, V_max, BV_min, BV_max, UB_min, UB_max, VI_min,
        VI_max):
    """
    """
    try:
        # Remove stars with 'nan' values in *all* colors and magnitude.
        N = len(phot)
        nan_msk = [
            ~phot['V'].mask, ~phot['BV'].mask, ~phot['VI'].mask,
            ~phot['UB'].mask]
        total_mask = reduce(np.logical_or, nan_msk)
        phot = phot[total_mask]
        print("Rejected stars with all nan values: {}".format(N - len(phot)))
    except AttributeError:
        print("No stars with all nan values, all stars kept: {}".format(N))

    # Filter colors by range, accepting 'nan' values.
    m1 = np.logical_or(phot['BV'] < BV_max, np.isnan(phot['BV']))
    m2 = np.logical_or(phot['BV'] > BV_min, np.isnan(phot['BV']))
    m3 = np.logical_or(phot['VI'] < VI_max, np.isnan(phot['VI']))
    m4 = np.logical_or(phot['VI'] > VI_min, np.isnan(phot['VI']))
    m5 = np.logical_or(phot['UB'] < UB_max, np.isnan(phot['UB']))
    m6 = np.logical_or(phot['UB'] > UB_min, np.isnan(phot['UB']))
    m7 = np.logical_or(phot['V'] < V_max, np.isnan(phot['V']))
    m8 = np.logical_or(phot['V'] > V_min, np.isnan(phot['V']))

    mask = [m1, m2, m3, m4, m5, m6, m7, m8]
    total_mask = reduce(np.logical_and, mask)

    # Save stars outside color ranges.
    phot_rjct = phot[~total_mask]
    print("Rejected stars outside ranges: {}".format(len(phot_rjct)))
    # Save stars within color ranges.
    phot = phot[total_mask]

    return phot, phot_rjct


def plotCMDs(
    f_id, phot, acpt_rjct_ID, V_min, V_max, BV_min, BV_max, UB_min, UB_max,
        VI_min, VI_max):
    """
    Plot photometry diagrams.
    """
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 2)

    fig.add_subplot(gs[0])
    plt.title("N={}".format(np.count_nonzero(~np.isnan(phot['BV']))))
    plt.xlabel("(B-V)")
    plt.ylabel("V")
    plt.scatter(phot['BV'], phot['V'], s=5)
    plt.axhline(y=V_max, c='r')
    plt.axhline(y=V_min, c='r')
    plt.axvline(x=BV_max, c='r')
    plt.axvline(x=BV_min, c='r')
    plt.gca().invert_yaxis()

    fig.add_subplot(gs[1])
    plt.title("N={}".format(np.count_nonzero(~np.isnan(phot['VI']))))
    plt.xlabel("(V-I)")
    plt.ylabel("V")
    plt.scatter(phot['VI'], phot['V'], s=5)
    plt.axhline(y=V_max, c='r')
    plt.axhline(y=V_min, c='r')
    plt.axvline(x=VI_max, c='r')
    plt.axvline(x=VI_min, c='r')
    plt.gca().invert_yaxis()

    fig.add_subplot(gs[2])
    plt.title("N={}".format(np.count_nonzero(~np.isnan(phot['UB']))))
    plt.xlabel("(U-B)")
    plt.ylabel("V")
    plt.scatter(phot['UB'], phot['V'], s=5)
    plt.axhline(y=V_max, c='r')
    plt.axhline(y=V_min, c='r')
    plt.axvline(x=UB_max, c='r')
    plt.axvline(x=UB_min, c='r')
    plt.gca().invert_yaxis()

    fig.add_subplot(gs[3])
    plt.title("N={}".format(np.count_nonzero(~np.isnan(phot['UB']))))
    plt.xlabel("(B-V)")
    plt.ylabel("(U-B)")
    plt.scatter(phot['BV'], phot['UB'], s=5)
    plt.axhline(y=UB_max, c='r')
    plt.axhline(y=UB_min, c='r')
    plt.axvline(x=BV_max, c='r')
    plt.axvline(x=BV_min, c='r')
    plt.gca().invert_yaxis()

    fig.tight_layout()
    plt.savefig(
        'out/' + f_id.split('.')[0] + '_' + acpt_rjct_ID + '.png', dpi=300,
        bbox_inches='tight')


def fileClean(f_id, phot):
    """
    Create clean photometry file.
    """
    ascii.write(
        phot, 'out/' + f_id, format='csv', fill_values=[(ascii.masked, 'nan')],
        overwrite=True, comment=False)


if __name__ == '__main__':
    main()
