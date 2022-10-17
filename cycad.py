# CYCAD - CYCling Autocorrelation Dataviz
# 1.0.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import h5py
import csv
from charset_normalizer import from_bytes
from pylab import figure, cm
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.spatial.distance import euclidean
from scipy.signal import resample, decimate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from tqdm import tqdm

class cycad:
    '''
    The cycad object contains data and methods pertaining to an *in situ* dataset
    '''

    def __init__(self):
        pass
        
    @staticmethod
    def natural_sort(l): 
        '''
        Sort a list of strings in a human friendly way.

        :param l: list of strings to be sorted
        :type l: list
        :return: sorted list of strings
        :rtype: list
        '''
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    @staticmethod
    def parse_filename(filename):
        '''
        Parse a file name to get the sample name.
        Overwrite for specific data if you want to use the dataframe directly

        :param filename: string containing a file name
        :type filename: str
        :return: string containing the sample name
        :rtype: str
        '''
    # return str(int(filename.split('.')[-2].split('_')[-1]))
        return filename.split('-')[1].split('_')[0]

    def read_folder(self, root, filetype):
        '''
        Read a sorted list of files in a folder to `self.files`. The path is stored as `self.root`. The lowest level folder name is stored as `self.runname` and used as the run name.

        The filetype can be specified currently HDF type (h5, hdf5), or csv type ('csv', 'txt', 'xy', 'xye', 'tsv') are supported. The file type is stored in self.filetype.

        Note: currently for HDF files the data is required to be individual scans with I and 2th keys.
        
        :param root: Path to a folder containing data files.
        :type root: str
        :param filetype: Filetype of the files.
        :type filetype: str
        '''
        self.root = root
        self.runname = os.path.normpath(self.root).split(os.path.sep)[-1]
        self.filetype = filetype
        self.files = self.natural_sort(glob.glob(self.root + '/*' + self.filetype))
        self.read_data()

    def read_data(self, parse_names=False):
        '''
        Read data from a list of files found by read_folder.
        Takes the first column of the first file as the x-values.
        Filetype is specified by :func:`self.read_folder()`.

        The delimited for csv files will be detected automatically.

        :param parse_names: If True, parse file names for column headings
        :type parse_names: bool
        '''
        try:
            maxfiles = len(self.files)

            # Initialize dataframe with x column
            self.df = pd.DataFrame()
            if self.filetype in ['h5', 'hdf5']:
                f = h5py.File(self.files[0])
                def get_data(f):
                    return h5py.File(f)['I'][0]
                self.df['x'] = f['2th']


            elif self.filetype in ['csv', 'txt', 'xy', 'xye', 'tsv']:
                # Find csv delimiter
                sniffer = csv.Sniffer()
                with open(self.files[0], 'r') as sniff_file:
                    delimiter = sniffer.sniff(sniff_file.read(1024)).delimiter

                def get_data(f):
                    return pd.read_csv(f, sep=delimiter, usecols=[0,1], names=['x', 'y'])

                f = get_data(self.files[0])
                self.df['x'] = f['x']

            # Initialise the other columns
            _ = []
            for i in tqdm(range(maxfiles)):

                # HDF type data formats
                if self.filetype in ['h5', 'hdf5']:
                    try:
                        # self.df[split_filename(self.files[i])] = h5py.File(self.files[i])['I'][0]
                        if parse_names:
                            _.append(pd.DataFrame(get_data(self.files[i]), columns=[self.parse_filename(self.files[i])]))
                        else:
                            _.append(pd.DataFrame(get_data(self.files[i]), columns=[self.files[i]]))
                    except:
                        print(f"File {i} is missing data")
                        _.append(pd.DataFrame(np.zeros(self.df.shape[0])+0.1, columns=[self.files[i]]))

                # Delimited data formats
                if self.filetype in ['csv', 'txt', 'xy', 'xye', 'tsv']:
                    try:
                        if parse_names:
                            temp = get_data(self.files[i])
                            temp.rename(columns={'y': self.parse_filename(self.files[i])}, inplace=True)
                            _.append(temp[self.parse_filename(self.files[i])])
                        else:
                            temp = get_data(self.files[i])
                            temp.rename(columns={'y': self.files[i]}, inplace=True)
                            _.append(temp[self.files[i]])
                    except:
                        print(f"File {i} is missing data")
                        _.append(pd.DataFrame(np.zeros(self.df.shape[0])+0.1, columns=[self.files[i]]))

            # Concatenate all dataframes
            self.df = pd.concat([self.df] + _, axis=1)
            
        except:
            print('Error reading data, no data read')

    def read_data_csv(self, path):
        '''
        Read a dataset to `self.df` from a single csv file

        :param path: Path to the csv file
        :type path: str
        '''
        self.df = pd.read_csv(path)
        self.df.rename({self.df.columns[0]: 'x'}, axis=1, inplace=True)
        # Fix this
        self.runname = None

    def read_echem_mpt(self, path, decimal=','):
        '''
        Read cycling data from an echem mpt file
        If this is called multiple times, the data will be concatenated
        The data are resampled to the size of the data dataframe in self.autocorrelate_ec()

        :param path: Path to the mpt file
        :type path: str
        :param decimal: Decimal separator in the mpt file
        :type decimal: str
        '''
        # Speed up encoding detection
        with open(path, 'rb') as f:
            encoding = from_bytes(f.read(2048)).best().encoding

        skip = self.get_skip(path, encoding=encoding)

        _ = pd.read_csv(path, encoding=encoding, sep='\t', decimal=decimal, skiprows=skip)['Ewe/V']

        try:
            self.df_echem = pd.concat((self.df_echem, pd.DataFrame(_).T), axis=1)
        except:
            self.df_echem = pd.DataFrame(_).T

    def read_echem_df(self, df):
        '''
        Read cycling data from a single-column dataframe or series
        If this is called multiple times, the data will be concatenated
        The data are resampled to the size of the data dataframe in self.autocorrelate_ec()

        :param df: Dataframe or series to read
        :type df: pandas.DataFrame or pandas.Series
        '''

        try:
            self.df_echem = pd.concat((self.df_echem, pd.DataFrame(df).T), axis=1)
        except:
            self.df_echem = pd.DataFrame(df).T



        
    @staticmethod
    def baseline_arPLS(y, ratio=1e-6, lam=100, niter=10, full_output=False):
        '''
        Baseline correction routine

        :param y: Data to be baseline corrected
        :type y: numpy.ndarray
        '''
        L = len(y)
        diag = np.ones(L - 2)
        D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
        H = lam * D.dot(D.T)  # The transposes are flipped w.r.t the Algorithm on pg. 252
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        crit = 1
        count = 0
        while crit > ratio:
            z = spsolve(W + H, W * y)
            d = y - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            count += 1
            if count > niter:
                # print('Maximum number of iterations exceeded')
                break

        if full_output:
            info = {'num_iter': count, 'stop_criterion': crit}
            return z, d, info
        else:
            return z

    def bkg_subtract(self, df):
        '''
        Apply baseline correction to a dataframe using the arPLS algorithm

        :param df: Dataframe to be baseline corrected
        :type df: pandas.DataFrame
        :return: Baseline corrected dataframe
        :rtype: pandas.DataFrame
        '''
        # Define cutoffs if necessary
        start, end = None, None

        # Concatenating a list is better than inserting one by one
        _ = []
        for i in tqdm(df.columns[1:]):
            subtracted = df[i][start:end] - self.baseline_arPLS(df[i][start:end])
            _.append(subtracted)
        return pd.concat([df['x']] + _, axis=1)

    @staticmethod
    def get_skip(file, encoding='utf-8'):
        '''
        Get the number of rows to skip in a mpt file

        :param file: Path to a mpt file
        :type file: str
        :param encoding: Encoding of the mpt file
        :type encoding: str
        :return: Number of rows to skip
        :rtype: int
        '''
        skip = 0
        with open(file, 'r', encoding=encoding) as f:
            for line in f:
                if 'Ewe/V' in line:
                    return skip
                else:
                    skip += 1
        return skip
            
    def autocorrelate(self, samples=None, bkg_subtract=False):
        '''
        Apply autocorrelation to self.df to generate correlation matrix.
        Store the correlation matrix in self.correlation_matrix.

        :param samples: Number of samples to use for the downsampled output
        :type samples: int
        :param bkg_subtract: Whether to apply baseline correction to the data
        :type bkg_subtract: bool
        '''

        if bkg_subtract:
            self.df = self.bkg_subtract(self.df)
            self.df[self.df.columns[1:]] = self.df[self.df.columns[1:]] - self.df[self.df.columns[1:]].min().min() + 1e-6
            
        try:
            pairs = list(combinations(self.df.columns[1:], 2))
            array_size = self.df.shape[1]-1
            self.correlation_matrix = np.zeros((array_size, array_size))
            for i in tqdm(pairs):
                if samples:
                    distance = np.corrcoef(
                        resample(self.df[i[0]], samples),
                        resample(self.df[i[1]], samples))[0,1]
                else:
                    distance = np.corrcoef(self.df[i[0]], self.df[i[1]])[0,1]
                self.correlation_matrix[self.df.columns.get_loc(i[0])-1, self.df.columns.get_loc(i[1])-1] = distance
                self.correlation_matrix[self.df.columns.get_loc(i[1])-1, self.df.columns.get_loc(i[0])-1] = distance
            
            # Set the diagonal to one for color consistency
            self.correlation_matrix[np.diag_indices_from(self.correlation_matrix)] = 1

            # Raise baseline to zero
            self.correlation_matrix[self.correlation_matrix < 0] = 0.01
        except:
            print('Could not generate correlation matrix')

    def autocorrelate_ec(self):
        '''
        Generate a distance matrix from the single dimensional array stored in self.df_echem.
        This would usally be the cycling potentials.

        # TODO: add in data file reading

        '''
        try:
            self.df_echem = pd.DataFrame(resample(self.df_echem.T, self.df.shape[1]-1)).T

            pairs = list(combinations(self.df_echem.columns[:], 2))
            array_size = self.df_echem.shape[1]
            self.correlation_matrix_ec = np.zeros((array_size, array_size))
            for i in tqdm(pairs):
                distance = np.linalg.norm(self.df_echem[i[0]] - self.df_echem[i[1]])
                self.correlation_matrix_ec[self.df_echem.columns.get_loc(i[0])-1, self.df_echem.columns.get_loc(i[1])-1] = distance
                self.correlation_matrix_ec[self.df_echem.columns.get_loc(i[1])-1, self.df_echem.columns.get_loc(i[0])-1] = distance
            
            # # Set the diagonal to one for color consistency
            # self.correlation_matrix_ec[np.diag_indices_from(self.correlation_matrix_ec)] = 1

            # # Raise baseline to zero
            # self.correlation_matrix_ec[self.correlation_matrix_ec < 0] = 0.01
        except:
            print('Could not generate distance matrix')

    def plot(
        self, 
        qmin=0.15, 
        qmax=0.9, 
        echem=False, 
        echem_alpha=0.2, 
        echem_quantile=0.2,
        save=False,
        filename=None,
        ):
        '''
        Plot the full correlation matrix and the components

        :param qmin: Minimum quantile to use for the color scale
        :type qmin: float
        :param qmax: Maximum quantile to use for the color scale
        :type qmax: float
        :param echem: Whether to plot the echem data
        :type echem: bool
        :param echem_alpha: Opacity of the echem overlay
        :type echem_alpha: float
        :param echem_quantile: Quantile to use for the echem overlay (how close should the voltages be in the highlighted region)
        :type echem_quantile: float
        :param save: Whether to save the figure
        :type save: bool
        :param filename: Filename to save the figure as
        :type filename: str
        '''
        if self.correlation_matrix is not None:

            # Define plot grid with or without echem
            if echem==False:
                fig = plt.figure(figsize=(10,10), tight_layout=True)
                gs = GridSpec(5, 7, hspace=0.0, wspace=0.0,
                width_ratios=[1, 1, 1, 1, 1, 0.2, 0.2], height_ratios=[1, 1, 1, 1, 1])

                # fig, ((ax_x, ax2), (ax3, ax_y)) = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)
                ax_main = fig.add_subplot(gs[1:4, 0:4])
                ax_y = fig.add_subplot(gs[1:4, 4])
                ax_x = fig.add_subplot(gs[0, 0:4])
                ax_cbar = fig.add_subplot(gs[1:4, 6])
                # ax_ec = fig.add_subplot(gs[4, 0:4])

            # With echem, add extra grid positions and axes ax_ec_x & ..._y
            else:
                fig = plt.figure(figsize=(10, 10), tight_layout=True)
                gs = GridSpec(5, 8, hspace=0.0, wspace=0.0,
                width_ratios=[1, 1, 1, 1, 1, 1, 0.2, 0.2], height_ratios=[1, 1, 1, 1, 1])

                # fig, ((ax_x, ax2), (ax3, ax_y)) = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)
                ax_main = fig.add_subplot(gs[1:4, 1:5])
                ax_y = fig.add_subplot(gs[1:4, 5])
                ax_x = fig.add_subplot(gs[0, 1:5])
                ax_cbar = fig.add_subplot(gs[1:4, 7])
                ax_ec_y = fig.add_subplot(gs[1:4, 0])
                ax_ec_x = fig.add_subplot(gs[4, 1:5])
            
            # set the colormap limites as quantiles of the data
            # default to 0.15 and 0.9 to avoid outliers
            vmin = np.quantile(np.reshape(self.correlation_matrix, -1), qmin)
            vmax = np.quantile(np.reshape(self.correlation_matrix, -1), qmax)

            # Plot main correlation matrix on central axis
            main = ax_main.imshow(self.correlation_matrix, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gray_r', origin='lower', aspect='auto')
            
            # Add colorbar to the right of the main plot
            plt.colorbar(main, cax=ax_cbar)
            
            # Plot echem correlation matrix on top of main axis
            # plot echem 1-D plots on left and bottom axes
            if echem:    
                # Use a contourf with transparency
                # Set cutoff and colourmap manually
                # Colors can be written as RGBAlpha
                cutoff = np.quantile(np.reshape(self.correlation_matrix_ec, -1), echem_quantile)
                ax_main.contourf(self.correlation_matrix_ec, levels=[0, cutoff], colors = [(1,0,0,echem_alpha), (0, 0, 0, 0)], origin='lower')

                ax_ec_x.plot(self.df_echem.T)
                ax_ec_x.set_ylabel('V')
                ax_ec_x.set_xlim(0, len(self.df_echem.columns))

                x = np.arange(self.df_echem.shape[1])
                ax_ec_y.plot(self.df_echem.T, x)
                ax_ec_y.set_xlabel('V')
                ax_ec_y.set_ylim(0, len(self.df_echem.columns))


            # Plot in-situ data on right and bottom axes

            start = 0
            end = self.df.shape[0] - 1
            # aspect = len(self.df.columns[1:])/(end - start)

            # Calculate color scaling for in-situ data
            vmin = self.df[self.df.columns[1]].quantile(qmin)
            vmax = self.df[self.df.columns[1]].quantile(qmax)

            # Plot in-situ data
            ax_y.imshow(self.df.iloc[start:end, 1:].T,
                    extent = [self.df[self.df.columns[0]][start], self.df[self.df.columns[0]][end], 0, self.df.shape[1]],
                    norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower', aspect='auto')

            ax_x.imshow(self.df.iloc[start:end, 1:],
                    extent = [0, self.df.shape[1], self.df[self.df.columns[0]][start], self.df[self.df.columns[0]][end]],
                    norm=LogNorm(vmin=vmin, vmax=vmax), aspect='auto')

            # Set ticks and labels
            ax_main.set_xlabel('Pattern number')
            ax_main.set_ylabel('Pattern number')
            ax_x.set_xticks([])
            ax_x.set_ylabel('x')
            ax_y.set_yticks([])
            ax_y.set_xlabel('x')

            try:
                fig.suptitle(self.runname)
            except:
                pass

            if save:
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                
                if filename == None:
                    filename = self.runname

                fig.savefig(r'./plots/' + filename, transparent=False, facecolor='white')

            plt.show()
        
            plt.close(fig)
        else:
            print('No data to plot!')
    