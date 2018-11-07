import numpy as np
import pandas as pd
import pickle
import string
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

##Generate ROC Curves for simulations and compare to RSD

class simulate:
    """
    Generates a properly formulated simulated circadian dataset for testing.


    This class takes user specifications on dataset size along with the number and frequency of batch effects and levels of background noise and outputs a simulated dataset along with a key showing which rows represent truly circadian data.


    Parameters
    ----------
    tpoints : int
        
    nrows : int

    nreps : int

    tpoint_space : int
        
    pcirc : float

    phase_prop : float

    phase_noise : float

    amp_noise : float

    n_batch_effects : int

    pbatch : float

    Probability of each batch effects appearance in a given peptide.

    effect_size : float

    Average size of batch effect


    Attributes
    ----------

    simdf : dataframe

    Simulated data without noise.

    simndf : dataframe

    Simulated data with noise.

    """

    def __init__(self, tpoints=24, nrows=1000, nreps=3, tpoint_space=2, pcirc=.5, phase_prop=.5, phase_noise=.25, amp_noise=.75, n_batch_effects=3, pbatch=.5, effect_size=2, rseed=4574):
        """
        Simulates circadian data and saves as a properly formatted example .csv file.

        Takes a file from one of two data types protein ('p') which has two index columns or rna ('r') which has only one.  Opens a pickled file matching pooled controls to corresponding samples if data_type = 'p' and opens a picked file matching samples to blocks if designtype = 'b'.

        """

        np.random.seed(rseed)
        self.tpoints = int(tpoints)
        self.nreps = int(nreps)
        self.nrows = int(nrows)
        self.tpoint_space = int(tpoint_space)
        self.pcirc = float(pcirc)
        self.phase_prop = float(phase_prop)
        self.phase_noise = float(phase_noise)
        self.amp_noise = float(amp_noise)
        self.n_batch_effects = int(n_batch_effects)
        self.pbatch = float(pbatch)
        self.effect_size = float(effect_size)

        #procedurally generate column names
        self.cols = []
        for i in range(self.tpoints):
            for j in range(self.nreps):
                self.cols.append(
                    ("{0:0=2d}".format(self.tpoint_space*i+self.tpoint_space))+'_'+str(j+1))

        #randomly determine which rows are circadian
        self.circ = np.random.binomial(1, self.pcirc, self.nrows)
        #generate a base waveform
        base = np.arange(0, (4*np.pi), (4*np.pi/self.tpoints))
        #simulate data
        self.sim = []
        phases = []
        for i in self.circ:
            if i == 1:
                temp = []
                p = np.random.binomial(1, self.phase_prop)
                phases.append(p)
                for j in range(self.nreps):
                    temp.append(np.sin(base+np.random.normal(0, self.phase_noise, 1) +
                                       np.pi*p)+np.random.normal(0, self.amp_noise, self.tpoints))
                self.sim.append([item for sublist in zip(*temp)
                                 for item in sublist])
            else:
                phases.append('nan')
                self.sim.append(np.random.normal(
                    0, self.amp_noise, (self.tpoints*self.nreps)))
        #add in batch effects
        batch_effects = []
        for i in range(self.n_batch_effects):
            batch_effects.append(np.random.normal(
                0., self.effect_size, (self.tpoints*self.nreps)))
        self.simnoise = []
        for i in self.sim:
            temp = i
            bts = np.random.binomial(1, self.pbatch, self.n_batch_effects)
            for j in range(self.n_batch_effects):
                temp += bts[j]*batch_effects[j]
            self.simnoise.append(temp)

    def write_output(self, out_name='simulated_data'):
        """

        out_name : str

        output file stem

        """

        self.out_name = str(out_name)
        self.simndf = pd.DataFrame(self.simnoise, columns=self.cols)
        self.simndf.index.names = ['#']
        self.simndf.to_csv(out_name+'_with_noise.txt', sep='\t')

        self.simdf = pd.DataFrame(np.asarray(self.sim), columns=self.cols)
        self.simdf.index.names = ['#']
        self.simdf.to_csv(out_name+'_baseline.txt', sep='\t')

        pd.DataFrame(self.circ, columns=['Circadian'], index=self.simndf.index).to_csv(out_name+'_true_classes.txt', sep='\t')


class analyze:
    def __init__(self, filename_classes):
        self.true_classes = pd.read_csv(filename_classes, sep='\t')
        self.tags = {}
        self.merged = []
        self.i = 0

    def add_data(self, filename_ejtk, tag):
        self.tags[tag] = self.i
        ejtk = pd.read_csv(filename_ejtk, sep='\t')
        self.merged.append(pd.merge(self.true_classes[['#', 'Circadian']], ejtk[['ID', 'GammaBH']], left_on='#', right_on='ID', how='left'))
        self.i += 1

    def generate_roc_curve(self):
        for j in self.tags.keys():
            fpr, tpr, thresholds = roc_curve(self.merged[self.tags[j]]['Circadian'].values, (
                1-self.merged[self.tags[j]]['GammaBH'].values), pos_label=1)
            plt.plot(fpr, tpr, label=j)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Comparison')
        plt.legend(loc="lower right")
        plt.savefig('ROC.pdf')

    def calculate_auc(self):
        out = {}
        for j in self.tags.keys():
            fpr, tpr, thresholds = roc_curve(self.merged[self.tags[j]]['Circadian'].values, (
                1-self.merged[self.tags[j]]['GammaBH'].values), pos_label=1)
            out[j] = auc(fpr, tpr)
        self.roc_auc = out
