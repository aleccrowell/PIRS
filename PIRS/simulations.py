import numpy as np
import pandas as pd
import pickle
import string
from sklearn.preprocessing import scale
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import random

##Simulating single expression level, would have to equalize distribution of mean expressins between classes if simulating multiple expression levels

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
        
    pconst : float

    phase_prop : float

    phase_noise : float

    amp_noise : float


    Attributes
    ----------

    simdf : dataframe

    Simulated data without noise.

    simndf : dataframe

    Simulated data with noise.

    """

    def __init__(self, tpoints=24, nrows=1000, nreps=3, tpoint_space=2, pconst=.2, phase_prop=.5, phase_noise=.25, amp_noise=.75,  rseed=4574):
        """
        Simulates circadian data and saves as a properly formatted example .csv file.

        Takes a file from one of two data types protein ('p') which has two index columns or rna ('r') which has only one.  Opens a pickled file matching pooled controls to corresponding samples if data_type = 'p' and opens a picked file matching samples to blocks if designtype = 'b'.

        """

        np.random.seed(rseed)
        self.tpoints = int(tpoints)
        self.nreps = int(nreps)
        self.nrows = int(nrows)
        self.tpoint_space = int(tpoint_space)
        self.pconst = float(pconst)
        self.phase_prop = float(phase_prop)
        self.phase_noise = float(phase_noise)
        self.amp_noise = float(amp_noise)

        #procedurally generate column names
        self.cols = []
        for i in range(self.tpoints):
            for j in range(self.nreps):
                self.cols.append(
                    ("{0:0=2d}".format(self.tpoint_space*i+self.tpoint_space))+'_'+str(j+1))

        #randomly determine which rows are circadian
        self.const = np.random.binomial(1, self.pconst, self.nrows)
        #generate a base waveform
        base = np.arange(0, (4*np.pi), (4*np.pi/self.tpoints))
        #simulate data
        self.sim = []
        phases = []
        trend_types = np.random.binomial(1, 0.5, len(np.where(~self.const)[0]))
        for ind, val in enumerate(self.const):
            if val == 0:
                if trend_types[ind] == 1:
                    temp = []
                    p = np.random.binomial(1, self.phase_prop)
                    phases.append(p)
                    for j in range(self.nreps):
                        temp.append(np.sin(base+np.random.normal(0, self.phase_noise, 1) + np.pi*p)+np.random.normal(0, self.amp_noise, self.tpoints))
                    self.sim.append([item for sublist in zip(*temp) for item in sublist])
                else:
                    phases.append('nan')
                    temp = []
                    for j in range(self.nreps):
                        temp.append((1/(self.tpoints*self.tpoint_space))*np.arange(0, (self.tpoints*self.tpoint_space),
                                              self.tpoint_space)+np.random.normal(0, self.amp_noise, self.tpoints)-.5)
                    self.sim.append([item for sublist in zip(*temp) for item in sublist])
            else:
                phases.append('nan')
                self.sim.append(np.random.normal(
                    0, self.amp_noise, (self.tpoints*self.nreps)))
        
        self.sim = np.array(self.sim)

    def write_output(self, out_name='simulated_data'):
        """

        out_name : str

        output file stem

        """

        self.out_name = str(out_name)
        self.simndf = pd.DataFrame(self.sim, columns=self.cols)
        self.simndf.index.names = ['#']
        self.simndf.to_csv(out_name+'_with_noise.txt', sep='\t')

        pd.DataFrame(self.const, columns=['Const'], index=self.simndf.index).to_csv(out_name+'_true_classes.txt', sep='\t')


class analyze:
    def __init__(self, filename_classes):
        self.true_classes = pd.read_csv(filename_classes, sep='\t')
        self.tags = {}
        self.merged = []
        self.i = 0

    def add_data(self, filename_pirs, tag):
        self.tags[tag] = self.i
        pirs = pd.read_csv(filename_pirs, sep='\t')
        self.merged.append(pd.merge(self.true_classes[['#', 'Const']], pirs[['#', 'score']], left_on='#', right_on='#', how='left'))
        self.merged[-1]['score'].fillna(self.merged[-1]['score'].max(), inplace=True)
        self.i += 1

    def generate_roc_curve(self):
        for j in self.tags.keys():
            precision, recall, _ = precision_recall_curve(self.merged[self.tags[j]]['Const'].values, 1/self.merged[self.tags[j]]['score'].values, pos_label=1)
            plt.plot(recall, precision, label=j)
        plt.plot([0, 1], [np.mean(self.true_classes['Const']), np.mean(
            self.true_classes['Const'])], color='r', linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Comparison')
        plt.legend(loc="lower right")
        plt.savefig('PR.pdf')
        plt.close()

    def calculate_auc(self):
        out = {}
        for j in self.tags.keys():
            fpr, tpr, thresholds = roc_curve(self.merged[self.tags[j]]['Const'].values, (self.merged[self.tags[j]]['score'].values), pos_label=1)
            out[j] = auc(fpr, tpr)
        self.roc_auc = out
