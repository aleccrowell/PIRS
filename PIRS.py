import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import *
import math
from sklearn import linear_model
import scipy
from scipy import stats
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import random
from tqdm import tqdm
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource

class ranker:
    def __init__(self, norm_filename, circ_filename, abs_filename, ex_g_size_fname, tgene):
        self.data = pd.read_csv(norm_filename,sep='\t',header=0,index_col=0)
        self.circ = pd.read_csv(circ_filename,sep='\t',header=0,index_col=0)
        self.abs = pd.read_csv(abs_filename,sep='\t',header=0,index_col=0)
        self.egs = pd.read_csv(ex_g_size_fname,sep='\t',header=0,index_col=0).T
        self.tgene = tgene
        self.notdone = True

    def remove_circ(self):
        to_remove = self.circ[self.circ['GammaP']<.05].index.values.tolist()
        self.data = self.data[~self.data.index.isin(to_remove)]

    def get_tpoints(self):
        tpoints = [i.replace('CT','') for i in self.data.columns.values]
        if any(['_' in i for i in tpoints]):
            tpoints = [i.split('_')[0] for i in tpoints]
        if any(['.' in i for i in tpoints]):
            tpoints = [i.split('.')[0] for i in tpoints]
        self.tpoints = np.asarray(tpoints).astype(int)

    def get_abs_exp(self):
        self.abs = self.abs.mean(axis=1)
        temp = pd.concat([self.abs, self.egs], axis=1, join='inner')
        self.abs = temp[temp.columns[0]]/temp[temp.columns[1]]
        self.abs = self.abs/(self.abs.sum())
        self.abs = self.abs*(10**6)
        self.data = self.data[~self.data.index.isin(self.abs[self.abs<10].index.values)]
        self.data = self.data[self.data.index.isin(self.abs.index.values)]

    def remove_anova(self,length=24):
        to_remove = []
        for index, row in tqdm(self.data.iterrows()):
            vals = []
            for i in list(set(self.tpoints)):
                vals.append([row.values[j] for j in range(len(row)) if self.tpoints[j] == i])
            f_val, p_val = stats.f_oneway(*vals)
            if p_val < 0.05:
                to_remove.append(index)
        self.data = self.data[~self.data.index.isin(to_remove)]

    def rank_on_error(self,dof=24,alpha=0.05):
        es = {}
        for index, row in tqdm(self.data.iterrows()):
            regr = linear_model.LinearRegression()
            regr.fit(np.array(self.tpoints)[:,np.newaxis], np.array(row))
            rsq = np.sum((regr.predict(np.array(self.tpoints)[:,np.newaxis]) - np.array(row)) ** 2)
            regr_error = math.sqrt(rsq/(dof-2))
            xsq = np.sum((np.array(self.tpoints) - np.mean(np.array(self.tpoints))) ** 2)
            pred = []
            for x in np.array(self.tpoints):
                pred.append(regr.predict(x) + scipy.stats.t.ppf(1.0-alpha/2.,dof)*regr_error*math.sqrt(1+1/dof+(((x-np.mean(self.tpoints))**2)/xsq)))
                pred.append(regr.predict(x) - scipy.stats.t.ppf(1.0-alpha/2.,dof)*regr_error*math.sqrt(1+1/dof+(((x-np.mean(self.tpoints))**2)/xsq)))
            error = np.sum([(i - np.mean(np.array(row)))**2 for i in pred])/(self.abs.loc[index]**2)
            es[index] = np.mean(error)
        self.errors = pd.DataFrame.from_dict(es,orient='index')
        self.errors.columns = ['errors']
        self.errors.sort_values('errors',inplace=True)

    def get_exp_diff(self):
        self.exp_diff = np.abs(self.abs - self.abs.loc[self.tgene])

    def plot_diff_v_error(self):
        temp = pd.concat([np.log(self.exp_diff), np.log(self.errors)], axis=1, join='inner')
        temp.columns = ['exp_diff','errors']
        temp = temp.replace([np.inf, -np.inf], np.nan)
        temp = temp.dropna()
        source = ColumnDataSource(data=dict(
            exp_diff=temp['exp_diff'].values,
            error=temp['errors'].values,
            transcript=temp.index.values
        ))
        p = figure(title="PIRS Results "+self.tgene,tools="hover",x_axis_label="Expressin Difference (ln(count))",y_axis_label="ln(Prediction Interval Ranking Score)")
        p.circle('exp_diff', 'error', size=6, color="navy", alpha=0.5, source=source)
        p.select_one(HoverTool).tooltips = [
            ('transcript', '@transcript'),
            ('ln(exp_diff)', '@exp_diff'),
            ('ln(PIRS)', '@error'),
        ]
        output_file(self.tgene+".html", title=self.tgene)
        show(p)

    def output(self,fname):
        self.errors.to_csv(fname,sep='\t')
