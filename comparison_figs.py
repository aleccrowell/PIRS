from PIRS import simulations, rank
import pandas as pd
from sklearn.metrics import precision_recall_curve

for i in range(20):
    simulation = simulations.simulate(rseed=i, pcirc=.8, plin=0, amp_noise=.5)
    simulation.write_output("sim_"+str(i)+"_circ.txt")
    simulation.write_genorm("sim_"+str(i)+"_gn_circ.txt")
    data = rank.ranker("sim_"+str(i)+"_circ.txt")
    sorted_data = data.pirs_sort("pirs_"+str(i)+"_circ.txt")
    old_data = rank.rsd_ranker("sim_"+str(i)+"_circ.txt")
    old_sorted_data = old_data.rsd_sort("rsd_"+str(i)+"_circ.txt")


for i in range(20):
    simulation = simulations.simulate(rseed=i, pcirc=0, plin=.8, amp_noise=1.5)
    simulation.write_output("sim_"+str(i)+"_lin.txt")
    simulation.write_genorm("sim_"+str(i)+"_gn_lin.txt")
    data = rank.ranker("sim_"+str(i)+"_lin.txt")
    sorted_data = data.pirs_sort("pirs_"+str(i)+"_lin.txt")
    old_data = rank.rsd_ranker("sim_"+str(i)+"_lin.txt")
    old_sorted_data = old_data.rsd_sort("rsd_"+str(i)+"_lin.txt")

import matplotlib.pyplot as plt
import numpy as np
temp = pd.merge(analysis.true_classes, analysis.merged[analysis.merged.method == 'NormFinder'], on=['#', 'rep'])
plt.scatter(temp.Const, temp.score)
plt.savefig('test.pdf')
plt.close()

#parse R output files
for i in range(20):
    nf = pd.read_csv('sim_'+str(i)+'_nf_circ_proc.txt')
    nf.columns = ['#','score']
    #nf.columns = ['score', '#']
    nf['#'] = nf['#'].str.split('gene_').str[1]
    #nf['score'] = nf['score'].str.split('gene_').str[1].astype(int)
    #nf['score'] = (1-nf['score']/nf['score'].max()) + .0001
    nf['score'] = 1-nf['score']
    nf.to_csv('nf_'+str(i)+'_circ.txt', sep='\t', index=False)
    gn = pd.read_csv('sim_'+str(i)+'_gn_circ_proc.txt')
    gn.columns = ['#','score']
    gn['score'] = gn['score'].str.split('gene_').str[1].astype(int)
    #gn.columns = ['score', '#']
    #gn['#'] = gn['#'].str.split('gene_').str[1]
    gn['score'] = (1-gn['score']/gn['score'].max()) + .0001
    gn[['#', 'score']].to_csv('gn_'+str(i)+'_circ.txt', sep='\t', index=False)

for i in range(20):
    nf = pd.read_csv('sim_'+str(i)+'_nf_lin_proc.txt')
    nf.columns = ['#','score']
    #nf.columns = ['score', '#']
    nf['#'] = nf['#'].str.split('gene_').str[1]
    #nf['score'] = nf['score'].str.split('gene_').str[1].astype(int)
    #nf['score'] = (1-nf['score']/nf['score'].max()) + .0001
    nf['score'] = 1-nf['score']
    nf.to_csv('nf_'+str(i)+'_lin.txt', sep='\t', index=False)
    gn = pd.read_csv('sim_'+str(i)+'_gn_lin_proc.txt')
    gn.columns = ['#', 'score']
    gn['score'] = gn['score'].str.split('gene_').str[1].astype(int)
    #gn.columns = ['score', '#']
    #gn['#'] = gn['#'].str.split('gene_').str[1]
    gn['score'] = (1-gn['score']/gn['score'].max()) + .0001
    gn[['#', 'score']].to_csv('gn_'+str(i)+'_lin.txt', sep='\t', index=False)

analysis = simulations.analyze()
for i in range(20):
    analysis.add_classes("sim_"+str(i)+"_circ_true_classes.txt", rep=i)
    analysis.add_data("pirs_"+str(i)+"_circ.txt", 'PIRS', rep=i)
    analysis.add_data("rsd_"+str(i)+"_circ.txt", 'SD/RSD', rep=i)
    analysis.add_data("gn_"+str(i)+"_circ.txt", 'GeNorm', rep=i)
    analysis.add_data("nf_"+str(i)+"_circ.txt", 'NormFinder', rep=i)


analysis.generate_pr_curve()


analysis.curves = pd.DataFrame(columns=['precision', 'recall', 'method'])
#colors = ["light grey", "black"]
colors = ["windows blue", "amber", "light grey", "black"]
for rep in analysis.merged.rep.unique():
    for method in analysis.merged.method.unique():
        pr = pd.merge(analysis.true_classes[analysis.true_classes['rep'] == rep].reset_index(),
                      analysis.merged[(analysis.merged.rep == rep) & (analysis.merged.method == method)].reset_index(), on=['#', 'rep'])
        precision, recall, _ = precision_recall_curve(
            pr['Const'].values, 1/pr['score'].values, pos_label=1)
        temp = pd.DataFrame()
        temp['precision'] = precision
        temp['recall'] = recall
        temp['method'] = method
        temp['rep'] = rep
        analysis.curves = pd.concat([analysis.curves, temp])

analysis = simulations.analyze()
for i in range(20):
    analysis.add_classes("sim_"+str(i)+"_lin_true_classes.txt", rep=i)
    analysis.add_data("pirs_"+str(i)+"_lin.txt", 'PIRS', rep=i)
    analysis.add_data("rsd_"+str(i)+"_lin.txt", 'SD/RSD', rep=i)
    analysis.add_data("gn_"+str(i)+"_lin.txt", 'GeNorm', rep=i)
    analysis.add_data("nf_"+str(i)+"_lin.txt", 'NormFinder', rep=i)


analysis.generate_pr_curve()


analysis.curves = pd.DataFrame(columns=['precision', 'recall', 'method'])
#colors = ["light grey", "black"]
colors = ["windows blue", "amber", "light grey", "black"]
for rep in analysis.merged.rep.unique():
    for method in analysis.merged.method.unique():
        pr = pd.merge( analysis.true_classes[ analysis.true_classes['rep'] == rep].reset_index(),
        analysis.merged[(analysis.merged.rep == rep) & (analysis.merged.method==method)].reset_index(), on=['#','rep'])
        precision, recall, _ = precision_recall_curve(pr['Const'].values, 1/pr['score'].values, pos_label=1)
        temp = pd.DataFrame()
        temp['precision'] = precision
        temp['recall'] = recall
        temp['method'] = method
        temp['rep'] = rep
        analysis.curves = pd.concat([ analysis.curves, temp])

import matplotlib.pyplot as plt
import numpy as np
plt.scatter([np.repeat(i,3) for i in np.arange(24)], simulation.sim[3])
plt.savefig('test.pdf')
plt.close()
