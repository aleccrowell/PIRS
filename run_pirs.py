from PIRS import ranker
import os
import sys
import getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hn:p:r:s:t:",["nexp=","circp=","rexp=","genesize=","tgene="])
    except getopt.GetoptError:
        print('knn_impute.py -n <normedexpression> -p <circadianpvalues> -r <rawexpression> -s <genesizes> -t <targetgene>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('knn_impute.py -n <normedexpression> -p <circadianpvalues> -r <rawexpression> -s <genesizes> -t <targetgene>')
            sys.exit()
        elif opt in ("-n", "--nexp"):
            nfile = arg
        elif opt in ("-p", "--circp"):
            pfile = arg
        elif opt in ("-r", "--rexp"):
            rfile = arg
        elif opt in ("-s", "--genesize"):
            sfile = arg
        elif opt in ("-t", "--tgene"):
            tgene = arg

    print('Importing Data')
    to_rank = ranker(nfile,pfile,rfile,sfile,tgene)
    print('Removing Circadian Genes')
    to_rank.remove_circ()
    to_rank.get_tpoints()
    print('Calculating TPMs')
    to_rank.get_abs_exp()
    print('Removing ANOVA Differentially Expressed Genes')
    to_rank.remove_anova()
    print('Calculating PIRS')
    to_rank.rank_on_error()
    print('Calculating Expression Differences')
    to_rank.get_exp_diff()
    print('Plotting')
    to_rank.plot_diff_v_error()

if __name__ == '__main__':
    main(sys.argv[1:])
