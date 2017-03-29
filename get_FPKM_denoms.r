source("https://bioconductor.org/biocLite.R")
biocLite("GenomicFeatures")
library(GenomicFeatures)
# First, import the GTF-file that you have also used as input for htseq-count
txdb <- makeTxDbFromGFF("GCA_000182925.2_NC12_genomic.gff",format="gff")
# then collect the exons per gene id
exons.list.per.gene <- exonsBy(txdb,by="gene")
# then for each gene, reduce all the exons to a set of non overlapping exons, calculate their lengths (widths) and sum then
exonic.gene.sizes <- lapply(exons.list.per.gene,function(x){sum(width(reduce(x)))})
write.csv(exonic.gene.sizes, file = "exonic_gene_sizes.csv")

columns(txdb)
keytypes(txdb)
select(txdb, keys = keys, columns="TXSTART", keytype="GENEID")
