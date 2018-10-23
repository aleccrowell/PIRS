PIRS: Prediction Interval Ranking Score
=======================================

PIRS provides a means for identifying constitutive expression from time series data.  The software was designed for transcriptomic or proteomic data, but can be applied to any quantitative time series data. There is only one class 
rank, which performs the ranking.

----------
Motivation
----------

The issue of identifying increasing, decreasing or even cyclical patterns in time series data is well studied.  Little effort has been devoted to screening for stable expression.  Identifying constitutive expression is especially 
important when selecting reference genes which are later assumed to be stably expressed for purposes such as qPCR.  In the past many 'constitutively expresse' reference genes have later been identified to have circadian or other 
dynamic expression patterns.  PIRS provides for the systematic screening of expression profiles from high throughput time series to identify those which are truly constitutively expressed.

--------
Features
--------

* Prescreening of profiles for differential expression using ANOVA

* Ranking of peptides based on linear regression prediction intervals

-------------
Example Usage
-------------

```python
from PIRS import rank

data = rank.ranker(path_to_data)
sorted_data = data.pirs_sort()
```
### A Note on Data Formatting
PIRS expects input files to be formatted as tab seperated.  The first column should indicate the transcript or protein identifier.  The header should start with '#' and the rest of the header should be of the form 02_1 for data with
the first number indicating the timepoint and the second the replicate.  It is important that single digit timepoints include the leading zero for 
formatting. Missing values should bbe indicated by the string 'NULL'.  Example data file:

| Peptide | Protein | 00_1 | 00_2 | 00_3 | 02_1 | 02_2 | 02_3 |
|---|---|---|---|---|---|---|---|
| Peptide_ID | Protein_ID | data | data | data | data | data | data |

------------
Installation
------------

pip install pirs

-------------
API Reference
-------------

http://pirs.readthedocs.io/en/latest/source/modules.html
