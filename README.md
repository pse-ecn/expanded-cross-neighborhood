# Expanded Cross Neighborhood distance based Re-ranking (ECN)

In this repository, we provide the reranking and evaluation code used for our paper **A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking**. 

This includes our Expanded Cross Neighborhood Re-Ranking Matlab and Python code. The training code for our [Pose Sensitive Embedding Model](https://github.com/pse-ecn/pose-sensitive-embedding) is located in a separate repository.

If you find our work helpful in your research, please cite:

``` 
M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen,
"A Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross Neighborhood Re-Ranking", 
https://arxiv.org/abs/1711.10378 2017. [accepted at CVPR 2018]
``` 



### Usage of Expanded Cross Neighborhood Re-ranking

Our matlab implementation of ECN can be found in the file [ECN_rerank](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/ECN_rerank.m) and Python implementation can be found in [ecn.py](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/ecn.py). 
The script can be called with default parameters in the following way to calculate the re-ranked distance for a query and a test set.

### Python Usage:
Typically,
```
from ecn import ECN
ECN_dist = ECN(query, test)
```
or from command line
```
python3 ecn.py --queryset=path-to-your-query-features-csv-file --testset=path-to-your-test-features-csv-file --outputpath=path-to-write-ECN-distance
```

### Matlab usage
```
ECN_dist= ECN_rerank(queryset, testset);
```

This will run ECN with the default parameters `k=25`, `t=3`, `q=8` and `method=rankdist`. The returned `ECN_dist` can then be used to calculate the scores.

#### ECN Parameters
Alternatively, you can also supply different values for these parameters by running it like

```
ECN_dist= ECN_rerank(queryset, testset, 'k',25, 't',3, 'q',8, 'method','rankdist') (Matlab or python)

python3 ecn.py --queryset=path-to-your-query-features-csv-file --testset=path-to-your-test-features-csv-file --outputpath=path-to-write-ECN-distance --method='rankdist' --k=25 --t=3 --q=8  (Python)
```

Supported values for parameter `method` are:
* `rankdist` (default): Using the rank distance for ECN re-ranking
* `origdist`: Using the original euclidian distance for ECN re-ranking


#### Query and Test Set Format

Our ECN_rerank script expects the query and testsets in the following format:

* queryset: probe matrix (#_of_probes x feature_dimension);  feature vectors in rows
* testset = gallery matrix (#_of_gallery_images x feature_dimension); feature vectors in rows

### Evaluation

You can download our PSE model's features for the Market-1501 and Duke datasets here

https://drive.google.com/drive/folders/18eZlJIB5_WZ0tAa_SLbMJqxQS3C_76wh?usp=sharing

You can then run the evaluation by setting the evalPath to your download features folder in the function [evalECNRerankingForPath](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/evalECNRerankingForPath.m).

If you would like to test our ECN reranking on your own features , you can directly use the [evalECNRerankingForFeatures](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/evalECNRerankingForFeatures.m) or directly the main [ECN_rerank](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/ECN_rerank.m). 



