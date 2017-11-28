# Expanded Cross Neighborhood distance based Re-ranking (ECN)

In this repository, we provide the reranking code used for our paper **A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking**. 

This includes our Expanded Cross Neighborhood Re-Ranking Matlab code. The training code for our [Pose Sensitive Embedding Model](https://github.com/pse-ecn/pose-sensitive-embedding) is located in a separate repository.

If you find our work helpful in your research, please cite:

``` 
M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen,
"A Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross Neighborhood Re-Ranking", 
arxiv 2017
``` 



### Usage of Expanded Cross Neighborhood Re-ranking

Our implementation of ECN can be found in the file [ECN_rerank](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/ECN_rerank.m). The script can be called with default parameters in the following way to calculate the re-ranked distance for a query and a test set.

```
ECN_dist= ECN_rerank(queryset, testset);
```

This will run ECN with the default parameters `k=25`, `t=3`, `q=8` and `method=rankdist`. The returned `ECN_dist` can then be used to calculate the scores.

#### ECN Parameters
Alternatively, you can also supply different values for these parameters by running it like

```
ECN_dist= ECN_rerank(queryset, testset, 'k',25, 't',3, 'q',8, 'method','rankdist')
```

Supported values for parameter `method` are:
* `rankdist` (default): Using the rank distance for ECN re-ranking
* `origdist`: Using the original euclidian distance for ECN re-ranking


#### Query and Test Set Format

Our ECN_rerank script expects the query and testsets in the following format:

* queryset: probe matrix (#_of_probes x feature_dimension);  feature vectors in rows
* testset = gallery matrix (#_of_gallery_images x feature_dimension); feature vectors in rows


