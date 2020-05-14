# Expanded Cross Neighborhood distance based Re-ranking (ECN)

In this repository, we provide the code for our ECN re-ranking method described in the CVPR2018 paper **A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking**. [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.pdf).

This repo includes both Python and Matlab code of our Expanded Cross Neighborhood Re-Ranking. 

If you find our work helpful in your research, please cite:

``` 
@inproceedings{ECN,
    author    = {M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen}, 
    title     = {A Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross Neighborhood Re-Ranking}, 
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year  = {2018}
}
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

* In MATLAB 
You can then run the evaluation by setting the evalPath to your download features folder in the function [evalECNRerankingForPath](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/evalECNRerankingForPath.m).

** If you would like to test our ECN reranking on your own features , you can directly use the [evalECNRerankingForFeatures](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/evalECNRerankingForFeatures.m) or directly the main [ECN_rerank](https://github.com/pse-ecn/expanded-cross-neighborhood/blob/master/ECN_rerank.m). 

* In Python:
From command  line
```
python3 evaluate_person_reid.py --querypath [path to query prediction folder ../../query] --testpath [path to test prediction folder ../../test] 

```

** or if you want to evalute with ECN reranking,  speciy with flag --do_rerank

```
python3 evaluate_person_reid.py --querypath [path to query prediction folder ../../query] --testpath [path to test prediction folder ../../test] --do_rerank

```

OR

from evaluate_person_reid import evaluate

mAP, CMC= evaluate(querymat=queryset, testmat=testset, query_ids=query_labels, gallery_ids=test_labels,
            query_cams=query_cams_id, gallery_cams=test_cam_id, do_rerank=False)

            



