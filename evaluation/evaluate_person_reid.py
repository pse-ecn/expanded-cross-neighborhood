from __future__ import absolute_import
import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise, average_precision_score
import pandas as pd

from ecn import ECN

'''
Evalaution script adapted from https://github.com/Cysu/open-reid

'''


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        single_gallery_shot=False,
        first_match_break=True):

    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = (gallery_cams[indices[i]] != query_cams[i])
        '''
        valid = ((gallery_ids[indices[i]] == -1) | #query_ids[i]
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        '''

        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    #distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = (gallery_cams[indices[i]] != query_cams[i])
        '''
        valid = ((gallery_ids[indices[i]] == -1) | #query_ids[i]
                 (gallery_cams[indices[i]] != query_cams[i]))
        '''
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def evaluate(querymat=None, testmat=None, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, do_rerank=False):
    if do_rerank:
        distmat = ECN(querymat, testmat).transpose()
    else:
        distmat = pairwise.pairwise_distances(querymat, testmat, metric='cosine', n_jobs=-1)

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    CM = cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    return mAP, CM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--querypath', help='Specify the folder where the query csv prediction data is written', dest='querypath')
    parser.add_argument('--testpath', help='Specify the folder where the test csv prediction data is written', dest='testpath')
    parser.add_argument('--do_rerank', help='flag to specify if results should be computed with ECNN reranking', action='store_true')
    args = parser.parse_args()

    querymat = pd.read_csv(os.path.join(args.querypath, 'features.csv')).to_numpy().astype('float32')
    query_ids = pd.read_csv(os.path.join(args.querypath,'labels.csv')).to_numpy().squeeze()
    query_cams = pd.read_csv(os.path.join(args.querypath, 'cameras.csv')).to_numpy().squeeze()

    testmat = pd.read_csv(os.path.join(args.testpath, 'features.csv')).to_numpy().astype('float32')
    gallery_ids = pd.read_csv(os.path.join(args.testpath,'labels.csv')).to_numpy().squeeze()
    gallery_cams = pd.read_csv(os.path.join(args.testpath,'cameras.csv')).to_numpy().squeeze()

    mAP, CMC = evaluate(querymat, testmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, do_rerank=args.do_rerank)

    if args.do_rerank:
        str = 'With ECN rerank'
    else:
        str = 'Base: No-rerank'

    print('{}: mAP={:4.1%}, Rank-1={:3.2%}, Rank-5={:3.2%}, Rank-10={:3.2%}, Rank-50={:3.2%}'.format(str, mAP, CMC[0], CMC[4], CMC[9], CMC[49]))

if __name__ == '__main__':
    main()