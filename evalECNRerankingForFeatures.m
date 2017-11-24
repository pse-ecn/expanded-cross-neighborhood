function evalECNRerankingForFeatures(querymat, queryLab, queryCam, testmat, testLab, testCam)
    addpath(genpath('evaluation'))

    noRerankingDist = pdist2(testmat, querymat, 'cosine'); % you can use euclidian here, if yor features are normalized
    [resultNoRerankingRanks, resultNoRerankingMap, ~, ~] = evaluation(noRerankingDist, testLab, queryLab, testCam, queryCam);
    
    origDistECN = ECN_rerank(querymat, testmat, 'method', 'origdist');
    [resultOrigDistECNRanks, resultOrigDistECNMap, ~, ~] = evaluation(origDistECN, testLab, queryLab, testCam, queryCam);
     
	% This calculates the main result of ECN with rank distance (which is also default if method is not specified)
    rankDistECN = ECN_rerank(querymat, testmat, 'method', 'rankdist');
    [resultRankDistECNRanks, resultRankDistECNMap, ~, ~] = evaluation(rankDistECN, testLab, queryLab, testCam, queryCam);
    
    disp('results:');
    fprintf('no reranking: mAP = %2.2f, Rank-1 = %2.2f, Rank-5 = %2.2f , Rank-10 = %2.2f, Rank-50= %2.2f\n', 100*resultNoRerankingMap, 100*resultNoRerankingRanks(1), 100*resultNoRerankingRanks(5), 100*resultNoRerankingRanks(10), 100*resultNoRerankingRanks(50));
    fprintf('origDistECN:  mAP = %2.2f, Rank-1 = %2.2f, Rank-5 = %2.2f , Rank-10 = %2.2f, Rank-50= %2.2f\n', 100*resultOrigDistECNMap, 100*resultOrigDistECNRanks(1), 100*resultOrigDistECNRanks(5), 100*resultOrigDistECNRanks(10), 100*resultOrigDistECNRanks(50));
    fprintf('rankDistECN:  mAP = %2.2f, Rank-1 = %2.2f, Rank-5 = %2.2f , Rank-10 = %2.2f, Rank-50= %2.2f\n', 100*resultRankDistECNMap, 100*resultRankDistECNRanks(1), 100*resultRankDistECNRanks(5), 100*resultRankDistECNRanks(10), 100*resultRankDistECNRanks(50));
 
end