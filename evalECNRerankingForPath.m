function evalECNRerankingForPath(evalPath)
    querymat=csvread([evalPath '/query/features.csv']);
    queryLab=csvread([evalPath '/query/labels.csv']);
    queryCam=csvread([evalPath '/query/cameras.csv']);

    testmat=csvread([evalPath '/test/features.csv']);
    testLab=csvread([evalPath '/test/labels.csv']);
    testCam=csvread([evalPath '/test/cameras.csv']);  
    
    evalECNRerankingForFeatures(querymat, queryLab, queryCam, testmat, testLab, testCam);
end