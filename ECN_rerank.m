function ECN_dist= ECN_rerank(queryset, testset, varargin)
% Expanded Cross Neighbourhood distance based Re-ranking (ECN)
% Usage:  
% ECN_dist= ECN_rerank(queryset, testset); with default parameters
%
% or supply parameters as key value pair
% ECN_dist= ECN_rerank(queryset, testset, 'k',25, 't',3, 'q',8, 'method','rankdist')
%
% Inputs:
% queryset =probe matrix (#_of_probes x featdim) feature vectors in rows
% testset = Gallery matrix (feature vectors in rows)
% k,t,q= ECN parmaters (defaults k=25, t=3, q=8)
% method = rankdist :default(based on rank list compariosn) or origdist (orignol euclidean dist) : specifies the dist to be used for reranking
%
% Output:
% ECN_dist = reranked distance matrix [size: #test x #query]
%
% Copyright 
% M. Saquib Sarfraz (Karlsruhe Institute of Technology (KIT)), 2017
% if you use this code , please cite
%
% M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen, " A
% Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross
% Neighborhood Re-Ranking", arxiv 2017


%%
% Load default ECN parameters (k,t,q) and dist method (orig dist or rank dist) if not provided
ECN_param = parse_it(); parse(ECN_param,varargin{:});
k= ECN_param.Results.k; t= ECN_param.Results.t; q=ECN_param.Results.q; method=ECN_param.Results.method; 


nQuery=size(queryset,1); ntest=size(testset,1);
mat=[queryset ; testset];

orig_dist=pdist2(mat,mat,'cosine'); % use 'euclidian' dist here if your features are normalised
[~, initial_rank]=sort(orig_dist,2,'ascend'); 
clear mat; clear queryset; clear testset; 

switch method
    case 'rankdist'
        clear orig_dist;
        r_dist= get_rank_dist(initial_rank,k);
    case 'origdist'
        r_dist=orig_dist;
        clear orig_dist
end

disp('ECN re-ranking ... prepared dist mats and rank lists.. commencing to match')

%% ECN re-ranking %%
top_t_nb=initial_rank(:,2:t+1);  % top neighbour indxs

t_ind=top_t_nb(nQuery+1:end,:).';  % test top t nbr
next_2_tnbr=initial_rank(t_ind,2:q+1);   %   .. (t*ntest,lq2)

q_ind=top_t_nb(1:nQuery,:).'; % query top t nbr
next_2_qnbr=initial_rank(q_ind,2:q+1);   %    .. (t*nQuery,lq2)
clear initial_rank;

next_2_tnbr=reshape(next_2_tnbr',[t*q,ntest]);
t_ind=[t_ind;next_2_tnbr];
clear next_2_tnbr;
next_2_qnbr=reshape(next_2_qnbr',[t*q,nQuery]);
q_ind=[q_ind;next_2_qnbr];
clear next_2_qnbr;

t_nbr_dist=r_dist(t_ind,1:nQuery); % dist of test top nbrs wrt to query: size is [(t x testsize) , nQuery]
t_nbr_dist=reshape(t_nbr_dist,[t+t*q,ntest,nQuery]);  %size. [:,:,nquery]

q_nbr_dist=r_dist(q_ind,nQuery+1:end); % dist of query top nbrs wrt to test: size is [(t x nQuery) , ntest]
q_nbr_dist=reshape(q_nbr_dist,[t+t*q,nQuery,ntest]);
q_nbr_dist=permute(q_nbr_dist,[1,3,2]);   

ECN_dist=squeeze(mean([q_nbr_dist;t_nbr_dist]));  %%%final ECN distance: average nbr distances for each query-test pair

%uncomment this to get the rank dist ECN equation 4
%rdist=r_dist(1:nQuery,nQuery+1:end);   % rank dist (query x test)

%%
function Rank_dist=get_rank_dist(initial_rank,k)
[~,pos_L1]=sort(initial_rank, 2,'ascend'); 
fac_1=(max(0,(k+1- pos_L1)));
Rank_dist = fac_1*fac_1';
%convert the similarites to distance
 Rank_dist=min_max_norm(Rank_dist);
 Rank_dist=(1-Rank_dist) ;
end

function [mat]=min_max_norm(mat)
   mat= bsxfun(@minus,mat,min(mat,[],2)); %subtract each row min from each row
   max_mat=max(mat,[],2); min_mat=min(mat,[],2); % geting max and min 
   max_min=max_mat-min_mat;
   mat=bsxfun(@rdivide,mat,max_min);
end
%%
function ECN_param = parse_it()
   ECN_param= inputParser;
   default_k = 25;
   default_t = 3;
   default_q = 8;
   
   defaultmethod='rankdist';
   expectedmethods = {'origdist','rankdist'};
   
   addParameter(ECN_param,'method',defaultmethod,...
                 @(x) any(validatestring(x,expectedmethods)));

    addParameter(ECN_param,'k',default_k);
    addParameter(ECN_param,'t',default_t );
    addParameter(ECN_param,'q',default_q);
   
end
%%
end