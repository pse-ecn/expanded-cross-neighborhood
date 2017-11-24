%% This function is the original market evaluation function from
%% https://github.com/zhunzhong07/IDE-baseline-Market-1501/blob/master/market_evaluation/utils/compute_AP.m 
%% by Zheng et al.

function [ap, cmc] = compute_AP(good_image, junk_image, index)

cmc = zeros(length(index), 1);
ngood = length(good_image);

old_recall = 0; 
old_precision = 1.0; 
ap = 0; 
intersect_size = 0; 
j = 0; 
good_now = 0; 
njunk = 0;
for n = 1:length(index) 
    flag = 0;
    if ~isempty(find(good_image == index(n), 1)) 
        cmc(n-njunk:end) = 1;
        flag = 1; % good image 
        good_now = good_now+1; 
    end
    if ~isempty(find(junk_image == index(n), 1))
        njunk = njunk + 1;
        continue; % junk image 
    end
    
    if flag == 1%good
        intersect_size = intersect_size + 1; 
    end 
    recall = intersect_size/ngood; 
    precision = intersect_size/(j + 1); 
    ap = ap + (recall - old_recall)*((old_precision+precision)/2); 
    old_recall = recall; 
    old_precision = precision; 
    j = j+1; 
    
    if good_now == ngood 
        return; 
    end 
end 

end


