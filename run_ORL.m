%% 加载数据
clear
warning off all
addpath(genpath('./'));
load ORL_mtv.mat  
nCluster = length(unique(truth));
%% 测试UAL
para.lambda1 = 20;  % 低秩的权重, 从范围内选择: {1,10,20,30,40,50,60,70,80,90,100}
para.lambda2 = 1;  % 图融合的权重, 固定
para.lambda3 = 1;  % 谱聚类的权重, 固定
% 运行UAL
tic
[idx,S] = UEMCTBG(X,para,nCluster);
timer = toc;
result = Clustering8Measure(truth, idx);
fprintf('Anchors: %d \t ACC: %6.4f \t NMI: %6.4f \t Purity: %6.4f \t Fscore: %6.4f \t Time: %.1f second \n',[nCluster result(1) result(2) result(5) result(4) timer]);
%% 可视化
subplot(1,2,1)
image(S,'CDataMapping','scaled')  % 二分图可视化
title('Bipartite graph')
ylabel("Data point")
xlabel("Anchor")
subplot(1,2,2)
image(S*S','CDataMapping','scaled')  % 相似度图可视化
title('Similarity graph')
ylabel("Data point")
xlabel("Data point")