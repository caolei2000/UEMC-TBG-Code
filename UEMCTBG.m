function [idx,S] = UEMCTBG(X,para,k)
% Unified and Efficient Multi-View Clustering with Tensorized Bipartite Graph
%Input:
%       X: 数据矩阵, cell类型, size=d*n;
%       para: 参数
%           lambda1: 低秩张量项的权重
%           lambda2: 图融合项的权重
%           lambda3: 谱分解项的权重
%           k: 簇数
%Output:
%       idx: 结果索引, size=1*n
%       S: 学习到的融合二分图
%
% Written by Lei Cao
% leicao2000@gmail.com
% 2024/1/16
%% 初始化
lambda1 = para.lambda1;lambda2 = para.lambda2;lambda3 = para.lambda3;
d = k;m = k;MaxIter = 10;
v = length(X);
omega = ones(1,v);
n = size(X{1},2);
sX = [n,m,v];
[X] = data_prep(X);  % 数据预处理
W = cell(1,v);Z = cell(1,v);J = cell(1,v);Y = cell(1,v);
for i = 1:v
    W{i} = zeros(size(X{i},1),d);  % 初始化Wi
    Z{i} = eye(n,m);  % 初始化Zi
    J{i} = eye(n,m);  % 初始化Ji
    Y{i} = zeros(n,m);  % 初始化乘子Yi
end
A = zeros(d,m);  % 初始化A
S = eye(n,m);  % 初始化S
G = zeros(n+m,k);  % 初始化谱嵌入G 
rho = 1;  % 惩罚参数
%% 开始迭代求解
for iter = 1:MaxIter
    %% 1.更新Wi
    parfor i = 1:v
        [U_b,~,V_b] = svd(X{i}*Z{i}*A',"econ");
        W{i} = U_b*V_b';
    end
    %% 2.更新A
    WX = cell(1, v);  % 预先计算W'X
    for i = 1:v
        WX{i} = W{i}' * X{i};
    end
    C = zeros(d,m);
    for i = 1:v
        C = C + WX{i}*Z{i};
    end
    [U_c,~,V_c] = svd(C,"econ");
    A = U_c*V_c';
    %% 3.更新Zi
    opts = optimset("Display","off");
    parfor i = 1:v
        H = (2+2*lambda2*omega(i)+rho)*eye(m);
        for j = 1:n
            f = -2*A'*WX{i}(:,j)-2*lambda2*omega(i)*S(j,:)'-rho*J{i}(j,:)'+Y{i}(j,:)';
            Z{i}(j,:) = quadprog(H,f,[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],opts);
        end
    end
    %% 4.更新J
    Z_tensor = cat(3, Z{:,:});
    Y_tensor = cat(3, Y{:,:});
    z = Z_tensor(:);
    y = Y_tensor(:);
    [j, ~] = wshrinkObj(z + 1/rho*y,lambda1/rho,sX,0,3);
    J_tensor = reshape(j, sX);
    for i = 1:v
        J{i} = J_tensor(:,:,i);
    end
    %% 5.更新S
    D_c = diag(sum(S,1));  % 列度矩阵
    D_c_half_inv = diag(1./sqrt(diag(D_c) + eps));
    Z_tmp = zeros(n,m);
    for i = 1:v
        Z_tmp = Z_tmp + omega(i)*Z{i};
    end
    S = (lambda3*G(1:n,:)*G(n+1:end,:)'*D_c_half_inv'+lambda2*Z_tmp) / (lambda2*sum(omega) + eps);
    S(S<0) = 0;  % 去除小于0的数值
    %% 6.更新G
    D_c = diag(sum(S,1));  % 更新列度矩阵
    D_c_half_inv = diag(1./sqrt(diag(D_c) + eps));
    S_hat = S*D_c_half_inv;
    [G_n,~,G_m] = svd(S_hat,"econ");
    G = [G_n(:,1:k);G_m(:,1:k)];  % 取最大的K个特征向量
    %% 7.更新Y
    for i = 1:v
        Y{i} = Y{i} + rho*(Z{i}-J{i});
    end
    rho = min(2*rho,1e3);
    %% 8.记录obj
    term1 = 0;
    term3 = 0;
    for i = 1:v
        term1 = term1 + norm(X{i} - W{i} * A * Z{i}','fro')^2;
        term3 = term3 + omega(i)*norm(S-Z{i},'fro')^2;
    end
    term4 = 2*trace(G(1:n,:)'*S_hat*G(n+1:end,:));
    obj(iter) = term1 + lambda2*term3 - lambda3*term4;
    if iter>4 && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-1 || obj(iter) < 1e-3)
        break
    end
end
%% 9.执行kmeans
Gn = G(1:n,:);
Gn_normalized = Gn ./ repmat(sqrt(sum(Gn.^2, 2)),1,size(Gn,2));  % 将Pn的每一行标准化为单位向量
[idx] = kmeans(Gn_normalized,k,'Replicates',20,'Distance','cosine');
idx = idx';

end

function [X] = data_prep(X)
% 数据预处理函数
%Input:
%       X: cell类型,数据矩阵, size(X{i})=d*n;
%Output:
%       X: cell类型,数据矩阵, size(X{i})=d*n;

v = length(X);
for i = 1:v
    X{i} = X{i} ./ (repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);  % 单位向量归一化
end

end



