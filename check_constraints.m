function [constraints] = check_constraints( x_pair, y_pair, w)
%%calculate num of paired samples that satisfy the constraints
[~, D1] = size(x_pair);
w1 = w(1:D1); w2 = w(D1+1:end);
constraints = sum((x_pair*w1).*(y_pair*w2)>0);
end