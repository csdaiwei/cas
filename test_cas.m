function [general_info] = test_cas(name, data_index, option)
%% test_cas.m
%--------------------------------------------------------------------------
%% Split data for each modality
%--------------------------------------------------------------------------
load(name); content = content'; %content is created by "load"
data = content(:,1:end-1);label = content(:,end);
unique_label = unique(label); num_label = length(unique_label);
if num_label ~= 2   %not equal
    new_label = label;
    for ii = 1:floor(num_label/2)
        new_label(label == unique_label(ii)) = -1;
    end
    for ii = floor(num_label/2)+1 : num_label
        new_label(label == unique_label(ii)) = 1;
    end
    label = new_label; clear new_label
end
category = [ones(256,1); ones(7,1)*2; ones(64,1)*3; ones(75,1)*4; ones(128,1)*5; ones(144,1)*6; ones(225,1)*7];
modality_index1 = [1,2,4,5,6,7];
modality_index2 = [3];
% construct dimension index for two modality
index_modal1 = zeros(length(category),1); 
index_modal2 = zeros(length(category),1);
for ii = 1:length(modality_index1)
    index_modal1(category == modality_index1(ii)) = 1;
end
for ii = 1:length(modality_index2)
    index_modal2(category == modality_index2(ii)) = 1;
end
% split data into train and test, 66% for train and 33% for test
splitvector = generator(label,3,1,data_index);
train_data = data(~(splitvector == 3),:);
test_data = data(splitvector == 3,:);
train_label = label(~(splitvector == 3));
test_label = label(splitvector == 3);
%clear data

% normalization for train and test data, use range of train data to
% normalize test data
min_train = min(train_data); max_train = max(train_data);
train_data = train_data - ones(size(train_data,1),1)*min_train;
train_data = train_data./(ones(size(train_data,1),1)*(max_train-min_train));
test_data = test_data - ones(size(test_data,1),1)*min_train;
test_data = test_data./(ones(size(test_data,1),1)*(max_train-min_train));

% get the aligned data
align_rate = option.align_rate;
num_align = floor(align_rate * length(train_label));
train_align_data = train_data(1:num_align,:);
train_align_label = train_label(1:num_align,:);
train_partial_data = train_data(num_align+1:end,:);
train_partial_label = train_label(num_align+1:end,:);
clear train_data

% random sample out some data as patial unobserved data
sample_out_rate = option.sample_out_rate; % the percentage of unobserved labels
num_sample = floor(sample_out_rate * size(train_partial_data,1));
sample_in_index1 = ones(length(train_partial_label),1); sample_in_index2 = ones(length(train_partial_label),1);
sample_out_tmp_index1 = randsample(1:size(train_partial_data,1),num_sample);
sample_out_tmp_idnex2 = randsample(1:size(train_partial_data,1),num_sample);
sample_in_index1(sample_out_tmp_index1) = 0; sample_in_index2(sample_out_tmp_idnex2) = 0;
sample_in_align_index = sample_in_index1 & sample_in_index2;
train_align_data = [train_align_data; train_partial_data(logical(sample_in_align_index),:)];
train_align_label = [train_align_label; train_partial_label(logical(sample_in_align_index),:)];
sample_in_index1 = sample_in_index1 - sample_in_align_index;
sample_in_index2 = sample_in_index2 - sample_in_align_index;

% split data into two modality
x_pair = train_align_data(:,logical(index_modal1)); y_pair = train_align_data(:,logical(index_modal2));
x_single = train_partial_data(logical(sample_in_index1), logical(index_modal1));
x_single_label = train_partial_label(logical(sample_in_index1));
y_single = train_partial_data(logical(sample_in_index2), logical(index_modal2));
y_single_label = train_partial_label(logical(sample_in_index2));
x_test = test_data(:,logical(index_modal1)); y_test = test_data(:,logical(index_modal2));
clear train_align_data train_partial_data train_partial_label test_data

% construct unlabel data
% for aligned data, permute for two modality, and for single data, permute
% data for each modality and then split into label and unlabel ones
rand('state', data_index);
align_perm = randperm(length(train_align_label));
x_pair = x_pair(align_perm,:); y_pair = y_pair(align_perm,:); 
train_align_label = train_align_label(align_perm);
pair_label = train_align_label(1:floor(length(train_align_label)*option.label_ratio));
x_single_perm = randperm(size(x_single,1));
y_single_perm = randperm(size(y_single,1));
x_single = x_single(x_single_perm,:); x_single_label = x_single_label(x_single_perm);
y_single = y_single(y_single_perm,:); y_single_label = y_single_label(y_single_perm);
x_single_label = x_single_label(1:floor(length(x_single_label)*option.label_ratio));
y_single_label = y_single_label(1:floor(length(y_single_label)*option.label_ratio));

% set parameters for the Algorithm
option.MAX_ITER = 10;       % the number of iteration of ADMM
option.opt_MAX_PASS = 20;  % the number of iteration of SDCA 
option.stat_MAX_ITER = 10;  % the number of iteration of greedy projection
option.stat_scale = 0;
option.rho = 0.5;
option.lambda = 0.1;
w = cas_train(x_pair,y_pair,pair_label,x_single,x_single_label,y_single,y_single_label, option);

% test the algorithm on both modalities
w1 = w(1:size(x_pair,2)); w2 = w(size(x_pair,2)+1:end);
x_predict = x_test * w1; y_predict = y_test * w2;
acc_x = mean((x_predict .* test_label) > 0); acc_y = mean((y_predict .* test_label) > 0);
disp([acc_x, acc_y]);