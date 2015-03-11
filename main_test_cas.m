%% test cas Algorithm
%get last results using selected parameter
clear;clc;
name = 'MSRA_m';
%set the test parameter
data_index = 1;
option.align_rate = 0.2;
option.sample_out_rate = 0.3;
option.label_ratio = 0.6;
test_cas(name, data_index, option);
