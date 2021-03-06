function [output] = cas_train(x_pair, y_pair, pair_label, x_single, x_single_label, y_single, y_single_label, option)
%% train CAS(Concatenated ADMM-SDCA) for Multimodality Data
%% input and output variable
%   x_pair, y_pair                  paired data of two modality(labeld + unlabeled)
%   pair_label                      true label of paired labeled data
%   x(y)_single, x(y)_single_label  single modality data
%   option                          option for optimization

%% Initialization
% get basic information
[N01, D1] = size(x_pair);  [N02, D2] = size(y_pair);
assert(N01 == N02); % num of paired instance should be the same
N0 = N01;
N0_l = length(pair_label); % N0_u = N01 - N0_
[N1, D11] = size(x_single); [N2, D22] = size(y_single);
assert(D11 == D1 && D22 == D2);
N1_l = length(x_single_label); % N1_u = N1 - N1_l;
N2_l = length(y_single_label); % N2_u = N2 - N2_l;
total_num = N0_l*2 + N1_l + N2_l;

w = zeros(D1 + D2, 1);
z = zeros(D1 + D2, 1);
u = zeros(D1 + D2, 1);


%construct full samples and labels
samples = zeros(total_num, D1+D2);
samples(1:N0_l, 1:D1) = x_pair(1:N0_l, 1:D1);
samples(N0_l+1:N0_l+N1_l, 1:D1) = x_single(1:N1_l, 1:D1);
samples(N0_l+N1_l+1:2*N0_l+N1_l, D1+1:end) = y_pair(1:N0_l, 1:D2);
samples(2*N0_l+N1_l+1:end, D1+1:end) = y_single(1:N2_l, 1:D2);
    
labels = zeros(total_num, 1);
labels(1:N0_l, 1) = pair_label;
labels(N0_l+1:N0_l+N1_l, 1) = x_single_label;
labels(N0_l+N1_l+1:2*N0_l+N1_l, 1) = pair_label;
labels(2*N0_l+N1_l+1:end, 1) = y_single_label;


% disp('svm train')
% model = train(labels, sparse(samples));
% disp('svm predict')
% [svmpl, svmaccu, ~] = predict(labels, sparse(samples), model);

sdca_record = [];

%% ADMM frame
for ii = 1 :option.MAX_ITER
    %% SDCA for minimize loss for both modalities
    % w-update in a SGD manner
    % data       Modality1       Modality2
    %           N0_l + N1_l  +  N0_l + N2_l
    tic;
    disp(['ADMM Iter: ', num2str(ii)]);
    alpha = zeros(total_num,1);
   
    [loss, accu] = check_loss(samples, labels, w);
    r = w-z+u;
    obj_value = loss + 0.5 * option.lambda * (w'*w) + 0.5 * option.rho * (r'*r);
    disp(['before sdca, loss: ', num2str(loss), ', obj value: ', num2str(obj_value), ...,
        ', accu: ', num2str(accu)]);
    
    for kk = 1:option.opt_MAX_PASS
        perm_index = randperm(total_num);
        % only get 80% of index for opt
        perm_index = perm_index(1:round(total_num * 0.8));
        for index = 1:round(total_num*0.8)
            current_index =  perm_index(index);
            x_whole = samples(current_index, :)';
            y = labels(current_index);   
            % get eta
            c = option.rho/option.lambda;
            q = z - u;
            eta = option.lambda*c*(1+q'*x_whole*y)/((c-1)*(x_whole'*x_whole)) ...,
                - option.lambda*(w'*x_whole*y)/(x_whole'*x_whole) + alpha(current_index)*y;
            % get delta_alpha
            delta_alpha = y*max(0,min(1,eta)) - alpha(current_index);
            % update alpha and w
            alpha(current_index) = alpha(current_index) + delta_alpha;
            w = w + 1/option.lambda * delta_alpha * x_whole;
            
            %check_indice = [1:10,20:10:100, 200:1000:1000, 2000:1000:round(total_num * 0.8), round(total_num * 0.8)];
            check_indice = [];
            if(kk==1 && ismember(index, check_indice))
                %watch obj value descreasing or not
                %obj_value = loss + regularizer + residual
                [loss, accu] = check_loss(samples, labels, w);
                r = w-z+u;
                obj_value = loss + 0.5 * option.lambda * (w'*w) + 0.5 * option.rho * (r'*r);
                disp(['at sdca pass 1, iter: ', num2str(index), ', loss: ', num2str(loss), ' , obj value: '...
                    , num2str(obj_value), ', accu: ', num2str(accu)]);
            end
            
        end
        
        [loss, accu] = check_loss(samples, labels, w);
        r = w-z+u;
        obj_value = loss + 0.5 * option.lambda * (w'*w) + 0.5 * option.rho * (r'*r);
        disp(['after sdca pass: ', num2str(kk), ', loss: ', num2str(loss), ', obj value: ', num2str(obj_value), ...,
            ', accu: ', num2str(accu)]);
        
        
    end
    disp(['sdca time: ', num2str(toc)]);
    
    %% deal with constraints
    % z-update
    
    tic;
    z = w + u; 
    
    cstt_cnt_w = check_constraints(x_pair, y_pair, z);
    cstt_cnt_z = check_constraints(x_pair, y_pair, z);
    disp(['before z-update, constrains(w/z/all) ', num2str(cstt_cnt_w), '/', num2str(cstt_cnt_z),'/', num2str(N0)]);
    
    w1 = z(1:D1); w2 = z(D1+1:end);
    for kk = 1:option.stat_MAX_ITER
        
        % construct cube for unlabeled paired data
        temp_m1 = repmat(w1,1,N0) .* x_pair';
        temp_m2 = repmat(w2,1,N0) .* y_pair';
        cube = zeros(D1,D2);
        for jj = N0_l+1:N0
            if option.stat_scale == 0
                cube = cube + temp_m1(:,jj) * temp_m2(:,jj)';
            else
                cube = cube + (temp_m1(:,jj) * temp_m2(:,jj)' < 0);
            end
        end
        if option.stat_scale == 0 && all(all(cube > 0))
            break;
        elseif option.stat_scale == 1 && all(all(cube == 0))
            break;
        end
        % find the most-negative from the histogram
        if option.stat_scale == 0
            % find the min-value
            [col_min_value, col_min_index] = min(cube);
            [~, min_index_col] = min(col_min_value);
            min_index_row = col_min_index(min_index_col);
            % check the negative one by sum the col or row
            row_sum = sum(cube(min_index_row,:));
            col_sum = sum(cube(:,min_index_col));
            if(row_sum > col_sum)
                % choose min_index_col
                w2(min_index_col) = 0;
                disp(['make w2(', num2str(min_index_col), ') to 0']);
            else
                % choose min_index_row
                w1(min_index_row) = 0;
                disp(['make w1(', num2str(min_index_row), ') to 0']);
            end
        else
            % find the max-value, the most negative freq
            [col_max_value, col_max_index] = max(cube);
            [~, max_index_col] = max(col_max_value);
            max_index_row = col_max_index(max_index_col);
            % check the negative one by sum the col or row
            row_sum = sum(cube(max_index_row,:));
            col_sum = sum(cube(:,max_index_col));
            if(row_sum < col_sum)
                % choose max_index_col
                disp(['make w2(', num2str(max_index_col), ') to 0']);
                w2(max_index_col) = 0;
            else
                % choose max_index_row
                disp(['make w1(', num2str(max_index_row), ') to 0']);
                w1(max_index_row) = 0;
            end
        end
        z = [w1;w2];
         

        %check z-update work or not
        [loss, accu] = check_loss(samples, labels, z);
        cstt_cnt = check_constraints(x_pair, y_pair, z);
        r = w-z+u;
        obj_value = loss + 0.5 * option.lambda * (z'*z) + 0.5 * option.rho * (r'*r);
        disp(['proj pass: ', num2str(kk), ' loss: ', num2str(loss) ', obj_value: ', num2str(obj_value), ...,
            ' accu: ', num2str(accu), ...,
            ' constraints satisfied: ', num2str(cstt_cnt), '/', num2str(N0)]);
        if(cstt_cnt == N0)
            break;
        end
    end
    
    
    %% u-update
    res = w - z;    %residual
    u = u + res;    %running sum of residual
    
    disp(['constraints time: ', num2str(toc)]);
    
    disp(['w norm: ', num2str(norm(w)), ' non-zero: ', num2str(sum(w~=0))]);
    disp(['z norm: ', num2str(norm(z)), ' non-zero: ', num2str(sum(z~=0))]);
    disp(['u norm: ', num2str(norm(u)), ' non-zero: ', num2str(sum(u~=0))]);
    disp(['r norm: ', num2str(norm(res)), ' non-zero: ', num2str(sum(res~=0))]);
    
    if(norm(res) == 0) %<10e-6?
        %break;
    end
    
end % end of ADMM
output = w;