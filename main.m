clear
clc

%% data path
path ="\path\";
Modal = readtable(strcat(path, 'ModalLabel.csv')); %% ModalLabel
Modal = table2array(Modal(:,3:end));

%% paremeters settings
para = [1e-3, 1e-2, 1e-1, 1e0];
para1 = [5, 10, 20, 40];
opts.h = 10;
opts.lambda = para(3);
opts.beta = para(4);
opts.gamma = para(3);
numcv = 5;
opts.maxIter = 1e3;
opts.tol = 1e-5;

%% the nested 5-fold
for j = 1:5
    for i1 = 1:length(para)
        opts.lambda = para(i1);
        for i2 = 1:length(para)
            opts.beta = para(i2);
            for i3 = 1:length(para)
                opts.gamma = para(i3);
                for i4 = 1:length(para1)
                    opts.h = para1(i4);
                    for k = 1:numcv
                        fprintf('j = %d, lambda = %d, beta = %d, gamma = %d, h = %d, k = %d \n', j, opts.lambda, ...
                            opts.beta, opts.gamma, opts.h, k);

                        % read data
                        Train = table2array(readtable(strcat(path, 'Train', num2str(j), num2str(k), '.csv')));
                        Valid = table2array(readtable(strcat(path, 'Valid', num2str(j), num2str(k), '.csv')));
                        label_tr = (Train(:,2));
                        idx = find(label_tr == 0);
                        label_tr(idx) = -1;
                        label_va = (Valid(:,2));
                        idx = find(label_va == 0);
                        label_va(idx) = -1;
                        Xtr1 = (Train(:,3:end));
                        meanX = mean(Xtr1, "omitnan");
                        stdX = std(Xtr1, "omitnan");
                        Xtr1 = (Xtr1 - meanX) ./ (stdX + eps);
                        Xva1 = (Valid(:,3:end));
                        Xva1 = (Xva1 - meanX) ./ (stdX + eps);
                        
                        ind1 = find(isnan(Xtr1));
                        ind2 = find(isnan( Xva1));
                        Xtr1(ind1)=0;
                        Xva1(ind2)=0;
                        
                        for m = 1:max(Modal)
                            idx = find(Modal == m);
                            Xtr{m} = Xtr1(:,idx);
                            Xva{m} = Xva1(:,idx);
                        end
                        tic
                        [W, V, H, b] = Model(Xtr, label_tr, opts);
                        toc
                        Hva1 = zeros(opts.h, size(Xva{1}, 1));
                        for m = 1:max(Modal)
                            Hva = V{m}' * Xva{m}';
                            idx = find(isnan(Hva(1,:)));
                            Hva(:,idx) = 0;
                            Hva1 = Hva1 + Hva;
                        end
                        Hva1 = Hva1 / max(Modal);
                        yva = sign(Hva1'*W + b);
                        y = 1 ./ (1 + exp(-(Hva1'*W + b)));
                        ACCva(i1, i2, i3, i4, k) = sum(yva == label_va) /length(yva);
                        [A1,A2,T,AUC] = perfcurve(label_va, y, 1);
                        AUCva(i1, i2, i3, i4, k) = AUC;
                        fprintf("ACC = %2.5f, AUC = %2.5f \n", ACCva(i1, i2, i3, i4, k), AUC)
                        
                    end
                end
            end
        end
    end
    AUCva1 = mean(AUCva,5);
    [max_val, pos_max] = max(AUCva1(:));
    [idx1(j),idx2(j),idx3(j),idx4(j)] = ind2sub(size(AUCva1), pos_max);
    
    opts.h = para1(idx4(j));
    opts.lambda = para(idx1(j));
    opts.beta = para(idx2(j));
    opts.gamma = para(idx3(j));

    
    Train = table2array(readtable(strcat(path, 'Train', num2str(j), num2str(1), '.csv')));
    Valid = table2array(readtable(strcat(path, 'Valid', num2str(j), num2str(1), '.csv')));
    Test = table2array(readtable(strcat(path, 'Test', num2str(j), '.csv')));
    Train1 = [Train; Valid];
    
    labelTe = Test(:,2);
    labelTr = Train1(:,2);
    idx = find(labelTr == 0);
    labelTr(idx) = -1;
    idx = find(labelTe == 0);
    labelTe(idx) = -1;
    Xtr1 = Train1(:, 3:end);
    meanX = mean(Xtr1, "omitnan");
    stdX = std(Xtr1, "omitnan");
    Xtr1 = (Xtr1 - meanX) ./ (stdX + eps);
    Xte1 = Test(:,3:end);
    Xte1 = (Xte1 - meanX) ./ (stdX + eps);
    
    ind1 = find(isnan(Xtr1));
    ind2 = find(isnan( Xte1));
    Xtr1(ind1)=0;
    Xte1(ind2)=0;
    
    for m = 1:max(Modal)
        idx = find(Modal == m);
        Xtr{m} = Xtr1(:,idx);
        Xte{m} = Xte1(:,idx);
        
    end
    tic
    [W, V, H,b] = Model(Xtr, labelTr, opts);
    toc
    Hva1 = zeros(opts.h, size(Xte{1}, 1));
    for m = 1:max(Modal)
        Hva = V{m}' * Xte{m}';
        idx = find(isnan(Hva(1,:)));
        Hva(:,idx) = 0;
        Hva1 = Hva1 + Hva;
    end

    Hva1 = Hva1 / max(Modal);    
    y = 1 ./ (1 + exp( -(Hva1'*W + b)));
    yte = sign(Hva1'*W + b);
    ACCte(j) = sum(yte == labelTe) /length(yte);
    [A1,A2,T,AUC] = perfcurve(labelTe, y, 1);
    AUCte(j) = AUC;
    

    C = confusionmat(labelTe, yte);

    TP = C(2,2);  % True Positives
    FN = C(2,1);  % False Negatives
    TN = C(1,1);  % True Negatives
    FP = C(1,2);  % False Positives
    sensitivity(j) = TP / (TP + FN);
    specificity(j) = TN / (TN + FP);
    precision = TP / (TP + FP);
    recall = sensitivity;
    F1_score(j) = 2 * (precision * recall) / (precision + recall);
    
   
end
