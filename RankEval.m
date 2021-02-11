function [Metric] = RankEval(Result, Target, k)
    % Find Ranked Result
    k = int32(k);
    l = size(Target, 1);
    Label = max(Target, 0);
    [SResult, Is] = sort(Result, 'descend');

    Imax = zeros(size(Result));
    Imin = zeros(size(Result));
    SLabel = zeros(size(Result));
    TLabel = zeros(size(Result));
    DLabel = zeros(size(Result));
    PCum = zeros(size(Result));
    NCum = zeros(size(Result));

    %% Sample-wise Eval
    for i=1:size(Label, 2)
        % Max ranking
        [~, Imin(:, i)] = ismember(Result(:, i), SResult(:, i));
        [~, imax] = ismember(Result(:, i), flip(SResult(:, i)));
        Imax(:, i) = sort(l + 1 - imax);
        SLabel(:, i) = Label(Is(:, i), i);

        Top = Imax(:, i) <= k;
        Tie = Imax(:, i) > k & sort(Imin(:, i)) <= k;
        if any(Tie)
            TLabel(:, i) = SLabel(:, i) .* Top + Tie .* (sum(Tie .* SLabel(:, i)) ./ sum(Tie));
        else
            TLabel(:, i) = SLabel(:, i) .* Top; 
        end
        PCnt = full(sparse(Imin(:, i), 1, Label(:, i), l, 1));
        NCnt = full(sparse(Imin(:, i), 1, double(~Label(:, i)), l, 1));
        AvgCnt = PCnt ./ (PCnt + NCnt);
        DLabel(:, i) = AvgCnt(sort(Imin(:, i)));
        PCum(:, i) = cumsum(PCnt);
        NCum(:, i) = cumsum(NCnt);
    end  

    LabelC = max(sum(Label, 1), 1);
    Spec = all(~boolean(Label)) | all(boolean(Label));
    TP = sum(TLabel(1:k, :), 1);
    IDCG = sort(Label, 'descend');
    
    Metric.RL = mean(sum(SLabel .* NCum, 1) ./ max(LabelC .* sum(~Label, 1), 1), 'all');
    Metric.MAP = mean(max(sum(SLabel .* PCum ./ Imax, 1) ./ LabelC, Spec), 'all');
    Metric.AP = mean(max(sum(TLabel .* PCum ./ Imax, 1) ./ LabelC, Spec), 'all');
    Metric.COV = mean(max(Imax .* SLabel, [], 1)) / l;
    Metric.PREC = mean(TP ./ k, 'all');
    Metric.REC = mean(TP ./ LabelC, 'all');
    Metric.F1 = 2 .* (Metric.PREC .* Metric.REC) ./ (Metric.PREC + Metric.REC);
    Metric.NDCG = mean(max(sum(DLabel(1:k, :) ./ log2(2:k+1)', 1), Spec) ./ max(sum(IDCG(1:k, :) ./ log2(2:k+1)', 1), 1));
end