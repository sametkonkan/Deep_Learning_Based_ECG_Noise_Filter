%% 12. compare_qrs Helper Function
function [TP, FP, FN] = compare_qrs(detected, reference, tolerance)
    % Detected QRS peaks are compared with reference QRS peaks.
    % A detection is considered true positive (TP) if it is within the "tolerance"
    % sample distance from the reference QRS.
    TP = 0;
    FP = 0;
    FN = 0;
    ref_matched = false(size(reference));
    
    for i = 1:length(detected)
        diffs = abs(reference - detected(i));
        if any(diffs <= tolerance)
            TP = TP + 1;
            [~, idx] = min(diffs);
            ref_matched(idx) = true;
        else
            FP = FP + 1;
        end
    end
    FN = sum(~ref_matched);
    fprintf("FN =%.4f\n ",FN);
end