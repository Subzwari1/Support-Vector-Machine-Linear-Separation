function m = compute_metrics(TP, TN, FP, FN)
% Returns the five standard classification metrics in a struct

  denom = TP + TN + FP + FN;
  m.accuracy    = (TP + TN) / denom;

  if TP + FN > 0
    m.sensitivity = TP / (TP + FN);
  else
    m.sensitivity = NaN;
  end


  % True negative rate
  if TN + FP > 0
    m.specificity = TN / (TN + FP);
  else
    m.specificity = NaN;    % if there are no actual negatives (TN+FP == 0)
  end

  if TP + FP > 0            % (Positive Predictive Values):
    m.precision   = TP / (TP + FP);
  else
    m.precision   = NaN;
  end
s
  if ~isnan(m.precision) && ~isnan(m.sensitivity) && (m.precision + m.sensitivity > 0)
    m.f1        = 2 * (m.precision * m.sensitivity) / (m.precision + m.sensitivity);
  else
    m.f1        = NaN;
  end
end

