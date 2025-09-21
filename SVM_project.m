clear all; close all; clc;

addpath('D:/OptimizationProject');

pkg load optim statistics

% load dataset .mat file
load('dataset29.mat');
whos;

% Ensure y is a column vector (Nx1)
if size(y, 2) > 1
  y = y';
end

% Fixing random seed for reproducibility
rand('seed', 42)

% === Remap labels to ±1 ===
classes = unique(y);
if ~isequal(classes, [-1; 1])
  y_new = zeros(size(y));
  y_new(y == classes(1)) = -1;
  y_new(y ~= classes(1)) = +1;
  y = y_new;
end

% ============Standardize features to zero mean, unit variance ========
mu    = mean(X);
sigma = std(X);
X     = (X - mu) ./ sigma;

% =====================================================================
% Step 1: Bilevel Cross-Validation Setup
% =====================================================================
k_outer = 10;
k_inner = 5;
C_values = logspace(-3, 3, 10);  % Candidate C values
C_strs = arrayfun(@(c) num2str(c), C_values, 'UniformOutput', false);
fprintf('Candidate C values: %s\n', strjoin(C_strs, ', '));

% Use label vector 'y' in cvpartition to get correct boolean masks
cv_outer = cvpartition(y, 'KFold', k_outer);

% Preallocate performance metrics
train_correctness = zeros(k_outer,1);
train_sensitivity = zeros(k_outer,1);
train_specifity   = zeros(k_outer,1);
train_precision   = zeros(k_outer,1);
train_f1          = zeros(k_outer,1);

test_correctness = zeros(k_outer,1);
test_sensitivity = zeros(k_outer,1);
test_specifity   = zeros(k_outer,1);
test_precision   = zeros(k_outer,1);
test_f1          = zeros(k_outer,1);

% ============================================================
% Outer cross-validation loop
% ============================================================
for i = 1:k_outer
  fprintf('\n--- Outer Fold %d/%d ---\n', i, k_outer);

  % Get boolean masks for train/test split
  train_idx = training(cv_outer, i);
  test_idx  = test(cv_outer, i);

  fprintf(" Fold %d: train samples = %d, test samples = %d\n", ...
         i, sum(train_idx), sum(test_idx));


  % Split data
  X_train = X(train_idx, :);
  y_train = y(train_idx);
  X_test  = X(test_idx, :);
  y_test  = y(test_idx);

  % ------------------------------------------------------------
  % Inner loop for model selection
  % ------------------------------------------------------------
  n_train = length(y_train);
  cv_inner = cvpartition(y_train, 'KFold', k_inner);
  mean_accuracy = zeros(length(C_values), 1);

  for c_idx = 1:length(C_values)
    C = C_values(c_idx);
    acc_inner = zeros(k_inner, 1);

    for j = 1:k_inner
      tr = training(cv_inner, j);
      vl = test(cv_inner, j);

      Xt = X_train(tr,:);  yt = y_train(tr);
      Xv = X_train(vl,:);  yv = y_train(vl);

      [v_tmp, g_tmp] = train_svm_dual(Xt, yt, C);
      preds = sign(Xv * v_tmp - g_tmp);
      acc_inner(j) = mean(preds == yv);
    end

    mean_accuracy(c_idx) = mean(acc_inner);
  end

  fprintf(" mean_accuracy = [%s]\n", num2str(mean_accuracy',' %5.3f'));


  [~, best_idx] = max(mean_accuracy);
  C_best = C_values(best_idx);
  fprintf('Selected C = %g\n', C_best);

  [v_final, gamma_final] = train_svm_dual(X_train, y_train, C_best);
  y_pred_train = sign(X_train * v_final - gamma_final);
  y_pred_test  =  sign(X_test * v_final - gamma_final);

  %train_correctness(i) = mean(y_pred_train == y_train);

  % Compute Training metrics
  TP = sum((y_pred_train == 1) & (y_train == 1));
  TN = sum((y_pred_train == -1) & (y_train == -1));

  FP = sum((y_pred_train == 1) & (y_train == -1));
  FN = sum((y_pred_train == -1) & (y_train == 1));

  m = compute_metrics(TP, TN, FP, FN);
  train_correctness(i)  = m.accuracy;
  train_sensitivity(i)  = m.sensitivity;
  train_specifity(i)    = m.specificity;
  train_precision(i)    = m.precision;
  train_f1(i)           = m.f1;


  % Compute Test metrics
  TP = sum((y_pred_test == 1) & (y_test == 1));
  TN = sum((y_pred_test == -1) & (y_test == -1));

  FP = sum((y_pred_test == 1) & (y_test == -1));
  FN = sum((y_pred_test == -1) & (y_test == 1));

  m_test = compute_metrics(TP, TN, FP, FN);
  test_correctness(i)  = m_test.accuracy;
  test_sensitivity(i)  = m_test.sensitivity;
  test_specifity(i)    = m_test.specificity;
  test_precision(i)    = m_test.precision;
  test_f1(i)           = m_test.f1;

  % ------------------------------------------------------------
  % Plot hyperplane and margins
  % ------------------------------------------------------------
  if size(X_train, 2) == 2
    figure, hold on;
    scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 'bo');
    scatter(X_train(y_train == -1, 1), X_train(y_train == -1, 2), 'r+');

    x_range = linspace(min(X_train(:,1)), max(X_train(:,1)), 200);
    epsilon = 1e-5;

    if abs(v_final(2)) > epsilon
      decision = -(v_final(1)*x_range - gamma_final)/v_final(2);
      margin_p = -(v_final(1)*x_range - gamma_final - 1)/v_final(2);
      margin_n = -(v_final(1)*x_range - gamma_final + 1)/v_final(2);
    else
      x0 = gamma_final / v_final(1);
      decision = x0 * ones(size(x_range));
      offset = 1 / norm(v_final);
      margin_p = decision + offset;
      margin_n = decision - offset;
    end

    plot(x_range, margin_p, 'k--', 'LineWidth', 1);
    plot(x_range, margin_n, 'k--', 'LineWidth', 1);
    plot(x_range, decision, 'm-', 'LineWidth', 3);

    title(sprintf('Outer Fold %d', i));
    legend('Class +1', 'Class -1', 'H+', 'H-', 'H', 'Location', 'southeast');
    axis tight; grid on;
    saveas(gcf, sprintf('fold_%d.png', i));
    hold off;
  end
end

% =====================================================================
% Final Report
% =====================================================================
%fprintf('\nAverage Training Correctness: %.4f\n', mean(train_correctness));
%fprintf('Average Training Sensitivity: %.4f\n', mean(train_sensitivity));

avg_train_acc    = mean(train_correctness,  'omitnan');
avg_train_sens   = mean(train_sensitivity,  'omitnan');
avg_train_spec   = mean(train_specifity,    'omitnan');
avg_train_prec   = mean(train_precision,    'omitnan');
avg_train_f1     = mean(train_f1,           'omitnan');

fprintf('Avg Train Accuracy   = %.4f\n', avg_train_acc);
fprintf('Avg Train Sensitivity= %.4f\n', avg_train_sens);
fprintf('Avg Train Specificity= %.4f\n', avg_train_spec);
fprintf('Avg Train Precision  = %.4f\n', avg_train_prec);
fprintf('Avg Train F1 Score   = %.4f\n', avg_train_f1);

fprintf('\nAverage Test Metrics:\n');
fprintf('Accuracy    : %.4f\n', mean(test_correctness,  'omitnan'));
fprintf('Sensitivity : %.4f\n', mean(test_sensitivity,  'omitnan'));
fprintf('Specificity : %.4f\n', mean(test_specifity,    'omitnan'));
fprintf('Precision   : %.4f\n', mean(test_precision,    'omitnan'));
fprintf('F1-Score    : %.4f\n', mean(test_f1,           'omitnan'));



