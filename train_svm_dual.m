function [v, gamma] = train_svm_dual(X, y, C)
  m = size(X,1);
  K = X * X';

  % Quadratic Programming formulation
  H = (y * y') .* K + 1e-10 * eye(m);  % Add small regularization to ensure PSD
  f = -ones(m,1);
  Aeq = y';
  beq = 0;
  lb = zeros(m,1);
  ub = C * ones(m,1);

  opts = optimset('Algorithm','interior-point-convex', 'Display','off');

  % optimization, finds the lagrange multipliers (α) for the dual SVM problem
  alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);

  % Support vectors are the training points that define the decision boundary
  sv_count = sum(alpha > 1e-6 & alpha < C-1e-6);
  fprintf('   Found %d support vectors (C=%.3g)\n', sv_count, C);

  % Recover primal vector v
  v = X' * (alpha .* y);      % This is the normal vector to the decision hyperplane

  % Support vectors
  % Calculates the bias term γ using the average distance of support vectors to the decision boundary
  sv = find(alpha > 1e-5 & alpha < C - 1e-5);
  if isempty(sv)
    gamma = 0;  % fallback if no valid SVs
  else
    gamma = mean(y(sv) - X(sv,:) * v);
  end
end

fprintf("  #SVs: %d  (for C=%.3g)\n", sum(alpha > 1e-6), C);


