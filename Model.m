function [W, V, H, b] = Model(Xtr, Y, opts);
%% Initialize W, V, and H

rng
for m = 1:length(Xtr)
    X{m} = Xtr{m}'; %% p*n
    V{m} = ones(size(X{m}, 1), opts.h); %% p*h
    O{m} = eye(size(X{m}, 2)); %% n*n
    a = V{m}'*X{m};
    idx = find(isnan(a(1,:)));
    O{m}(idx,idx) = 0;

end
rng default
H = randn(opts.h, size(X{m},2));
C = max(Y);
W = ones(opts.h, 1);
b = 0;
P1 = zeros(opts.h, 1);
P2 = zeros(length(Y),1);
Z = P2;
Q = W;

opts.mu = 1;
rho = 1.1;
obj = 1 - Y'*(H'*W + b) + opts.lambda * norm(W) ;
maxmu = 1e3;
for iter = 1:opts.maxIter
    obj_old = obj;
    for iter1 = 1:opts.maxIter
        V_old = V;
        W_old = W;
        %% Upate V
        for m = 1:length(X)
            D = UpdateD(V{m});
            idx = find(diag(O{m}) == 0);
            A = X{m};
            A(:,idx) = 0;
            V{m} = (A*A' + opts.gamma/opts.beta * D + eps) \ (A*H');
            V{m} = V{m} ./ norm(H);
        end
    
    %% Update W
        W = (opts.lambda + opts.mu)^(-1) * (P1 + opts.mu*Q);
        for m = 1:length(Xtr)
            tv(m) = norm(V{m} - V_old{m}, 'fro') / max(norm(V_old{m}, 'fro'), 1);
        end
        tv1 = mean(tv);
        tw = norm(W - W_old) ./max(norm(W_old),1);
        fprintf('iter = %d, tv1 = %5.8f, tw = %5.8f\n', iter,tv1, tw)
        if tv1 < opts.tol & tw<opts.tol
            break
        end
    end

    %% Update H
    A1 = zeros(size(H,1), size(H,2));
    for m = 1:length(Xtr)
        A =  V{m}' * X{m};
        idx = find(diag(O{m}) == 0);
        A(:,idx) = 0;
        A1 = A1 + A;
    end
    F = 2*opts.beta/opts.mu * length(Xtr) * eye(opts.h) + Q*Q';
    E = 2*opts.beta/opts.mu * A1 - Q*(Z - Y + ones(length(Y),1) * b' + P2/opts.mu)';
    H = F \ E;

    %% Update Q
    
    Q = (eye(opts.h) + H*H') \ (W - P1/opts.mu - H * (Z - Y + ones(length(Y),1) * b' + P2/opts.mu));

    %% Update Z
    S = Y - H'*Q - ones(length(Y),1) * b' - P2/opts.mu;
    Omega = max(Y.*S, 0);
    idx = find(Omega >0);
    Omega(idx) = 1;
    Omega1 = min(Y.*S, 0);
    idx = find(Omega1 < 0);
    Omega1(idx) = 1;
    Z = Omega.* S / (1+2/opts.mu) + Omega1 .* S;

    %% Update b
    
    b = 1/length(Y) * (Y - Z - H'*Q - P2/opts.mu)' * ones(length(Y),1);
    
    %% Update P1 and P2
    P1 = P1 + opts.mu * (Q - W);
    P2 = P2 + opts.mu * (Z - Y + ones(length(Y),1) * b' + H'*Q);
    opts.mu = min(opts.mu*rho, maxmu);

    for m = 1:length(Xtr)
        a = V{m}'*X{m} - H;
        idx = find(isnan(a(1,:)));
        a(:, idx) = 0;
        error(m) = norm(a);
        for j = 1:size(V{m},1)
            a(j) = norm(V{m}(j,:));
        end
        a1(m) = norm(a, 1);
    end
    obj = max(1 - Y'*(H'*W + b),0)^2  + opts.lambda * norm(W) + opts.beta * sum(error) + opts.gamma * sum(a1);
    error = abs(obj - obj_old) / abs(obj_old);
    fprintf('iter = %d, error = %5.8f\n', iter, error)
    if error < opts.tol
        break
    end
end



end