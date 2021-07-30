% bestx: the best solution found by your algorithm
% recordedAvgY: array of  average fitnesses of each generation
% recordedBestY: array of best fitnesses of each generation
function [bestx, recordedAvgY, recordedBestY]=EA(funcName,n,lb,ub,nbEvaluation)
warning on MATLAB:divideByZero
if nargin < 5
  error('input_example :  not enough input')
end

eval(sprintf('objective=@%s;',funcName)); % Do not delete this line
% objective() is the evaluation function, for example : fitness = objective(x) 

%% Your algorithm

% The implementation of PSO partially refer to
% https://github.com/MatthewPeterKelly/ParticleSwarmOptimization/blob/master/PSO.m
% and http://www5.zzu.edu.cn/cilab/info/1025/1074.htm

%% Parameters setting
m = 50;
r_1 = 2;
r_2 = 2;
w_0 = 0.9;
w_1 = 0.4;

beta = 0.001;
eta = 1.2;

%% const setting
FES = m;
gen = 1;
genMax = ceil(nbEvaluation / m) - 1;

bestever = Inf;
bestx = NaN;
recordedAvgY = zeros(1,genMax);
recordedBestY = zeros(1,genMax);

rand('seed', sum(100 * clock));


%% PSO



X = lb + (ub-lb).*rand(n,m); % population initialization

F = arrayfun(@(i)objective(X(:, i).'), 1:m).';



X_Best = X;  % pBest in report
F_Best = F;  % record best value for each individual

[F_Global, I_Global] = min(F_Best); % Value of global best
X_Global = X(:, I_Global); % gBest in report

iwt=w_0-(1:genMax).*((w_0-w_1)./genMax); % deacrease intertia weight within searching

V_max = 0.5*(ub-lb); % define max velocity

V = -V_max+2.*V_max.*rand(n,m); % initialize velocity

g = ones(n,m);

% main loop
while(FES < nbEvaluation)
    
    %record
    if min(F) < bestever
        [bestever, i] = min(F);
        bestx = X(:, i);
    end
    recordedAvgY(gen) = mean(F);
    recordedBestY(gen) = bestever;
    
    %random matrix 
    randco1 = rand(n,m);
    randco2 = rand(n,m);
    
    %individual loop
    for idx = 1:m   
        
        delta = r_1*randco1(:,idx).*(X_Global-X(:,idx)) + ...  % Global direction
                r_2*randco2(:,idx).*(X_Best(:,idx)-X(:,idx));    % Local best direction
        delta_mean = repmat(mean(delta),[n 1]);
        delta_std = repmat(std(delta),[n 1]);
        delta_std(delta_std==0) = 1; % avoid divide 0
        
        g(:,idx) = (1-beta).*g(:,idx) ...,
                    + beta.*((delta-delta_mean)./(delta_std) + 1).^2; % Second order moment
        %Update velocity        
        V(:,idx) = iwt(gen).*V(:,idx) ...,    % momentum
                 + r_1*randco1(:,idx).*(X_Global-X(:,idx)) ...,  % Global direction
                 + r_2*randco2(:,idx).*(X_Best(:,idx)-X(:,idx));    % Local best direction
        
        % limit V by V_min and V_max
        V(:,idx) = min(V(:,idx),V_max);
        V(:,idx) = max(V(:,idx),-V_max);
        
        X(:,idx) = X(:,idx) + ...,
                   eta*(1./(sqrt(g(:,idx))+1e-8)).*V(:,idx);  % Update position with adaptive learning rate

        
        % re-initialize if out-of-bound
        X(:,idx) = ((X(:,idx)>=lb)&(X(:,idx)<=ub)).*X(:,idx) ...,
                 +(X(:,idx)<lb).*(lb+0.25.*(ub-lb).*rand(n,1)) ...,
                 +(X(:,idx)>ub).*(ub-0.25.*(ub-lb).*rand(n,1));
        
        F(idx) = objective(X(:,idx).');   %Evaluate

        [F_Best(idx), iBest] = min([F(idx), F_Best(idx)]);
        if iBest == 1  %new point is better than pBest
            X_Best(:,idx) = X(:,idx);
            [F_Global, iBest] = min([F_Best(idx), F_Global]);
            if iBest == 1 %new point is better than the global best
                X_Global = X_Best(:,idx);
            end
        end

    end

    FES = FES + m;

    gen = gen + 1;
end

end
