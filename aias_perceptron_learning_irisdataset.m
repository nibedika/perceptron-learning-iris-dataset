clc; clear; close all;

% --- Perceptron Learning on Iris Dataset (Setosa vs Versicolor) Fully Verbose & Animated ---

% Read CSV (species column is text)
data = readcell('iris.csv');  % Use readcell

% First 100 samples (Setosa and Versicolor)
X = cell2mat(data(1:100,1:2));     % Features
y_labels = data(1:100,5);          % Species names

% Encode targets: Setosa = -1, Versicolor = +1
y = zeros(size(y_labels));
for i = 1:length(y_labels)
    if strcmp(y_labels{i}, 'Iris-setosa')
        y(i) = -1;
    else
        y(i) = 1;
    end
end

% Transpose for column-wise inputs
P = X'; % 2x100
T = y;  % 1x100

% Initialize weights and bias
rng('shuffle');          % Random seed
W = rand(1,2)*2 - 1;    % Random weights between -1 and 1
b = rand()*2 - 1;       % Bias
eta = 0.01;             % Learning rate
max_epochs = 50;

% Plot input points
figure; hold on; grid on;
plot(P(1,T==1), P(2,T==1),'ro','MarkerSize',8,'LineWidth',2); % Versicolor
plot(P(1,T==-1), P(2,T==-1),'bx','MarkerSize',8,'LineWidth',2); % Setosa
xlabel('Sepal Length'); ylabel('Sepal Width');
title('Perceptron Learning - Iris Setosa vs Versicolor');

% Perceptron training
for epoch = 1:max_epochs
    errors = 0;
    fprintf('--- Epoch %d ---\n', epoch);
    
    for i = 1:size(P,2)
        xi = P(:,i)';
        ti = T(i);
        
        % Compute output
        oi = sign(W*xi' + b);
        if oi == 0
            oi = -1; % Treat 0 as -1
        end
        
        % Print detailed calculation
        fprintf('Input %d: xi = [%.2f %.2f], t_i = %d, o_i = %d\n', i, xi(1), xi(2), ti, oi);
        
        % Update weights if wrong
        if oi ~= ti
            delta_W = eta*(ti - oi)*xi;
            delta_b = eta*(ti - oi);
            W = W + delta_W;
            b = b + delta_b;
            errors = errors + 1;
            
            fprintf('  ΔW = [%.4f %.4f], Δb = %.4f --> Updated W = [%.4f %.4f], b = %.4f\n', ...
                delta_W(1), delta_W(2), delta_b, W(1), W(2), b);
            
            % Animate decision boundary
            x_plot = min(P(1,:))-0.5:0.01:max(P(1,:))+0.5;
            y_plot = -(W(1)*x_plot + b)/W(2);
            h = plot(x_plot, y_plot, 'g--', 'LineWidth', 1.5);
            pause(0.3);  % Pause to visualize
            delete(h);   % Remove previous line
        end
    end
    
    fprintf('End of Epoch %d: W = [%.4f %.4f], b = %.4f, errors = %d\n\n', epoch, W(1), W(2), b, errors);
    
    % Stop if no errors
    if errors == 0
        fprintf('Training converged at epoch %d!\n', epoch);
        break;
    end
end

% Plot final decision boundary
x_plot = min(P(1,:))-0.5:0.01:max(P(1,:))+0.5;
y_plot = -(W(1)*x_plot + b)/W(2);
plot(x_plot, y_plot, 'k-', 'LineWidth',2);
legend('Versicolor (+1)', 'Setosa (-1)', 'Decision Boundary');
