%% Perceptron and ADALINE Alphabet Classifier
% Reconstructed from uploaded assignment PDF and cleaned into a runnable script.
% Requires alphabet.mat in the current folder.

clear; clc;

if ~isfile('alphabet.mat')
    error('alphabet.mat not found. Place alphabet.mat in the same folder as this script.');
end

load('alphabet.mat'); % expected variable: X_bipolar

target_E = 5;
target_F = 6;

E = reshape(X_bipolar(target_E, :), [7, 5]);
F = reshape(X_bipolar(target_F, :), [7, 5]); %#ok<NASGU>

num_inputs = 7 * 5;
epochs = 10;

%% Part 1 - Perceptron
alpha = 0.1;
theta = 0.5;

weights = randn(num_inputs, 1);
bias = randn();

labels = -ones(size(X_bipolar, 1), 1);
labels(target_E) = 1;
labels(target_F) = -1;

for epoch = 1:epochs
    idx = randperm(size(X_bipolar, 1));
    for i = idx
        x = X_bipolar(i, :)';
        desired = labels(i);
        net = weights' * x + bias;
        output = -1;
        if net >= theta
            output = 1;
        end
        error_term = desired - output;
        weights = weights + alpha * error_term * x;
        bias = bias + alpha * error_term;
    end
end

fprintf('Perceptron training complete.\n');

correct_count = 0;
for i = 1:size(X_bipolar, 1)
    x = X_bipolar(i, :)';
    net = weights' * x + bias;
    if net > theta
        output = 1;
    elseif net < -theta
        output = -1;
    else
        output = 0;
    end

    if (i == target_E && output == 1) || (i == target_F && output == -1)
        correct_count = correct_count + 1;
    end
end

fprintf('Perceptron correctly handled %d of the 2 target letters (E/F).\n', correct_count);

E_modified = E;
E_modified(end, :) = 0;
x_modified = reshape(E_modified, [], 1);

net_modified = weights' * x_modified + bias;
if net_modified > theta
    output_modified = 1;
elseif net_modified < -theta
    output_modified = -1;
else
    output_modified = 0;
end

fprintf('Perceptron output for modified E input: %d\n', output_modified);

%% Part 2 - ADALINE
alpha_values = [0.1, 1.0, 0.01];
adaline_weights = cell(numel(alpha_values), 1);
adaline_bias = zeros(numel(alpha_values), 1);

for k = 1:numel(alpha_values)
    adaline_weights{k} = randn(num_inputs, 1);
    adaline_bias(k) = randn();
end

for epoch = 1:epochs
    idx = randperm(size(X_bipolar, 1));
    for ii = 1:length(idx)
        i = idx(ii);
        x = X_bipolar(i, :)';
        desired = labels(i);
        for k = 1:numel(alpha_values)
            net = adaline_weights{k}' * x + adaline_bias(k);
            err = desired - net;
            adaline_weights{k} = adaline_weights{k} + alpha_values(k) * err * x;
            adaline_bias(k) = adaline_bias(k) + alpha_values(k) * err;
        end
    end
end

threshold = 0;
for k = 1:numel(alpha_values)
    net_modified = adaline_weights{k}' * x_modified + adaline_bias(k);
    output_modified = net_modified >= threshold;
    fprintf('ADALINE output for modified E input (alpha = %.2f): %.4f, thresholded = %d\n', ...
        alpha_values(k), net_modified, output_modified);
end

%% Part 3 - Extra Credit: XOR Backpropagation
inputSize = 2;
hiddenSize = 2;
outputSize = 1;
learningRate = 0.1;

hiddenWeights = rand(hiddenSize, inputSize);
hiddenBias = rand(hiddenSize, 1);
outputWeights = rand(outputSize, hiddenSize);
outputBias = rand(outputSize, 1);

inputData = [0 0; 0 1; 1 0; 1 1];
targetOutput = [0; 1; 1; 0];

numEpochs = 10000;

for epoch = 1:numEpochs
    totalError = 0;
    for i = 1:size(inputData, 1)
        x = inputData(i, :)';
        target = targetOutput(i);

        hiddenInput = hiddenWeights * x + hiddenBias;
        hiddenOutput = 1 ./ (1 + exp(-hiddenInput));

        outputInput = outputWeights * hiddenOutput + outputBias;
        networkOutput = 1 ./ (1 + exp(-outputInput));

        err = target - networkOutput;
        totalError = totalError + abs(err);

        dOutput = networkOutput .* (1 - networkOutput) .* err;
        dHidden = hiddenOutput .* (1 - hiddenOutput) .* (outputWeights' * dOutput);

        outputWeights = outputWeights + learningRate * dOutput * hiddenOutput';
        outputBias = outputBias + learningRate * dOutput;
        hiddenWeights = hiddenWeights + learningRate * dHidden * x';
        hiddenBias = hiddenBias + learningRate * dHidden;
    end

    if totalError < 0.01
        break;
    end
end

newInput = [0; 1];
hiddenInput = hiddenWeights * newInput + hiddenBias;
hiddenOutput = 1 ./ (1 + exp(-hiddenInput));
outputInput = outputWeights * hiddenOutput + outputBias;
predictedOutput = 1 ./ (1 + exp(-outputInput));

fprintf('Predicted XOR output for [0;1]: %.4f\n', predictedOutput);
