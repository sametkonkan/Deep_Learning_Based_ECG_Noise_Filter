% Noise generation algorithm for ECG Records
function noise = kurtosis_noise_generator(N)
    % It produces noise with high 'kurtosis' value.
    % Peaked distributions, such as the Laplace distribution, have high kurtosis values.
    
    mu = 0;  % average
    sigma = 1; % Standart deviation
    noise = sigma * sign(randn(N,1)) .* log(1 + abs(randn(N,1))) + mu;
end