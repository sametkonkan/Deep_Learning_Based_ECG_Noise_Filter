%% Deep Learning-Based ECG Noise Filtering


%% 0. Add utils path
addpath(genpath('C:\Users\pc\Desktop\Deep_Learning_Based_ECG_Noise_Filter\src\utils'));


%% 1. WFDB Toolbox ve Adjusting Data Paths
clear all;
clc;

%  Add Data Paths for  WFDB Toolbox 
wfdb_path = 'C:\Users\pc\Downloads\wfdb-app-toolbox-0-10-0\mcode';
addpath(wfdb_path);
wfdbloadlib; % Start WFDB Toolbox

% Set data folder (MIT-BIH records should be here)
data_folder = 'C:\Users\pc\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0';
cd(data_folder); % Critical: Go to folder

%% 2. Processing All Records
records = arrayfun(@(x) num2str(x), [100:124, 200:234], 'UniformOutput', false); % Records 100-119 &  200-234

for rec_idx = 1:length(records)
    current_record = records{rec_idx};
    fprintf('\nProcessing Record: %s\n', current_record);
    
    %% 2.1 Loading ECG Data
    try
        [ecg_clean, Fs, tm] = rdsamp(current_record, 1);
    catch ME
        fprintf('Failed to load record: %s\n', current_record);
        continue; % Go to next record
    end


%% 3. Adding Artificial Noise
t = (0:length(ecg_clean)-1)/Fs;
rng(0); % Random seed for repeatability

% Basal drift (0.5 Hz)
baseline_wander = 0.3 * sin(2*pi*0.5*t)';

% 50 Hz güç hattı girişimi
powerline_noise = 0.2 * sin(2*pi*50*t + pi/4)';

% EMG noise (high frequency)
emg_noise = 0.4 * randn(size(ecg_clean));

% Sum of all noises
noisy_ecg = ecg_clean + baseline_wander + powerline_noise + emg_noise;

%% 3.1 Adding Realistic Motion Artifacts
motion_noise = zeros(size(ecg_clean));
random_peaks = randi([1 length(ecg_clean)], 1, 20); % 20 random motion artifacts
for i = 1:length(random_peaks)
    start_idx = max(1, random_peaks(i)-10);
    stop_idx = min(length(ecg_clean), random_peaks(i)+10);
    motion_noise(start_idx:stop_idx) = 0.5 * randn(size(motion_noise(start_idx:stop_idx)));
end
noisy_ecg = noisy_ecg + motion_noise; % Add to sum of all noises

%% 3.2 Advanced Noise Modeling
noise_types = {'EMG', 'Baseline', 'Powerline', 'Motion'};
for i = 1:length(noise_types)
    switch noise_types{i}
        case 'EMG'
            augmented_noise = 0.5 * randn(size(ecg_clean));
        case 'Baseline'
            augmented_noise = 0.7 * sin(2*pi*0.3*t)';
        case 'Powerline'
            augmented_noise = 0.4 * sin(2*pi*60*t + rand*pi)';
        case 'Motion'
            augmented_noise = 0.6 * kurtosis_noise_generator(length(ecg_clean));
    end
end


%% 4. Digital Filtering
% Bandpass Filter (0.5-45 Hz)
order = 4;
fc_low = 1; % HPF cutoff
fc_high = 35; % LPF cutoff
[b, a] = butter(order, [fc_low, fc_high]/(Fs/2), 'bandpass');
filtered_ecg = filtfilt(b, a, noisy_ecg);

% 50 Hz notch filtre
wo = 50/(Fs/2);
bw = wo/50;
[bn, an] = iirnotch(wo, bw);
filtered_ecg = filtfilt(bn, an, filtered_ecg);

%% 4.1 LMS Adaptive Filter Application
order_adapt = 32; % Filter order
mu = 0.001;     % Learning coefficient
lms_filter = dsp.LMSFilter(order_adapt, 'StepSize', mu);
[~, filtered_ecg_adaptif, ~] = lms_filter(noisy_ecg, ecg_clean); %!! Use clean ECG as reference signal

%% 5. Denoising and Threshold Optimization with Wavelet
% Symlets Wavelets were chosen because they are similar to ECG signals.
wavelet_name = 'sym8'; % Wavelet function to be used
level = 5;           % Start level

% Wavelet transform
[C, L] = wavedec(filtered_ecg, level, wavelet_name);

% Check C and L sizes
disp('C boyutu:'); disp(size(C));
disp('L boyutu:'); disp(size(L));

% Manuel threshold (universal threshold)
sigma = median(abs(C)) / 0.6745;           % Noise standard deviation
threshold = sigma * sqrt(2 * log(length(filtered_ecg)));  % Universal threshold

% Thresholding process (soft thresholding) - global denoising
denoised_ecg_optimized = wdencmp('gbl', C, L, wavelet_name, level, threshold, 's', 1);

%% 5.1Automatic Wavelet Level Selection
best_mse = Inf;
best_level = level;
for lev = 3:7
    [C_temp, L_temp] = wavedec(filtered_ecg, lev, wavelet_name);
    sigma_temp = median(abs(C_temp)) / 0.6745;
    threshold_temp = sigma_temp * sqrt(2*log(length(filtered_ecg)));
    denoised_temp = wdencmp('gbl', C_temp, L_temp, wavelet_name, lev, threshold_temp, 's', 1);
    current_mse = mean((ecg_clean - denoised_temp).^2);
    if current_mse < best_mse
        best_mse = current_mse;
        best_level = lev;
    end
end
level = best_level; % Use best level.

%% 5.2 Adaptive Threshold Optimization
threshold_levels = linspace(0.1, 2, 20); % Eşik aralığı
best_snr = -Inf;
for thr = threshold_levels
    current_threshold = thr * sigma * sqrt(2*log(length(filtered_ecg)));
    temp_denoised = wdencmp('gbl', C, L, wavelet_name, level, current_threshold, 's', 1);
    current_snr = 10*log10(var(ecg_clean)/var(temp_denoised - ecg_clean));
    if current_snr > best_snr
        best_snr = current_snr;
        optimal_threshold = current_threshold; 
    end
end



%% 6. Performance Metrics (SNR and MSE)
snr_before = 10*log10(var(ecg_clean)/var(noisy_ecg - ecg_clean));
snr_after_wavelet = 10*log10(var(ecg_clean)/var(denoised_ecg_optimized - ecg_clean));
mse_wavelet = mean((ecg_clean - denoised_ecg_optimized).^2);

fprintf('SNR (Noisy): %.2f dB\n', snr_before);
fprintf('SNR (Wavelet Optimized Cleaned): %.2f dB\n', snr_after_wavelet);
fprintf('MSE (Wavelet Optimized Cleaned): %.4f\n', mse_wavelet);

%% 7. Detection of QRS Complexes (Pan-Tompkins Algorithm)
% Simple QRS detection: derivative, square, moving average and thresholding
diff_ecg = diff(denoised_ecg_optimized);           % Derivative
squared_ecg = diff_ecg.^2;                           % Square
window_size = round(0.15 * Fs);                     % 150 ms 
integrated_ecg = movmean(squared_ecg, window_size);  % Moving average

threshold_qrs = 0.5 * max(integrated_ecg);          % Determining the threshold
qrs_peaks = find(integrated_ecg > threshold_qrs);     % Find QRS positions

%% 8. Corrected QRS Performance Calculation
[ann, anntype] = rdann('100', 'atr');

% Get all QRS annotations (N, V, L, R, A etc.)
valid_labels = {'N', 'L', 'R', 'V', 'A'}; % IEEE standard labels
is_valid = ismember(anntype, valid_labels);
true_qrs = ann(is_valid);

% Convert tolerance to number of samples (50 ms)
tolerance_samples = round(0.05 * Fs); 

% Performance metrics
if ~isempty(true_qrs)
    [TP, FP, FN] = compare_qrs(qrs_peaks, true_qrs, tolerance_samples);
    fprintf('Sensitivite: %.2f%%, Positive Predictive Value: %.2f%%\n', ...
        TP/(TP+FN)*100, TP/(TP+FP)*100);
else
    fprintf('Warning: No valid QRS annotation found!\n');
end

%% 9. Modified Visualization with Record-specific Titles
    figure('Name', sprintf('Record %s Analysis', current_record), ...
          'NumberTitle', 'off', 'Position', [100 100 1200 800]);
    
    subplot(4,1,1);
    plot(tm, ecg_clean); 
    title(sprintf('[%s] Original ECG', current_record)); % Record ID in title
    
    subplot(4,1,2);
    plot(tm, noisy_ecg);
    title(sprintf('[%s] Noisy ECG', current_record));
    
    subplot(4,1,3);
    plot(tm, filtered_ecg);
    title(sprintf('[%s] Filtered ECG', current_record));
    
    subplot(4,1,4);
    plot(tm, denoised_ecg_optimized);
    title(sprintf('[%s] Wavelet-Denoised ECG', current_record));
    
    %% Save Figures Automatically
    saveas(gcf, sprintf('ECG_Analysis_%s.png', current_record));
end

%% 10. Improved Pan-Tompkins Algorithm (Hilbert Based)
analytic_signal = hilbert(denoised_ecg_optimized);
envelope = abs(analytic_signal);
window_size_env = 5 * Fs;  % 5 saniyelik pencere
adaptive_threshold = movmax(envelope, window_size_env) * 0.6;
qrs_peaks_hilbert = find(envelope > adaptive_threshold);

% Visualizing Hilbert-based QRS detection:
figure;
plot(tm, denoised_ecg_optimized);
hold on;
scatter(tm(qrs_peaks_hilbert), denoised_ecg_optimized(qrs_peaks_hilbert), 'r', 'filled');
title('Hilbert-based QRS detection:');
xlabel('Zaman (second)'); ylabel('Amplitude');

%% 11. Deep Learning Model - With Model Saving/Loading
segment_length = 512;
num_segments = floor(length(ecg_clean)/segment_length);

% Split data into segments
noisy_segments = reshape(noisy_ecg(1:num_segments*segment_length), [num_segments, segment_length]);
clean_segments = reshape(ecg_clean(1:num_segments*segment_length), [num_segments, segment_length]);

% Convert to cell arrays
XTrain = num2cell(noisy_segments(1:end-10,:), 2);
YTrain = num2cell(clean_segments(1:end-10,:), 2);
XTest = num2cell(noisy_segments(end-9:end,:), 2);
YTest = num2cell(clean_segments(end-9:end,:), 2);

% Define model path
model_path = 'ecg_denoiser_net.mat';

% Check if pretrained model exists
if isfile(model_path)
    % Load pretrained model
    fprintf('Loading pretrained model...\n');
    load(model_path, 'net'); 
else
    % Define and train new model
    fprintf('Training new model...\n');
    layers = [
        sequenceInputLayer(1)
        convolution1dLayer(3, 16, 'Padding', 'same')
        bilstmLayer(64, 'OutputMode', 'sequence')
        dropoutLayer(0.3)
        fullyConnectedLayer(1)
        regressionLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {XTest, YTest}, ...
        'Plots', 'training-progress');
    
    net = trainNetwork(XTrain, YTrain, layers, options);
    
    % Save the trained model
    save(model_path, 'net');
    fprintf('Model saved to %s\n', model_path);
end

% Perform inference
denoised_dl = predict(net, XTest);

% Calculate RMSE
denoised_dl_numeric = cell2mat(denoised_dl);
YTest_numeric = cell2mat(YTest);
rmse_train = sqrt(mean((denoised_dl_numeric - YTest_numeric).^2));
fprintf('Deep Learning RMSE: %.4f\n', rmse_train);



%% 12. Automatic Report Generation
report_title = sprintf('ECG_Analysis_Report_%s.pdf', datestr(now, 'ddmmmyyyy_HHMM'));
exportgraphics(gcf, report_title, 'ContentType', 'vector', 'Append', true);

