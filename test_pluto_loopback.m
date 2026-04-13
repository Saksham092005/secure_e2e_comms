%% Continuous Transmission + Reception Test
clear all;
close all;
clc;

%% Parameters
centerFreq = 915e6;
sampleRate = 1e6;
txGain = 0;
rxGain = 50;  % Even higher RX gain

%% Create TX and RX objects
tx = sdrtx('Pluto', ...
    'RadioID', 'usb:0', ...
    'CenterFrequency', centerFreq, ...
    'Gain', txGain, ...
    'BasebandSampleRate', sampleRate);

rx = sdrrx('Pluto', ...
    'RadioID', 'usb:0', ...
    'CenterFrequency', centerFreq, ...
    'GainSource', 'Manual', ...
    'Gain', rxGain, ...
    'BasebandSampleRate', sampleRate, ...
    'OutputDataType', 'double', ...
    'SamplesPerFrame', 10000);

%% Generate continuous signal
duration = 1.0;
t = 0:1/sampleRate:duration-1/sampleRate;
frequency = 100e3;
txSignal = 0.9 * exp(1j*2*pi*frequency*t).';

fprintf('Starting continuous transmission...\n');

%% Start continuous transmission in background
% We'll transmit multiple times to create continuous signal
for i = 1:3
    tx(txSignal);
    fprintf('TX burst %d sent\n', i);
    pause(0.1);
end

fprintf('Switching to RX...\n');
pause(0.2);

%% Now receive
rxSignal = rx();

fprintf('Received data\n');

%% Stop and release
release(tx);
release(rx);

%% Analysis
txPower = mean(abs(txSignal).^2);
rxPower = mean(abs(rxSignal).^2);

fprintf('\n=== RESULTS ===\n');
fprintf('TX Power: %.6f\n', txPower);
fprintf('RX Power: %.6e\n', rxPower);

%% Plot
figure;
subplot(2,1,1);
plot(abs(txSignal(1:1000)));
title('TX Signal Magnitude');
grid on;

subplot(2,1,2);
plot(abs(rxSignal(1:1000)));
title('RX Signal Magnitude');
grid on;

if rxPower > 1e-3
    fprintf('✓ Strong signal detected!\n');
elseif rxPower > 1e-5
    fprintf('⚠ Weak signal detected - getting closer!\n');
else
    fprintf('✗ Still too weak\n');
end