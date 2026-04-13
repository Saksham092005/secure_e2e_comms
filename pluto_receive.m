%% pluto_receive.m
% Receive transmitted symbols via ADALM-Pluto SDR
%
% Workflow:
%   1. Receive signal from Pluto
%   2. Demodulate and extract symbols
%   3. Save to rx_symbols.txt
%   4. Python decoder reads this file
%
% Output file format: rx_symbols.txt
%   Each line: real,imag (e.g., 0.523,-0.156)

clear all;
close all;
clc;

%% ══════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════

% File paths
RX_FILE = 'rx_symbols.txt';

% SDR Parameters (MUST MATCH TRANSMITTER)
centerFreq = 915e6;         % 915 MHz
sampleRate = 1e6;           % 1 MHz
rxGain = 60;                % Receive gain (dB) - high for sensitivity

% Modulation Parameters (MUST MATCH TRANSMITTER)
samplesPerSymbol = 100;     % Oversampling factor
expectedSymbols = 6;        % Expected number of symbols (2*n * repetitions = 2*3 = 6)

% Reception Parameters
captureSeconds = 1.0;       % How long to receive (seconds)
samplesPerFrame = round(captureSeconds * sampleRate);

%% ══════════════════════════════════════════════════════════════════════
%  RECEIVE VIA ADALM-PLUTO
%% ══════════════════════════════════════════════════════════════════════

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' PLUTO SDR RECEIVER\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Configuring ADALM-Pluto receiver...\n');

try
    % Create receiver object
    rx = sdrrx('Pluto', ...
        'RadioID', 'usb:0', ...
        'CenterFrequency', centerFreq, ...
        'GainSource', 'Manual', ...
        'Gain', rxGain, ...
        'BasebandSampleRate', sampleRate, ...
        'OutputDataType', 'double', ...
        'SamplesPerFrame', samplesPerFrame);
    
    fprintf('  ✓ Pluto receiver configured\n');
    fprintf('    Center Frequency: %.2f MHz\n', centerFreq/1e6);
    fprintf('    Sample Rate: %.2f MHz\n', sampleRate/1e6);
    fprintf('    RX Gain: %d dB\n', rxGain);
    fprintf('    Capture duration: %.2f seconds\n', captureSeconds);
    
    % Receive
    fprintf('\n[2] Receiving signal...\n');
    rxSignal = rx();
    
    fprintf('  ✓ Received %d samples\n', length(rxSignal));
    
    % Release hardware
    release(rx);
    
catch ME
    fprintf('  ✗ ERROR during reception:\n');
    fprintf('    %s\n', ME.message);
    
    if exist('rx', 'var')
        release(rx);
    end
    
    rethrow(ME);
end

%% ══════════════════════════════════════════════════════════════════════
%  SIGNAL PROCESSING & SYMBOL EXTRACTION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[3] Extracting symbols...\n');

% Simple downsampling approach (take every Nth sample)
% In a real system, you'd use matched filtering, timing recovery, etc.
% But for same-room transmission, this simple approach should work

% Downsample by taking every samplesPerSymbol-th sample
symbolIndices = 1:samplesPerSymbol:length(rxSignal);
extractedSymbols = rxSignal(symbolIndices);

% Trim to expected length
if length(extractedSymbols) > expectedSymbols
    extractedSymbols = extractedSymbols(1:expectedSymbols);
end

fprintf('  ✓ Extracted %d symbols (expected %d)\n', ...
        length(extractedSymbols), expectedSymbols);

% Power normalization (should roughly match TX power)
rxPower = mean(abs(extractedSymbols).^2);
fprintf('  Received symbol power: %.4f\n', rxPower);

% Display first few symbols
fprintf('\n  Symbol preview (first 4):\n');
for i = 1:min(4, length(extractedSymbols))
    fprintf('    Symbol %d: %+.4f %+.4fj (mag: %.4f)\n', ...
            i, real(extractedSymbols(i)), imag(extractedSymbols(i)), ...
            abs(extractedSymbols(i)));
end

%% ══════════════════════════════════════════════════════════════════════
%  SAVE TO FILE FOR PYTHON DECODER
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[4] Saving symbols to %s...\n', RX_FILE);

% Open file for writing
fileID = fopen(RX_FILE, 'w');

% Write each symbol as real,imag
for i = 1:length(extractedSymbols)
    fprintf(fileID, '%.6f,%.6f\n', ...
            real(extractedSymbols(i)), imag(extractedSymbols(i)));
end

fclose(fileID);

fprintf('  ✓ Saved %d symbols to file\n', length(extractedSymbols));

%% ══════════════════════════════════════════════════════════════════════
%  VISUALIZATION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[5] Generating plots...\n');

figure('Position', [100, 100, 1200, 400]);

% Time domain
subplot(1,3,1);
plot(real(rxSignal(1:min(10000, length(rxSignal)))));
title('Received Signal (Real Part)');
xlabel('Sample');
ylabel('Amplitude');
grid on;

% Constellation
subplot(1,3,2);
plot(real(extractedSymbols), imag(extractedSymbols), 'bo', 'MarkerSize', 10);
title('Received Symbol Constellation');
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
grid on;
axis equal;

% Power spectrum
subplot(1,3,3);
spectrum = fftshift(abs(fft(rxSignal(1:min(10000, length(rxSignal))))));
freqs = linspace(-sampleRate/2, sampleRate/2, length(spectrum));
plot(freqs/1e6, 20*log10(spectrum));
title('Power Spectrum');
xlabel('Frequency (MHz)');
ylabel('Power (dB)');
grid on;

%% ══════════════════════════════════════════════════════════════════════
%  SUMMARY
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' RECEPTION SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Samples received: %d\n', length(rxSignal));
fprintf('  Symbols extracted: %d\n', length(extractedSymbols));
fprintf('  Saved to: %s\n', RX_FILE);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Reception complete. Run Python decoder now:\n');
fprintf('  python -c "from hardware_utils import decode_from_reception; decode_from_reception()"\n\n');
