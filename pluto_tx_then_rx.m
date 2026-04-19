%% pluto_tx_rx_corr.m
% TX-then-RX with correlation-based symbol detection
% Uses cross-correlation to find transmitted signal in received samples

clear all;
close all;
clc;

%% ══════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════

TX_FILE = 'tx_symbols.txt';
RX_FILE = 'rx_symbols.txt';

centerFreq = 915e6;
sampleRate = 1e6;
txGain = -10;        % Increased TX power
rxGain = 60;

samplesPerSymbol = 100;
expectedSymbols = 6;

transmissionDuration = 1.0;  % Short burst
receptionDuration = 2.0;     % Longer capture window

%% ══════════════════════════════════════════════════════════════════════
%  LOAD AND PREPARE TX SIGNAL
%% ══════════════════════════════════════════════════════════════════════

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' PLUTO TX-RX WITH CORRELATION DETECTION\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Loading symbols...\n');

if ~isfile(TX_FILE)
    error('Error: %s not found!', TX_FILE);
end

fileID = fopen(TX_FILE, 'r');
symbolData = textscan(fileID, '%f,%f');
fclose(fileID);

symbols = symbolData{1} + 1j * symbolData{2};
fprintf('  ✓ Loaded %d symbols\n', length(symbols));

% Add preamble (known pattern for correlation)
preamble = [1+1j, 1-1j, -1+1j, -1-1j];  % QPSK-like pattern
symbols_with_preamble = [preamble(:); symbols(:)];

fprintf('  ✓ Added %d-symbol preamble\n', length(preamble));

% Upsample
txSignal = repelem(symbols_with_preamble, samplesPerSymbol);

% Normalize
power = mean(abs(txSignal).^2);
txSignal = txSignal * sqrt(0.64 / power);

fprintf('  ✓ TX signal: %d samples\n', length(txSignal));

%% ══════════════════════════════════════════════════════════════════════
%  TRANSMIT
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[2] Transmitting...\n');

tx = sdrtx('Pluto', ...
    'RadioID', 'usb:0', ...
    'CenterFrequency', centerFreq, ...
    'Gain', txGain, ...
    'BasebandSampleRate', sampleRate);

tic;
txCount = 0;
while toc < transmissionDuration
    tx(txSignal);
    txCount = txCount + 1;
end

fprintf('  ✓ Transmitted %d times in %.2f sec\n', txCount, toc);

release(tx);
pause(0.01);  % Tiny delay

%% ══════════════════════════════════════════════════════════════════════
%  RECEIVE
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[3] Receiving...\n');

samplesPerFrame = round(receptionDuration * sampleRate);

rx = sdrrx('Pluto', ...
    'RadioID', 'usb:0', ...
    'CenterFrequency', centerFreq, ...
    'GainSource', 'Manual', ...
    'Gain', rxGain, ...
    'BasebandSampleRate', sampleRate, ...
    'OutputDataType', 'double', ...
    'SamplesPerFrame', samplesPerFrame);

rxSignal = rx();
release(rx);

fprintf('  ✓ Received %d samples\n', length(rxSignal));
fprintf('  ✓ RX Power: %.6f\n', mean(abs(rxSignal).^2));

%% ══════════════════════════════════════════════════════════════════════
%  CORRELATION-BASED DETECTION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[4] Detecting signal using correlation...\n');

% Cross-correlate received signal with preamble
preambleSignal = repelem(preamble, samplesPerSymbol);
preambleSignal = preambleSignal / sqrt(mean(abs(preambleSignal).^2));

correlation = abs(xcorr(rxSignal, preambleSignal));
[peakValue, peakIndex] = max(correlation);

fprintf('  Peak correlation: %.4f at sample %d\n', peakValue, peakIndex);

% Adjust index (xcorr output is 2*N-1 long)
startIndex = peakIndex - length(preambleSignal) + 1;

if startIndex < 1
    startIndex = 1;
end

fprintf('  Signal starts at sample %d\n', startIndex);

%% ══════════════════════════════════════════════════════════════════════
%  EXTRACT SYMBOLS
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[5] Extracting symbols...\n');

% Skip preamble, extract data symbols
dataStartIndex = startIndex + length(preambleSignal);
dataLength = expectedSymbols * samplesPerSymbol;

if dataStartIndex + dataLength > length(rxSignal)
    fprintf('  ⚠ Warning: Not enough samples after preamble\n');
    dataStartIndex = max(1, length(rxSignal) - dataLength);
end

dataSegment = rxSignal(dataStartIndex : min(dataStartIndex + dataLength - 1, end));

% Downsample
symbolIndices = 1:samplesPerSymbol:length(dataSegment);
extractedSymbols = dataSegment(symbolIndices);

% Trim
if length(extractedSymbols) > expectedSymbols
    extractedSymbols = extractedSymbols(1:expectedSymbols);
end

fprintf('  ✓ Extracted %d symbols\n', length(extractedSymbols));

for i = 1:length(extractedSymbols)
    fprintf('    Symbol %d: %+.4f %+.4fj (mag: %.4f)\n', ...
            i, real(extractedSymbols(i)), imag(extractedSymbols(i)), ...
            abs(extractedSymbols(i)));
end

%% ══════════════════════════════════════════════════════════════════════
%  SAVE
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[6] Saving to %s...\n', RX_FILE);

fileID = fopen(RX_FILE, 'w');
for i = 1:length(extractedSymbols)
    fprintf(fileID, '%.6f,%.6f\n', ...
            real(extractedSymbols(i)), imag(extractedSymbols(i)));
end
fclose(fileID);

fprintf('  ✓ Saved\n');

%% ══════════════════════════════════════════════════════════════════════
%  PLOTS
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[7] Plotting...\n');

figure('Position', [100, 100, 1600, 400]);

% Correlation
subplot(1,4,1);
plot(correlation);
hold on;
plot(peakIndex, peakValue, 'ro', 'MarkerSize', 10);
title('Correlation Peak');
xlabel('Sample');
ylabel('Correlation');
grid on;

% Time domain around detection
subplot(1,4,2);
plotRange = max(1, startIndex-500) : min(length(rxSignal), startIndex+2000);
plot(real(rxSignal(plotRange)));
title('RX Signal (around detection)');
xlabel('Sample');
ylabel('Amplitude');
grid on;

% Constellation comparison
subplot(1,4,3);
plot(real(extractedSymbols), imag(extractedSymbols), 'ro', 'MarkerSize', 10);
hold on;
plot(real(symbols), imag(symbols), 'bx', 'MarkerSize', 12, 'LineWidth', 2);
title('Constellation');
xlabel('I');
ylabel('Q');
legend('RX', 'TX');
grid on;
axis equal;

% Magnitude comparison
subplot(1,4,4);
bar([abs(symbols), abs(extractedSymbols)]);
title('Symbol Magnitudes');
xlabel('Symbol');
ylabel('Magnitude');
legend('TX', 'RX');
grid on;

fprintf('  ✓ Done\n');

%% ══════════════════════════════════════════════════════════════════════
%  SUMMARY
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Correlation peak: %.4f\n', peakValue);
fprintf('  Symbols extracted: %d\n', length(extractedSymbols));
fprintf('  Output: %s\n', RX_FILE);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

if peakValue > 0.1
    fprintf('✓ Strong correlation - signal detected!\n');
    fprintf('  Run: python test_hardware_rx.py\n\n');
elseif peakValue > 0.01
    fprintf('⚠ Weak correlation - signal may be present\n');
    fprintf('  Try increasing TX gain or reducing distance\n\n');
else
    fprintf('✗ No correlation - signal not detected\n');
    fprintf('  Check antenna connections and try again\n\n');
end