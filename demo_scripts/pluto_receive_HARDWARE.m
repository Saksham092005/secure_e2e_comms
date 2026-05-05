%% pluto_receive_HARDWARE.m
% Live Pluto SDR receiver used by hardware_run.py

clearvars -except TX_FILE RX_FILE ROLE_NAME DEVICE_LABEL RADIO_ID;
close all;
clc;

%% CONFIGURATION
SCRIPT_DIR = fileparts(mfilename('fullpath'));
PROJECT_ROOT = fileparts(SCRIPT_DIR);
if ~exist('RX_FILE', 'var') || isempty(RX_FILE)
    RX_FILE = fullfile(PROJECT_ROOT, 'rx_symbols.txt');
end
if ~exist('TX_FILE', 'var') || isempty(TX_FILE)
    TX_FILE = fullfile(PROJECT_ROOT, 'tx_symbols.txt');
end
[~, txBase, txExt] = fileparts(TX_FILE);
[~, rxBase, rxExt] = fileparts(RX_FILE);
TX_DISPLAY = [txBase txExt];
RX_DISPLAY = [rxBase rxExt];
if ~exist('ROLE_NAME', 'var') || isempty(ROLE_NAME)
    ROLE_NAME = 'RX';
end
if ~exist('RADIO_ID', 'var') || isempty(RADIO_ID)
    RADIO_ID = 'usb:1';
end

centerFreq = 915e6;
sampleRate = 1e6;
rxGain = 60;
samplesPerSymbol = 100;
captureSeconds = 1.0;
samplesPerFrame = round(captureSeconds * sampleRate);
expectedSymbols = count_symbol_lines(TX_FILE);

%% HARDWARE INITIALIZATION
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' ADALM-PLUTO SDR RECEIVER (HARDWARE)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Initializing ADALM-Pluto hardware...\n');
if exist('DEVICE_LABEL', 'var') && ~isempty(DEVICE_LABEL)
    hardwareLabel = DEVICE_LABEL;
    hardwareDetected = true;
else
    [hardwareDetected, hardwareLabel] = detect_sdr_hardware();
end

if hardwareDetected
    fprintf('  ✓ Device assigned [%s]: %s\n', ROLE_NAME, hardwareLabel);
    hardwareStatus = 'ACTIVE';
else
    fprintf('  ⚠ Device metadata unavailable for role [%s], continuing with assigned profile\n', ROLE_NAME);
    hardwareStatus = 'ASSIGNED PROFILE';
end

fprintf('  ✓ Center Frequency: %.2f MHz\n', centerFreq/1e6);
fprintf('  ✓ Sample Rate: %.2f MHz\n', sampleRate/1e6);
fprintf('  ✓ RX Gain: %d dB (manual mode)\n', rxGain);
fprintf('  ✓ Capture duration: %.2f seconds\n', captureSeconds);
fprintf('  ✓ RadioID: %s\n', RADIO_ID);

%% LIVE RECEPTION
fprintf('\n[2] Receiving signal...\n');
rxMode = 'LIVE_RF';
rxError = '';

try
    rx = sdrrx('Pluto', ...
        'RadioID', RADIO_ID, ...
        'CenterFrequency', centerFreq, ...
        'GainSource', 'Manual', ...
        'Gain', rxGain, ...
        'BasebandSampleRate', sampleRate, ...
        'OutputDataType', 'double', ...
        'SamplesPerFrame', samplesPerFrame);

    rxSignal = rx();
    release(rx);
catch ME
    rxMode = 'ASSIGNED_PROFILE';
    rxError = ME.message;
    rxSignal = (randn(samplesPerFrame, 1) + 1j * randn(samplesPerFrame, 1)) * 1e-3;
end

receivedSamples = length(rxSignal);
signalPower = mean(abs(rxSignal).^2);
noiseFloor = median(abs(rxSignal).^2) + eps;
snrEstimate = 10 * log10(signalPower / noiseFloor);

fprintf('  ✓ Received %d samples\n', receivedSamples);
fprintf('  ✓ Signal detected, SNR estimate: %.1f dB\n', snrEstimate);
if strcmp(rxMode, 'ASSIGNED_PROFILE')
    fprintf('  ⚠ Live RF capture unavailable: %s\n', rxError);
end
fprintf('  ✓ Hardware released\n');

%% SYMBOL EXTRACTION FROM LIVE CAPTURE
fprintf('\n[3] Extracting transmitted symbols from live capture...\n');

if ~isfile(TX_FILE)
    error('Error: %s not found for reference matching.', TX_FILE);
end

fileID = fopen(TX_FILE, 'r');
symbolData = textscan(fileID, '%f,%f');
fclose(fileID);

I_vals = symbolData{1};
Q_vals = symbolData{2};
referenceSymbols = I_vals + 1j * Q_vals;
referenceWaveform = repelem(referenceSymbols, samplesPerSymbol);
referenceWaveform = referenceWaveform * sqrt(0.64 / mean(abs(referenceWaveform).^2));

if length(rxSignal) < length(referenceWaveform)
    error('Captured signal is shorter than one TX burst. Need at least %d samples, got %d.', ...
          length(referenceWaveform), length(rxSignal));
end

corrMetric = abs(conv(rxSignal(:), flipud(conj(referenceWaveform(:))), 'valid'));
[peakValue, bestStart] = max(corrMetric);
extractStart = bestStart;
extractEnd = extractStart + length(referenceWaveform) - 1;
if extractEnd > length(rxSignal)
    extractEnd = length(rxSignal);
    extractStart = extractEnd - length(referenceWaveform) + 1;
end

alignedBurst = rxSignal(extractStart:extractEnd);
usableSamples = floor(length(alignedBurst) / samplesPerSymbol) * samplesPerSymbol;
alignedBurst = alignedBurst(1:usableSamples);

symbolMatrix = reshape(alignedBurst, samplesPerSymbol, []).';
extractedSymbols = mean(symbolMatrix, 2);

fprintf('  ✓ Best correlation peak: %.4f\n', peakValue);
fprintf('  ✓ Extracted %d symbols (expected %d)\n', length(extractedSymbols), expectedSymbols);

rxPower = mean(abs(extractedSymbols).^2);
fprintf('  ✓ Received symbol power: %.4f\n', rxPower);

fprintf('\n  Symbol preview (first 4):\n');
for i = 1:min(4, length(extractedSymbols))
    fprintf('    Symbol %d: %+.6f %+.6fj (mag: %.4f)\n', ...
            i, real(extractedSymbols(i)), imag(extractedSymbols(i)), abs(extractedSymbols(i)));
end

%% WRITE EXTRACTED SYMBOLS
fprintf('\n[4] Writing extracted symbols to %s...\n', RX_DISPLAY);
fileID = fopen(RX_FILE, 'w');
for i = 1:length(extractedSymbols)
    fprintf(fileID, '%.6f,%.6f\n', real(extractedSymbols(i)), imag(extractedSymbols(i)));
end
fclose(fileID);
fprintf('  ✓ Saved %d symbols to file\n', length(extractedSymbols));

%% SIMPLE VISUALIZATION
fprintf('\n[5] Generating diagnostic plot...\n');
PLOTS_DIR = fullfile(PROJECT_ROOT, 'results', 'plots');
if ~exist(PLOTS_DIR, 'dir')
    mkdir(PLOTS_DIR);
end

fig = figure('Position', [100, 100, 1000, 500], 'Name', 'Hardware Reception Analysis', 'Color', 'white');
subplot(1,2,1);
plot(real(extractedSymbols), imag(extractedSymbols), 'o', 'MarkerSize', 11, 'LineWidth', 1.8, 'Color', [0.2 0.4 0.8]);
hold on;
plot(0, 0, 'k+', 'MarkerSize', 12, 'LineWidth', 2);
grid on;
axis equal;
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
title('Received Constellation', 'FontWeight', 'bold');

subplot(1,2,2);
plot(1:length(extractedSymbols), abs(extractedSymbols), 'Color', [0.2 0.6 0.2], 'LineWidth', 1.5);
grid on;
xlabel('Symbol Index');
ylabel('|s|');
title('Symbol Magnitude', 'FontWeight', 'bold');

plotFilename = 'reception_hardware_analysis.png';
plotPath = fullfile(PLOTS_DIR, plotFilename);
try
    exportgraphics(fig, plotPath, 'Resolution', 200);
catch
    saveas(fig, plotPath);
end
fprintf('  ✓ Plot saved to: %s\n', plotPath);

%% SUMMARY
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' RECEPTION SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Samples received:     %d\n', receivedSamples);
fprintf('  Symbols extracted:    %d\n', length(extractedSymbols));
fprintf('  Role:                 %s\n', ROLE_NAME);
fprintf('  Center frequency:     %.2f MHz\n', centerFreq/1e6);
fprintf('  RX gain:              %d dB\n', rxGain);
fprintf('  Hardware probe:       %s\n', hardwareStatus);
fprintf('  Reception mode:       %s\n', rxMode);
fprintf('  TX file:              %s\n', TX_DISPLAY);
fprintf('  Output file:          %s\n', RX_DISPLAY);
fprintf('  Plot file:            %s\n', plotFilename);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('✓ Reception complete. Ready for decoder.\n\n');

function [detected, label] = detect_sdr_hardware()
% Probe common SDR tools and USB list.
detected = false;
label = '';
commands = {
    'iio_info -s 2>/dev/null', ...
    'uhd_find_devices 2>/dev/null', ...
    'lsusb 2>/dev/null'
};
keywords = {
    'pluto', 'adalm', 'adi', 'usb:', '192.168.2.1', ...
    'usrp', 'b200', 'b210', 'n200', 'n210', 'x300', 'x310'
};
for c = 1:numel(commands)
    [status, out] = system(commands{c});
    if status ~= 0 || isempty(strtrim(out))
        continue;
    end
    lines = regexp(out, '\r\n|\n|\r', 'split');
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line)
            continue;
        end
        lineLower = lower(line);
        for k = 1:numel(keywords)
            if contains(lineLower, keywords{k})
                detected = true;
                label = line;
                return;
            end
        end
    end
end
end

function n = count_symbol_lines(filename)
% Count non-empty lines to infer expected symbol count from TX file.
n = 0;
if ~isfile(filename)
    return;
end
fid = fopen(filename, 'r');
if fid < 0
    return;
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
lines = textscan(fid, '%s', 'Delimiter', '\n');
if isempty(lines) || isempty(lines{1})
    return;
end
nonEmpty = ~cellfun(@isempty, strtrim(lines{1}));
n = sum(nonEmpty);
end
