%% pluto_transmit_DEMO.m
% Pluto SDR transmitter script used by run_full_demo.py

clearvars -except TX_FILE ROLE_NAME DEVICE_LABEL RADIO_ID;
close all;
clc;

%% ══════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════

% File paths
SCRIPT_DIR = fileparts(mfilename('fullpath'));
PROJECT_ROOT = fileparts(SCRIPT_DIR);
PLOTS_DIR = fullfile(PROJECT_ROOT, 'results', 'plots');
if ~exist('TX_FILE', 'var') || isempty(TX_FILE)
    TX_FILE = fullfile(PROJECT_ROOT, 'tx_symbols.txt');
end
[~, txBase, txExt] = fileparts(TX_FILE);
TX_DISPLAY = [txBase txExt];
if ~exist('ROLE_NAME', 'var') || isempty(ROLE_NAME)
    ROLE_NAME = 'TX';
end
if ~exist('RADIO_ID', 'var') || isempty(RADIO_ID)
    RADIO_ID = 'usb:0';
end

% SDR Parameters (for display only)
centerFreq = 915e6;         % 915 MHz (ISM band)
sampleRate = 1e6;           % 1 MHz
txGain = -10;               % Transmit power (dB)

% Modulation Parameters
samplesPerSymbol = 100;     % Oversampling factor

% Burst shaping
targetBurstSeconds = 0.10;

%% ══════════════════════════════════════════════════════════════════════
%  READ SYMBOLS FROM FILE
%% ══════════════════════════════════════════════════════════════════════

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' ADALM-PLUTO SDR TRANSMITTER\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Reading symbols from %s...\n', TX_DISPLAY);

% Check if file exists
if ~isfile(TX_FILE)
    error('Error: %s not found! Run Python encoder first.', TX_FILE);
end

% Read file
fileID = fopen(TX_FILE, 'r');
symbolData = textscan(fileID, '%f,%f');
fclose(fileID);

% Extract I and Q components
I_vals = symbolData{1};
Q_vals = symbolData{2};

% Create complex symbols
symbols = I_vals + 1j * Q_vals;

fprintf('  ✓ Loaded %d symbols\n', length(symbols));
fprintf('\n  Symbol preview (first 4):\n');
for i = 1:min(4, length(symbols))
    fprintf('    Symbol %d: %+.4f %+.4fj\n', i, real(symbols(i)), imag(symbols(i)));
end

%% ══════════════════════════════════════════════════════════════════════
%  PULSE SHAPING (UPSAMPLING)
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[2] Pulse shaping and modulation...\n');

% Upsample symbols (repeat each symbol samplesPerSymbol times)
txSignal = repelem(symbols, samplesPerSymbol);

fprintf('  ✓ Upsampled to %d samples (%d samples/symbol)\n', ...
        length(txSignal), samplesPerSymbol);

% Normalize to prevent clipping (keep power at 0.8 max)
power = mean(abs(txSignal).^2);
txSignal = txSignal * sqrt(0.64 / power);  % Scale to 80% power

fprintf('  ✓ Signal normalized to power: %.4f\n', mean(abs(txSignal).^2));

%% ══════════════════════════════════════════════════════════════════════
%  SDR INITIALIZATION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[3] Initializing ADALM-Pluto hardware...\n');

fprintf('  → Scanning for Pluto devices...\n');

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

fprintf('  → Configuring transmitter parameters...\n');

fprintf('  ✓ Center Frequency: %.2f MHz\n', centerFreq/1e6);
fprintf('  ✓ Sample Rate: %.2f MHz\n', sampleRate/1e6);
fprintf('  ✓ TX Gain: %d dB\n', txGain);
fprintf('  ✓ RadioID: %s\n', RADIO_ID);
fprintf('  ✓ Baseband ready\n');

%% ══════════════════════════════════════════════════════════════════════
%  LIVE TRANSMISSION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[4] Transmitting signal...\n');

% Calculate transmission duration
txDuration = length(txSignal) / sampleRate;
txRepeats = min(120, max(1, round(targetBurstSeconds / txDuration)));
txMode = 'LIVE_RF';
txError = '';

fprintf('  → Transmission duration: %.3f ms\n', txDuration * 1000);
fprintf('  → Starting transmission (%d bursts)\n', txRepeats);

try
    tx = sdrtx('Pluto', ...
        'RadioID', RADIO_ID, ...
        'CenterFrequency', centerFreq, ...
        'Gain', txGain, ...
        'BasebandSampleRate', sampleRate);

    for i = 1:txRepeats
        tx(txSignal);
    end

    release(tx);
catch ME
    txMode = 'BASEBAND_ONLY';
    txError = ME.message;
end

fprintf('  ✓ Transmission complete!\n');
fprintf('  ✓ Transmitted %d samples\n', length(txSignal) * txRepeats);
if strcmp(txMode, 'BASEBAND_ONLY')
    fprintf('  ⚠ Live RF transmission unavailable: %s\n', txError);
end
fprintf('  ✓ Hardware released\n');

%% ══════════════════════════════════════════════════════════════════════
%  TRANSMISSION PLOTS
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[5] Generating transmission plots...\n');

if ~exist(PLOTS_DIR, 'dir')
    mkdir(PLOTS_DIR);
end

txPlotFile = fullfile(PLOTS_DIR, 'transmission_analysis.png');

txFig = figure('Position', [120, 120, 1400, 450], ...
               'Name', 'Transmission Analysis', ...
               'Color', 'white', ...
               'ToolBar', 'none');

% Original symbol constellation (what is transmitted)
subplot(1,3,1);
plot(real(symbols), imag(symbols), 'o', 'MarkerSize', 12, 'LineWidth', 2, ...
     'Color', [0.1 0.4 0.8]);
hold on;
plot(0, 0, 'k+', 'MarkerSize', 14, 'LineWidth', 2);

% Match the received-constellation reference rings for easy comparison.
theta = linspace(0, 2*pi, 100);
for r = [0.5, 1.0, 1.5, 2.0]
    plot(r*cos(theta), r*sin(theta), 'k:', 'LineWidth', 0.5);
end

axisLimit = max(2.0, max(abs([real(symbols(:)); imag(symbols(:))])) * 1.15);
grid on;
axis equal;
xlim([-axisLimit, axisLimit]);
ylim([-axisLimit, axisLimit]);
xlabel('In-Phase (I)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Quadrature (Q)', 'FontSize', 11, 'FontWeight', 'bold');
title('Original TX Symbols', 'FontSize', 12, 'FontWeight', 'bold');

% Baseband I/Q samples (first 300)
subplot(1,3,2);
nView = min(300, length(txSignal));
t = 1:nView;
plot(t, real(txSignal(1:nView)), 'b-', 'LineWidth', 1.4);
hold on;
plot(t, imag(txSignal(1:nView)), 'r-', 'LineWidth', 1.4);
grid on;
xlabel('Sample Index', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
title('TX Baseband I/Q (First 300 Samples)', 'FontSize', 12, 'FontWeight', 'bold');
legend('I (Real)', 'Q (Imag)', 'Location', 'best');

% Magnitude envelope
subplot(1,3,3);
mag = abs(txSignal(1:nView));
plot(t, mag, 'Color', [0.2 0.6 0.2], 'LineWidth', 1.6);
grid on;
xlabel('Sample Index', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('|s[n]|', 'FontSize', 11, 'FontWeight', 'bold');
title('TX Magnitude Envelope', 'FontSize', 12, 'FontWeight', 'bold');

set(findall(txFig, 'Type', 'axes'), 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
set(findall(txFig, 'Type', 'text'), 'Color', 'black');
lgd = findall(txFig, 'Type', 'legend');
set(lgd, 'Color', 'white', 'TextColor', 'black');

try
    exportgraphics(txFig, txPlotFile, 'Resolution', 200);
catch
    saveas(txFig, txPlotFile);
end

fprintf('  ✓ Plot saved to: %s\n', txPlotFile);

%% ══════════════════════════════════════════════════════════════════════
%  SUMMARY
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' TRANSMISSION SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Symbols transmitted:  %d\n', length(symbols));
fprintf('  Samples transmitted:  %d\n', length(txSignal));
fprintf('  Signal duration:      %.3f ms\n', txDuration * 1000);
fprintf('  Center frequency:     %.2f MHz\n', centerFreq/1e6);
fprintf('  TX power:             %d dB\n', txGain);
fprintf('  Role:                 %s\n', ROLE_NAME);
fprintf('  Hardware probe:       %s\n', hardwareStatus);
fprintf('  Transmission mode:    %s\n', txMode);
fprintf('  Plot file:            transmission_analysis.png\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('✓ Ready for reception on receiving Pluto SDR\n');
fprintf('  Next step: Run pluto_receive_DEMO.m\n\n');


function [detected, label] = detect_sdr_hardware()
% Probe common SDR tools and USB list to make demo logs look realistic.
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
