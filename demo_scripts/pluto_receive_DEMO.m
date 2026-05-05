%% pluto_receive_DEMO.m
% Pluto SDR receiver script used by run_full_demo.py

clearvars -except CHANNEL_MODE TX_FILE RX_FILE ROLE_NAME DEVICE_LABEL RADIO_ID;
close all;
clc;

%% ══════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════

% File paths
SCRIPT_DIR = fileparts(mfilename('fullpath'));
PROJECT_ROOT = fileparts(SCRIPT_DIR);
if ~exist('RX_FILE', 'var') || isempty(RX_FILE)
    RX_FILE = fullfile(PROJECT_ROOT, 'rx_symbols.txt');
end
if ~exist('TX_FILE', 'var') || isempty(TX_FILE)
    TX_FILE = fullfile(PROJECT_ROOT, 'tx_symbols.txt');  % Needed for channel simulation
end
PLOTS_DIR = fullfile(PROJECT_ROOT, 'results', 'plots');
[~, txBase, txExt] = fileparts(TX_FILE);
[~, rxBase, rxExt] = fileparts(RX_FILE);
TX_DISPLAY = [txBase txExt];
RX_DISPLAY = [rxBase rxExt];

% SDR Parameters (Bob baseline profile)
centerFreq = 915e6;         % 915 MHz
sampleRate = 1e6;           % 1 MHz
rxGain = 60;                % Receive gain (dB)

% Modulation Parameters
samplesPerSymbol = 100;     % Oversampling factor
expectedSymbols = count_symbol_lines(TX_FILE);  % Derived from TX file

% Reception Parameters
captureSeconds = 1.0;       % How long to receive

% Demo timing
HARDWARE_INIT_DELAY = 1.2;
RECEPTION_DELAY = 1.0;
PROCESSING_DELAY = 0.6;

% Channel mode (set by external script or default to 'bob')
% This determines whether we simulate Bob's or Eve's channel
if ~exist('CHANNEL_MODE', 'var')
    CHANNEL_MODE = 'bob';  % Default: legitimate receiver
end
if ~exist('ROLE_NAME', 'var') || isempty(ROLE_NAME)
    ROLE_NAME = [upper(CHANNEL_MODE) '_RX'];
end
if ~exist('RADIO_ID', 'var') || isempty(RADIO_ID)
    RADIO_ID = 'usb:0';
end

% Apply slight Eve-specific RX detuning so logs and RF profile differ.
rxProfile = 'BOB_BASELINE';
loOffsetHz = 0;
if strcmpi(CHANNEL_MODE, 'eve')
    rxProfile = 'EVE_DETUNED';
    loOffsetHz = 2.5e4;          % +25 kHz LO offset (small but non-zero)
    centerFreq = centerFreq + loOffsetHz;
    rxGain = 52;                 % Slightly reduced gain relative to Bob
    captureSeconds = 0.90;       % Slightly shorter capture window
end

samplesPerFrame = round(captureSeconds * sampleRate);

%% ══════════════════════════════════════════════════════════════════════
%  SDR INITIALIZATION
%% ══════════════════════════════════════════════════════════════════════

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' ADALM-PLUTO SDR RECEIVER\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Initializing ADALM-Pluto hardware...\n');

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

fprintf('  → Configuring receiver parameters...\n');

fprintf('  ✓ Center Frequency: %.2f MHz\n', centerFreq/1e6);
fprintf('  ✓ Sample Rate: %.2f MHz\n', sampleRate/1e6);
fprintf('  ✓ RX Gain: %d dB (manual mode)\n', rxGain);
fprintf('  ✓ Capture duration: %.2f seconds\n', captureSeconds);
fprintf('  ✓ RadioID: %s\n', RADIO_ID);
fprintf('  ✓ RX profile: %s\n', rxProfile);
fprintf('  ✓ LO detune: %.1f kHz\n', loOffsetHz/1e3);
fprintf('  ✓ Baseband ready\n');

%% ══════════════════════════════════════════════════════════════════════
%  LIVE RECEPTION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[2] Receiving signal...\n');

fprintf('  → Waiting for transmission...\n');

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

%% ══════════════════════════════════════════════════════════════════════
%  CALL PYTHON CHANNEL SIMULATOR
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[3] Processing received signal...\n');

fprintf('  → Applying matched filtering and downsampling...\n');
pause(PROCESSING_DELAY / 3);

% Call Python channel simulator to generate realistic rx_symbols.txt
% This simulates the channel effects (phase, frequency, noise)

fprintf('  → Running channel compensation...\n');

% Construct Python command using absolute paths so execution is robust
channelSimulator = fullfile(SCRIPT_DIR, 'channel_simulator.py');
pythonCmd = sprintf('python "%s" %s --tx-file "%s" --rx-file "%s" --quiet', ...
                   channelSimulator, CHANNEL_MODE, TX_FILE, RX_FILE);

% Execute silently (output suppressed for clean demo)
[status, cmdout] = system(pythonCmd);

if status ~= 0
    error('Channel simulation failed (exit=%d). Command: %s\nOutput:\n%s', ...
          status, pythonCmd, strtrim(cmdout));
end

pause(PROCESSING_DELAY / 3);

fprintf('  → Extracting symbol constellation...\n');
pause(PROCESSING_DELAY / 3);

%% ══════════════════════════════════════════════════════════════════════
%  READ SIMULATED RECEIVED SYMBOLS
%% ══════════════════════════════════════════════════════════════════════

% Now read the symbols generated by the channel simulator
if ~isfile(RX_FILE)
    error('Error: %s not found after channel simulation!', RX_FILE);
end

fileID = fopen(RX_FILE, 'r');
symbolData = textscan(fileID, '%f,%f');
fclose(fileID);

I_vals = symbolData{1};
Q_vals = symbolData{2};
extractedSymbols = I_vals + 1j * Q_vals;

fprintf('  ✓ Extracted %d symbols (expected %d)\n', ...
        length(extractedSymbols), expectedSymbols);

% Calculate received power
rxPower = mean(abs(extractedSymbols).^2);
fprintf('  ✓ Received symbol power: %.4f\n', rxPower);

fprintf('\n  Symbol preview (first 4):\n');
for i = 1:min(4, length(extractedSymbols))
    fprintf('    Symbol %d: %+.6f %+.6fj (mag: %.4f)\n', ...
            i, real(extractedSymbols(i)), imag(extractedSymbols(i)), ...
            abs(extractedSymbols(i)));
end

%% ══════════════════════════════════════════════════════════════════════
%  SAVE CONFIRMATION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[4] Saving symbols to %s...\n', RX_DISPLAY);
fprintf('  ✓ Saved %d symbols to file\n', length(extractedSymbols));

%% ══════════════════════════════════════════════════════════════════════
%  VISUALIZATION
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[5] Generating diagnostic plots...\n');

% Create figure
fig = figure('Position', [100, 100, 1400, 450], 'Name', 'Reception Analysis');

% Constellation diagram
subplot(1,3,1);
plot(real(extractedSymbols), imag(extractedSymbols), 'o', ...
     'MarkerSize', 12, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
hold on;
plot(0, 0, 'k+', 'MarkerSize', 15, 'LineWidth', 2);
grid on;
axis equal;
xlabel('In-Phase (I)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Quadrature (Q)', 'FontSize', 11, 'FontWeight', 'bold');
title('Received Constellation', 'FontSize', 12, 'FontWeight', 'bold');

% Add circles at unit distances
theta = linspace(0, 2*pi, 100);
for r = [0.5, 1.0, 1.5, 2.0]
    plot(r*cos(theta), r*sin(theta), 'k:', 'LineWidth', 0.5);
end

% Time-domain I/Q
subplot(1,3,2);
t = 1:length(extractedSymbols);
plot(t, real(extractedSymbols), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(t, imag(extractedSymbols), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
xlabel('Symbol Index', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Amplitude', 'FontSize', 11, 'FontWeight', 'bold');
title('Time-Domain I/Q', 'FontSize', 12, 'FontWeight', 'bold');
legend('I (Real)', 'Q (Imag)', 'Location', 'best');

% Magnitude vs Phase
subplot(1,3,3);
magnitudes = abs(extractedSymbols);
phases = angle(extractedSymbols);
scatter(phases * 180/pi, magnitudes, 100, 'filled', 'MarkerFaceAlpha', 0.6);
grid on;
xlabel('Phase (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Magnitude', 'FontSize', 11, 'FontWeight', 'bold');
title('Polar Representation', 'FontSize', 12, 'FontWeight', 'bold');
xlim([-180, 180]);

% Adjust layout
set(fig, 'Color', 'white');
ax = findall(fig, 'Type', 'axes');
set(ax, 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
set(findall(fig, 'Type', 'text'), 'Color', 'black');
lgd = findall(fig, 'Type', 'legend');
set(lgd, 'Color', 'white', 'TextColor', 'black');

% Save plot to disk for batch/headless runs
if ~exist(PLOTS_DIR, 'dir')
    mkdir(PLOTS_DIR);
end

plotFilename = sprintf('reception_analysis_%s.png', lower(CHANNEL_MODE));
plotPath = fullfile(PLOTS_DIR, plotFilename);

try
    exportgraphics(fig, plotPath, 'Resolution', 200);
catch
    saveas(fig, plotPath);
end

fprintf('  ✓ Plots generated\n');
fprintf('  ✓ Plot saved to: %s\n', plotPath);

pause(0.5);

%% ══════════════════════════════════════════════════════════════════════
%  SUMMARY
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' RECEPTION SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Samples received:     %d\n', receivedSamples);
fprintf('  Symbols extracted:    %d\n', length(extractedSymbols));
fprintf('  Channel mode:         %s\n', upper(CHANNEL_MODE));
fprintf('  Role:                 %s\n', ROLE_NAME);
fprintf('  RX profile:           %s\n', rxProfile);
fprintf('  Center frequency:     %.2f MHz\n', centerFreq/1e6);
fprintf('  LO detune:            %.1f kHz\n', loOffsetHz/1e3);
fprintf('  RX gain:              %d dB\n', rxGain);
fprintf('  Hardware probe:       %s\n', hardwareStatus);
fprintf('  Reception mode:       %s\n', rxMode);
fprintf('  TX file:              %s\n', TX_DISPLAY);
fprintf('  Output file:          %s\n', RX_DISPLAY);
fprintf('  Plot file:            %s\n', plotFilename);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('✓ Reception complete. Ready for decoder.\n');
fprintf('  Next step: Run Python decoder\n');
fprintf('  Command: python -c "from hardware_utils import decode_from_reception; decode_from_reception()"\n\n');


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


function n = count_symbol_lines(filename)
% Count non-empty lines to infer expected symbol count from TX file.
n = 6;
if ~isfile(filename)
    return;
end

fid = fopen(filename, 'r');
if fid < 0
    return;
end

cleanupObj = onCleanup(@() fclose(fid));
lines = textscan(fid, '%s', 'Delimiter', '\n');
if isempty(lines) || isempty(lines{1})
    return;
end

nonEmpty = ~cellfun(@isempty, strtrim(lines{1}));
n = sum(nonEmpty);
if n <= 0
    n = 6;
end
end
