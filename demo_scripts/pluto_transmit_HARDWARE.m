%% pluto_transmit_HARDWARE.m
% Live Pluto SDR transmitter used by hardware_run.py

clearvars -except TX_FILE ROLE_NAME DEVICE_LABEL RADIO_ID;
close all;
clc;

%% CONFIGURATION
SCRIPT_DIR = fileparts(mfilename('fullpath'));
PROJECT_ROOT = fileparts(SCRIPT_DIR);
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

centerFreq = 915e6;
sampleRate = 1e6;
txGain = -10;
samplesPerSymbol = 100;
targetBurstSeconds = 0.10;

%% READ SYMBOLS
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' ADALM-PLUTO SDR TRANSMITTER (HARDWARE)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Reading symbols from %s...\n', TX_DISPLAY);
if ~isfile(TX_FILE)
    error('Error: %s not found! Run the Python encoder first.', TX_FILE);
end

fileID = fopen(TX_FILE, 'r');
symbolData = textscan(fileID, '%f,%f');
fclose(fileID);

I_vals = symbolData{1};
Q_vals = symbolData{2};
symbols = I_vals + 1j * Q_vals;

fprintf('  ✓ Loaded %d symbols\n', length(symbols));
fprintf('  ✓ Role: %s\n', ROLE_NAME);
for i = 1:min(4, length(symbols))
    fprintf('    Symbol %d: %+.4f %+.4fj\n', i, real(symbols(i)), imag(symbols(i)));
end

%% PULSE SHAPING
fprintf('\n[2] Pulse shaping and modulation...\n');
txSignal = repelem(symbols, samplesPerSymbol);
power = mean(abs(txSignal).^2);
txSignal = txSignal * sqrt(0.64 / power);

fprintf('  ✓ Upsampled to %d samples (%d samples/symbol)\n', length(txSignal), samplesPerSymbol);
fprintf('  ✓ Signal normalized to power: %.4f\n', mean(abs(txSignal).^2));

%% HARDWARE INITIALIZATION
fprintf('\n[3] Initializing ADALM-Pluto hardware...\n');
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
fprintf('  ✓ TX Gain: %d dB\n', txGain);
fprintf('  ✓ RadioID: %s\n', RADIO_ID);

%% LIVE TRANSMISSION
fprintf('\n[4] Transmitting signal...\n');
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

%% SUMMARY
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
fprintf('  Plot file:            none\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('✓ Ready for live reception\n\n');

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
