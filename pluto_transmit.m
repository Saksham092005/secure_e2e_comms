%% pluto_transmit.m
% Transmit encoded symbols from Python encoder via ADALM-Pluto SDR
%
% Workflow:
%   1. Python encoder creates tx_symbols.txt
%   2. This script reads the file
%   3. Modulates and transmits via Pluto
%
% File format: tx_symbols.txt
%   Each line: real,imag (e.g., 0.523,-0.156)

clear all;
close all;
clc;

%% ══════════════════════════════════════════════════════════════════════
%  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════

% File paths
TX_FILE = 'tx_symbols.txt';

% SDR Parameters
centerFreq = 915e6;         % 915 MHz (ISM band)
sampleRate = 1e6;           % 1 MHz
txGain = -10;                 % Transmit power (dB)

% Modulation Parameters
samplesPerSymbol = 100;     % Oversampling factor

%% ══════════════════════════════════════════════════════════════════════
%  READ SYMBOLS FROM FILE
%% ══════════════════════════════════════════════════════════════════════

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf(' PLUTO SDR TRANSMITTER\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[1] Reading symbols from %s...\n', TX_FILE);

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
fprintf('  Symbol preview (first 4):\n');
for i = 1:min(4, length(symbols))
    fprintf('    Symbol %d: %+.4f %+.4fj\n', i, real(symbols(i)), imag(symbols(i)));
end

%% ══════════════════════════════════════════════════════════════════════
%  PULSE SHAPING (UPSAMPLING)
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[2] Pulse shaping...\n');

% Upsample symbols (repeat each symbol samplesPerSymbol times)
txSignal = repelem(symbols, samplesPerSymbol);

fprintf('  ✓ Upsampled to %d samples (%d samples/symbol)\n', ...
        length(txSignal), samplesPerSymbol);

% Normalize to prevent clipping (keep power at 0.8 max)
power = mean(abs(txSignal).^2);
txSignal = txSignal * sqrt(0.64 / power);  % Scale to 80% power

fprintf('  ✓ Normalized signal power: %.4f\n', mean(abs(txSignal).^2));

%% ══════════════════════════════════════════════════════════════════════
%  TRANSMIT VIA ADALM-PLUTO
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n[3] Configuring ADALM-Pluto transmitter...\n');

try
    % Create transmitter object
    tx = sdrtx('Pluto', ...
        'RadioID', 'usb:0', ...
        'CenterFrequency', centerFreq, ...
        'Gain', txGain, ...
        'BasebandSampleRate', sampleRate);
    
    fprintf('  ✓ Pluto transmitter configured\n');
    fprintf('    Center Frequency: %.2f MHz\n', centerFreq/1e6);
    fprintf('    Sample Rate: %.2f MHz\n', sampleRate/1e6);
    fprintf('    TX Gain: %d dB\n', txGain);
    
    % Transmit
    fprintf('\n[4] Transmitting %d samples...\n', length(txSignal));
    tx(txSignal);
    
    % Wait for transmission to complete
    pause(0.5);
    
    fprintf('  ✓ Transmission complete!\n');
    
    % Release hardware
    release(tx);
    
catch ME
    fprintf('  ✗ ERROR during transmission:\n');
    fprintf('    %s\n', ME.message);
    
    % Try to release if it was created
    if exist('tx', 'var')
        release(tx);
    end
    
    rethrow(ME);
end

%% ══════════════════════════════════════════════════════════════════════
%  SUMMARY
%% ══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf(' TRANSMISSION SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Symbols transmitted: %d\n', length(symbols));
fprintf('  Total samples sent: %d\n', length(txSignal));
fprintf('  Duration: %.3f ms\n', length(txSignal)/sampleRate * 1000);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('Ready for reception. Run pluto_receive.m now.\n\n');
