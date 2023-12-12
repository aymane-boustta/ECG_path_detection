function [tm,ecgsig,ann,Fs,sizeEcgSig,timeEcgSig] = loadEcgSig(Name)
% USAGE: [tm,ecgsig,ann,Fs,sizeEcgSig,timeEcgSig] = loadEcgSig('../data/200m')
% This function reads a pair of files (RECORDm.mat and RECORDm.info) generated
% by 'wfdb2mat' from a PhysioBank record, baseline-corrects and scales the time
% series contained in the .mat file, returning time, amplitude and frequency.
% The baseline-corrected and scaled time series are the rows of matrix 'val', and each
% column contains simultaneous samples of each time series.
% 'wfdb2mat' is part of the open-source WFDB Software Package available at
%    http://physionet.org/physiotools/wfdb.shtml
% If you have installed a working copy of 'wfdb2mat', run a shell command
% such as wfdb2mat -r 100s -f 0 -t 10 >100sm.info
% to create a pair of files ('100sm.mat', '100sm.info') that can be read
% by this function.
% The files needed by this function can also be produced by the
% PhysioBank ATM, at http://physionet.org/cgi-bin/ATM

% Adapted from
% loadEcgSignal.m           O. Abdala			16 March 2009
% 		      James Hislop	       27 January 2014	version 1.1

% Last version
% loadEcgSignal.m           D. Kawasaki			15 June 2017
% 		      Davi Kawasaki	       15 June 2017 version 1.0
increment = 1;
infoName = strcat(Name, '.info');
matName = strcat(Name, '.mat');
disp(matName);
load(matName);
ecgsig = val;
fid = fopen(infoName, 'rt');
fgetl(fid);
fgetl(fid);
line = fgetl(fid);
disp(line)
[freqint] = sscanf(fgetl(fid), 'Sampling frequency: %f Hz  Sampling interval: %f sec');
Fs = freqint(1);
interval = freqint(2);
fgetl(fid);

for i = 1:size(ecgsig, 1)
  [row(i), signal(i), gain(i), base(i), units(i)]=strread(fgetl(fid),'%d%s%f%f%s','delimiter','\t');
end

fclose(fid);
ecgsig(ecgsig==-32768) = NaN;

for i = 1:size(ecgsig, 1)
    ecgsig(i, :) = (ecgsig(i, :) - base(i)) / gain(i);
end
N = size(ecgsig, 2);
tm1 = 1/Fs:1/Fs:N/Fs;
tm = (1:size(ecgsig, 2)) * interval;
sizeEcgSig = size(ecgsig, 2);
timeEcgSig = sizeEcgSig*interval;
%plot(tm', val');

%for i = 1:length(signal)
%    labels{i} = strcat(signal{i}, ' (', units{i}, ')'); 
%end

%legend(labels);
%xlabel('Time (sec)');
% grid on

% load annotations

annotationName = strcat(Name, '.txt');
fid = fopen(annotationName, 'rt');
disp(fid);
% was annotationsEcg = textscan(fid, '%d:%f %d %*c %*d %*d %*d %s', 'HeaderLines', 1, 'CollectOutput', 1);
ann = textscan(fid, '[%d:%d:%f] %d %c %d %d %d %s', 'HeaderLines', 1, 'CollectOutput', 1);
disp(increment);
increment = plus(increment, 1);
disp(ann);
fclose(fid);
% % Initialize cell array to store data
%     data = cell(0, 1);
% 
%     % Read each line from the file
%     while ~feof(fid)
%         % Read the line as a string
%         line = fgetl(fid);
% 
%         % Use regular expression to parse the line
%         pattern = '\[(\d+:\d+:\d+\.\d+)\]\s*1?\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)';
%         match = regexp(line, pattern, 'tokens');
%         increment = plus(increment, 1);
%         disp(increment);
% 
%         % Check if the line matches the expected format
%         if ~isempty(match) && numel(match) == 1 && numel(match{1}) == 6
%             % Convert matched values to the desired data types
%             timestamp_str = match{1}{1};
%             timestamp = datetime(timestamp_str, 'InputFormat', 'HH:mm:ss.SSS');
%             value1 = str2double(match{1}{2});
%             value2 = match{1}{3};
%             value3 = str2double(match{1}{4});
%             value4 = str2double(match{1}{5});
%             value5 = str2double(match{1}{6});
% 
%             % Store the data in a structure
%             entry = struct('timestamp', timestamp, 'value1', value1, 'value2', value2, ...
%                            'value3', value3, 'value4', value4, 'value5', value5);
% 
%             % Append the entry to the data cell array
%             data{end+1} = entry;
%         end
%     end
% 
%     % Close the file
%     fclose(fid);
% 
%     % Return the parsed annotations
%     ann = data;


end