% --- HOW TO USE THIS FUNCTION ---
% Every folder in the parent folder should only contain two other folders named Videos and Annotation_files
% Copy the block below and remove the leading `%` characters, then use it in the command window 
% (or use an editor shortcut to uncomment all at once).
%
%{
%% --- 1. Set parent folder (this is an example i used in my case) ---
parentFolder = '/Users/simeonstamboliyski/Desktop/GATE/week_5/videos_comp_vision';

%% --- 2. Run the dataset function ---
infoList = mapLE2iDataset(parentFolder);

%% --- 3. Add headers for CSV ---
headers = {'video_path', 'fall_label', 'startFrame', 'endFrame'};
infoWithHeaders = [headers; infoList];

%% --- 4. Export to CSV manually ---
fid = fopen('video_info.csv', 'w');
fprintf(fid, '%s,%s,%s,%s\n', headers{:});  % write header
for i = 1:size(infoList,1)
    fprintf(fid, '%s,%d,%d,%d\n', infoList{i,1}, infoList{i,2}, infoList{i,3}, infoList{i,4});
end
fclose(fid);

%% --- 5. Display all rows neatly in console ---
maxPathLength = max(cellfun(@length, infoWithHeaders(:,1)));
fprintf('%-*s | %-10s | %-10s | %-10s\n', maxPathLength, headers{1}, headers{2}, headers{3}, headers{4});
fprintf('%s\n', repmat('-',1,maxPathLength+36));
for i = 2:size(infoWithHeaders,1)
    fprintf('%-*s | %-10d | %-10d | %-10d\n', ...
        maxPathLength, infoWithHeaders{i,1}, infoWithHeaders{i,2}, infoWithHeaders{i,3}, infoWithHeaders{i,4});
end
%}

% --- END USAGE EXAMPLE ---



function infoList = mapLE2iDataset(parentFolder)
% parentFolder: path to the root folder containing all environment folders
% Returns: infoList as a cell array with columns: 
% {video_path, fall_label, startFrame, endFrame}

infoList = {};  % initialize empty cell array

% List all items in parent folder
envFolders = dir(parentFolder);
envFolders = envFolders([envFolders.isdir]);       % keep only folders
envFolders = envFolders(~ismember({envFolders.name},{'.','..'})); % remove . and ..

for i = 1:length(envFolders)
    envPath = fullfile(parentFolder, envFolders(i).name);
    
    % Detect subfolders for videos and annotations
    videoFolder = fullfile(envPath, 'Videos');
    annotFolder = fullfile(envPath, 'Annotation_files');
    
    % Skip if either subfolder doesn't exist
    if ~exist(videoFolder, 'dir') || ~exist(annotFolder, 'dir')
        warning('Missing Videos or Annotation_files folder in %s', envPath);
        continue;
    end
    
    % --- Process annotation files ---
    annotFiles = dir(fullfile(annotFolder, '*.txt'));
    nAnnot = length(annotFiles);
    
    arrLabel = zeros(nAnnot,1);
    fallFrames = zeros(nAnnot,2);
    
    for j = 1:nAnnot
        annotPath = fullfile(annotFolder, annotFiles(j).name);
        nums = readFirstNumbers(annotPath);  % returns [startFrame, endFrame]
        startFrame = nums(1);
        endFrame   = nums(2);
        
        fallFrames(j,:) = [startFrame, endFrame];
        if startFrame > 0
            arrLabel(j) = 1;   % fall
        else
            arrLabel(j) = 0;  % no fall
        end
    end
    
    % --- Get video files ---
    videoFiles = dir(fullfile(videoFolder, '*.avi'));
    nVideos = length(videoFiles);
    
    % Build full paths and map info
    for k = 1:nVideos
        vPath = fullfile(videoFolder, videoFiles(k).name);
        infoList(end+1, :) = {vPath, arrLabel(k), fallFrames(k,1), fallFrames(k,2)};
    end
end
end

function nums = readFirstNumbers(filePath)
fid = fopen(filePath, 'r');          
firstLine  = fgetl(fid);             
secondLine = fgetl(fid);             
fclose(fid);                          

startFrame = str2double(firstLine);
endFrame   = str2double(secondLine);

nums = [startFrame, endFrame];       
end