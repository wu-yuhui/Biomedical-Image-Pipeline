% Read in single image with bounding boxes & Show boxes

% =============== %
% --- Reading --- %
% =============== %
Exist = false;
while(~Exist)
    try
        image_path = input('Enter Image Path: ','s');
        imread(image_path);
        Exist = true;
    catch ME
        fprintf('---- ERROR: \"%s\" does not exist. ----\n=> Try again... \n', image_path);
    end
end

% ===================== %
% --- Preprocessing --- %
% ===================== %
% Find image name & folder
last_slash = find(image_path == '/', 1, 'last');
image_name = image_path(last_slash+1:end);
image_folder = image_path(1:last_slash);

% Find corresponding bounding box data
txtname = strcat(strtok(image_name, '.'), '.txt');
anno_slash = find(image_path == '/', 2, 'last');
anno_path = strcat(image_path(1:anno_slash(1)), 'boxes/', txtname);   %%%%%%%%%% This anno_path can be read

% ================= %
% --- Displaying --- % 
% ================= %
h = figure('Name', image_name);
imshow(image_path);
hold on
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.25, 0.1, 0.5, 0.8]);

% Read annotated box one-by-o]ne
% Data:[xmin ymin xmax ymax], MatlabPlot: [xmin ymin width(=xmax-xmin) height(=ymax-ymin)]
fid = fopen(anno_path);
readline = fgetl(fid);

while ischar(readline)
    boxdata = str2double(strsplit(readline));
    boxdata(3) = boxdata(3) - boxdata(1);
    boxdata(4) = boxdata(4) - boxdata(2);
    rectangle('Position',boxdata, 'LineWidth',2, 'EdgeColor','r')
    readline = fgetl(fid);
end

