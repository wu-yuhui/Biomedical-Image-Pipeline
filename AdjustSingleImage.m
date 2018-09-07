% Read in single image with bounding boxes & Adjust boxes(in-place)

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
% --- Adjusting --- % 
% ================= %
h = figure('Name', image_name);
imshow(image_path);
title('ADJUST or DELETE incorrect bounding boxes');
hold on
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);  % Full Screen Window

% Read annotated box one-by-o]ne
% Data:[xmin ymin xmax ymax], MatlabPlot: [xmin ymin width(=xmax-xmin) height(=ymax-ymin)]
fid = fopen(anno_path);
readline = fgetl(fid);
boxnum = 0;
box = {};
loc = [];
while ischar(readline)
    boxnum = boxnum + 1;
    boxdata = str2double(strsplit(readline));
    boxdata(3) = boxdata(3) - boxdata(1);
    boxdata(4) = boxdata(4) - boxdata(2);
    box{boxnum} = imrect(gca, boxdata);
    readline = fgetl(fid);
end

% Show detection network prediction locations
for j = 1:length(box)
    loc = [loc; getPosition(box{j})];
end


% "DONE" botton & waiting UI
btn = uicontrol('Style', 'pushbutton', 'String', 'Done', 'Position', [20 20 100 40],'Callback', 'uiresume(gcbf)');
uiwait(h);

% Picking out deleted boxes to prevent memory issue
availbox = [1:length(box)];
for j = 1:length(box)
    try
        getPosition(box{j});
    catch
        availbox = availbox(find(availbox~=j));
    end
end

% Saving the boxes which are not deleted
pos = [];
for k = 1:length(availbox)
    curr = getPosition(box{availbox(k)});
    curr(3) = curr(1) + curr(3);  % xmax = xmin + width
    curr(4) = curr(2) + curr(4);  % ymax = ymin + height
    pos = [pos; curr];
end

% For error handling (close window before editing) -> Don't replace other data
if (~isempty(pos))
    dlmwrite(anno_path, pos, ' ');
else
    fprintf('ERROR MESSAGE: No cells chosen in figure: \"%s\".  Please Check!!!\n\n', image_name)
end
close(h)

% -----------------------------------------------%
%%%%%% Add Part %%%%%%
h = figure('Name', image_name);
imshow(image_path);
title('ADD correct bounding boxes');
hold on
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);  % Full Screen Window

% Read annotated box one-by-o]ne
% Data:[xmin ymin xmax ymax], MatlabPlot: [xmin ymin width(=xmax-xmin) height(=ymax-ymin)]

fid = fopen(anno_path);
readline = fgetl(fid);
boxnum = 0;
box = [];
while ischar(readline)
    boxnum = boxnum + 1;
    boxdata = str2double(strsplit(readline));
    box = [box; boxdata];       % Only add here, store original correct results
    boxdata(3) = boxdata(3) - boxdata(1);
    boxdata(4) = boxdata(4) - boxdata(2);
    rectangle('Position',boxdata, 'LineWidth',2, 'EdgeColor','r')
    readline = fgetl(fid);
end

btn = uicontrol('Style', 'pushbutton', 'String', 'Done', 'Position', [20 20 100 40],'Callback', @clickCallback);

global val;
val = false;
while(~val)
    curr = getrect(h);  % getrect: [xmin ymin width height]
    if(val)
        break;
    end
    rectangle('Position',curr, 'LineWidth',2, 'EdgeColor','r')
    curr(3) = curr(1) + curr(3);  % xmax = xmin + width
    curr(4) = curr(2) + curr(4);  % ymax = ymin + height
    box = [box; curr];
end

% Write back to the same file as ADJUST/DELETE
dlmwrite(anno_path, box, ' ');
close(h)

function clickCallback(source, event)
    global val
    val = true;
end