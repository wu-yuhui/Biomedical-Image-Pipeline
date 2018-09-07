% Read in whole folder of images with corresponding bounding boxes & Adjust
% boxes(in-place)

% ========================== %
% --- Reading all folder --- %
% ========================== %
Exist = false;
while(~Exist)
    image_folder = input('Enter Image Folder Path: ','s');
    if image_folder(end) ~= '/'
        image_folder = strcat(image_folder, '/');
    end
    
    all_images = strcat(image_folder, '/*.png');
    all_images = dir(all_images);
    
    if isempty(all_images) == true
        fprintf('---- ERROR: \"%s\" has no images. ----\n=> Try again... \n', image_folder);
    else
        Exist = true;
    end
end

images = {all_images.name}; % All images

for i = 1:length(images)
    
    % ==================================== %
    % --- Preprocessing for each image --- %
    % ==================================== %
    % Get image name & path
    %%%%last_slash = find(image_path == '/', 1, 'last');
    image_name = images{i};
    image_path = strcat(image_folder, image_name);
    fprintf('-> Displaying \"%s\". \n', image_path);
    %%%%image_folder = image_path(1:last_slash);

    % Find corresponding bounding box data
    txtname = strcat(strtok(image_name, '.'), '.txt');
    anno_slash = find(image_path == '/', 2, 'last');
    anno_path = strcat(image_path(1:anno_slash(1)), 'boxes/', txtname);   %%%%%%%%%% This anno_path can be read

    % ================== %
    % --- Displaying --- % 
    % ================== %
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
end
   