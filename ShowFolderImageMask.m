% Read from image & mask both from XXX.mat and plot crops

% ========================== %
% --- Reading all folder --- %
% ========================== %
Exist = false;
while(~Exist)
    mask_folder = input('Enter Mask Folder Path\n => ','s');
    if mask_folder(end) ~= '/'
        mask_folder = strcat(mask_folder, '/');
    end
    
    all_masks = strcat(mask_folder, '*.mat');
    all_masks = dir(all_masks);
    
    if isempty(all_masks) == true
        fprintf('---- ERROR: \"%s\" has no masks. ----\n=> Try again... \n', mask_folder);
    else
        Exist = true;
    end
end

masks = {all_masks.name}; % All images
t = 0;

for i = 1:length(masks)
    mask_path = strcat(mask_folder, masks{i});
	a = load(mask_path);

	im = a.images;		% cell_num ,1 ,240, 240, 4
	pred = a.preds;		% cell_num, 1, 240, 240
	[cell_num, dz1, dz2 ,dz3 , dz4] = size(im);

	wid = ceil(sqrt(cell_num/2)*1.5);

	AllIm = [];
	for row = 1:ceil(cell_num/wid)
		RowIm = [];
		for col = 1:wid
			if col+(row-1)*wid <= cell_num
				k = col+(row-1)*wid;
				I = squeeze(im(k,1,:,:,1));
				P = squeeze(pred(k,1,:,:));
				% Denormalize
				mu = 110.56704713;
				std = 31.27434463;
				I_norm = (mu + std * I)/255;

				maskIm = I_norm.*P;
				maskIm(~P) = 0.94;

				SidebysideIm = cat(2, I_norm, maskIm);
			else
				SidebysideIm = ones(dz2 ,dz3*2);
			end
			RowIm = cat(2, RowIm, SidebysideIm);
		end
		AllIm = cat(1, AllIm, RowIm);
	end

	save_path = strcat(mask_path(1:end-4), '.png');
	imwrite(AllIm, save_path);

	last_slash = find(save_path == '/', 1, 'last');
	save_name = save_path(last_slash+1:end);

	warning('off', 'images:initSize:adjustingMag')
	figure('Name', save_name);
	imshow(AllIm);
	set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);

end



