% Read from image & mask both from XXX.mat and plot crops

% =============== %
% --- Reading --- %
% =============== %
Exist = false;
while(~Exist)
    try
        mask_path = input('Enter Mask Path: ','s');
        a = load(mask_path);
        Exist = true;
    catch ME
        fprintf('---- ERROR: \"%s\" does not exist. ----\n=> Try again... \n', mask_path);
    end
end

im = a.images;		% cell_num ,1 ,240, 240, 4
pred = a.preds;		% cell_num, 1, 240, 240
[cell_num, dz1, dz2 ,dz3 , dz4] = size(im);

wid = ceil(sqrt(cell_num/2)*1.5);
%wid = 6;

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
			%figure
			%imshow(SidebysideIm)
		else
			SidebysideIm = ones(dz2 ,dz3*2);
		end
		RowIm = cat(2, RowIm, SidebysideIm);
	end
	AllIm = cat(1, AllIm, RowIm);
end

% Save Concatenated Result
save_path = strcat(mask_path(1:end-4), '.png');
imwrite(AllIm, save_path);

last_slash = find(save_path == '/', 1, 'last');
save_name = save_path(last_slash+1:end);
% Display Concatenated Result
warning('off', 'images:initSize:adjustingMag')
figure('Name', save_name);
imshow(AllIm);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);

