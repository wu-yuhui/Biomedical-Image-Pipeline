#### Visualizing/Adjusting Detection Network Results
* Data Layout:
	* Result directory should contains 2 folders, "image/" & "boxes/", storing cell image and predicted bounding boxes data, respectively. 
	The file name corresponding (image, boxes) should be have the same name as '__NAME__.png' and '__NAME__.txt'. Any violations to above
	rules will be an erroneous usage of the software.
* Execution Files:
	1. Show predicted bounding boxes: 
		* 'ShowSingleImageBox.m' for showing single image, please specify exact image path. Ex. "result/image/__NAME__.png".
	'ShowFolderImageBox.m' for showing all images in the specified folder. Ex. Input 'result/image/'.   
	2. Adjust bounding boxes:
		* 'AdjustSingleImage.m' for adjusting single image. 'AdjustFolderImage.m' for adjusting all images in the folder.
		* Path rules are as same as above. It would first show an window for adjusting & deleting boxes. Right click to delete. 
		Press 'Done' after completing are adjustments. Immediately after that, You can add bounding boxes. REMEMBER, you cannot 		undo them at this stage. Run 'Adjust__NAME_Image.m' after this if something went wrong. Press 'Done' AND THEN Press the 
		image AGAIN to leave.
		* Execute:
		> matlab -nodesktop
	
		( You would get matlab prompt, or you could open matlab window instead)
		
		> __MATLAB FILE NAME___
		
		(ex. 'ShowSingleImageBox' or 'AdjustFolderImage')

#### Visualing & Saving Segmentation Network Results
* Data Layout:
	* Result directory should contains 3 folders, "image/" & "boxes/" are the results from the detection stage, "masks/" holds the
	results of segmentation network. Names of the results should have the file names as "__NAME__.mat", which is same as 
	"__NAME__.png" in the image/ folder. We execute Matlab script indicated below to get visualize results.
* Execution Files:
	* Show and save segmentation results:
		* 'ShowSingleImageMask.m' for showing and saving all cell segments in  specified path. Ex. "result/masks/__NAME__.mat".
		The visualization will also be stored as '__NAME__.png' in the same folder. 'ShowFolderImageMask.m' for showing all 
		cell segments of all images in specified . Ex. "result/masks/__NAME__.mat". The visualizations will also be stored 
		as '__NAME__.png' in the same folder.
		* Execute:
		> matlab -nodesktop
                
		( You would get matlab prompt, or you could open matlab window instead)
                
		> __MATLAB FILE NAME___
                
		(ex. 'ShowSingleImageMask' or 'ShowFolderImageMask.m')


