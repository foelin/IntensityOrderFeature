# Intensity Order based Local Features


IntensityOrderFeature is open source with a [public repository](https://github.com/foelin/IntensityOrderFeature.git) on GitHub.


This is a free software.
You can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
You should have received a copy of the GNU General Public License
along with this software.  If not, see <http://www.gnu.org/licenses/>.



### Usage
Command line arguments:

    -img                    	the input image file
    -kp 						the input region file 
	-des 						the output descriptor file
	-type       	[liop]		liop/oiop/miop/miop_fast
	-initSigma		[0.0]		Gaussian sigma for pre-smoothing
	-nSigma			[1.2]		Gaussian sigma for smoothing after normalization
	-srNum			[1]			the number of support regions
	-lsRadius		[6]			the local sampling radius of each pixel
	-normPatchWidth	[41]		the size of the normalized patch
	-liopType		[1]			weight type of LIOP, 1 for uniform weighing used in PAMI paper, 2 for weighting used in ICCV paper
	-liopRegionNum	[6]			the number of ordinal bins in LIOP
	-liopNum 		[4]			the number of local sampling points around each pixel in LIOP
	-oiopType		[1]			the quantization strategy of OIOP, 1 for learning based quantization, 2 for standard quantization
	-oiopRegionNum	[4]			the number of ordinal bins in OIOP
	-oiopNum 		[3]			the number of local sampling points around each pixel in OIOP
	-oiopQuantLevel	[4]			the number of quantization levels in OIOP
	-pcaFile					the PCA parameters for MIOP
	-pcaBasisNum	[128]		the expected dimension after PCA in MIOP
	-isApplyPCA		[0]			applying PCA dimension reduction or not
	
### Version
1.0

### Requirement
OpenCV 3.0+

Reference:
[1] Zhenhua Wang, Bin Fan and Fuchao Wu, “Local Intensity Order Pattern for Feature Description”,
IEEE International Conference on Computer Vision (ICCV) , Nov. 2011

[2] Zhenhua Wang, Bin Fan, Gang Wang and Fuchao Wu, “Exploring Local and Overall Ordinal Information for Robust Feature Description”,
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Dec. 2016.


