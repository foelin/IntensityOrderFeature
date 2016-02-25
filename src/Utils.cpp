#include "Utils.h"
#include "assert.h"

/*---------------------------------------------------
Normalize ROI, from ellipse to circle
-----------------------------------------------------*/
bool NormalizePatch_ROI(const Mat& image, Mat& outPatch, Mat& flagPatch, const AffineKeyPoint& kp, const float origTrans[4], const float axis[2],
		float mrScale, int normPatchWidth, float initSigma, float domiOrien)
{
	int outPatchWidth = outPatch.cols;
	int outRadius = outPatchWidth / 2;

	//A
	float temp[4];

	int normPatchRadius = normPatchWidth/2;

	temp[0] = origTrans[0] * (mrScale / normPatchRadius);
	temp[1] = origTrans[1] * (mrScale / normPatchRadius);
	temp[2] = origTrans[2] * (mrScale / normPatchRadius);
	temp[3] = origTrans[3] * (mrScale / normPatchRadius);

	//AR = A*R
	float trans[4];
	trans[0] = temp[0] * cos(domiOrien) + temp[1] * sin(domiOrien);
	trans[1] = -temp[0] * sin(domiOrien) + temp[1] * cos(domiOrien);
	trans[2] = temp[2] * cos(domiOrien) + temp[3] * sin(domiOrien);
	trans[3] = -temp[2] * sin(domiOrien) + temp[3] * cos(domiOrien);

	int x,y;

	//get the max rang
	float vertexX[4]={-(float)outRadius,(float)outRadius,(float)outRadius,-(float)outRadius};
	float vertexY[4]={-(float)outRadius,-(float)outRadius,(float)outRadius,(float)outRadius};
	float vertexX_t[4];
	float vertexY_t[4];
	for (int i=0; i<4;i++)
	{
		vertexX_t[i] = fabs(trans[0]*vertexX[i] + trans[1]*vertexY[i]);
		vertexY_t[i] = fabs(trans[2]*vertexX[i] + trans[3]*vertexY[i]);
	}
	int max_x = int( max(max(vertexX_t[0], vertexX_t[1]), max(vertexX_t[2], vertexX_t[3])))+1;
	int max_y = int( max(max(vertexY_t[0], vertexY_t[1]), max(vertexY_t[2], vertexY_t[3])))+1;

	int roi_left_up_x, roi_left_up_y, roi_right_bot_x, roi_right_bot_y;
	int roi_width, roi_height;

	int orig_left_up_x =  (int)floor(kp.pt.x - max_x);
	int orig_left_up_y =  (int)floor(kp.pt.y - max_y);
	int orig_right_bot_x = (int)ceil(kp.pt.x + max_x);
	int orig_right_bot_y = (int)ceil(kp.pt.y + max_y);

	roi_left_up_x = orig_left_up_x > 0 ? orig_left_up_x : 0;
	roi_left_up_y = orig_left_up_y > 0 ? orig_left_up_y : 0;

	roi_right_bot_x = orig_right_bot_x < image.cols-1 ? orig_right_bot_x : image.cols-1;
	roi_right_bot_y = orig_right_bot_y < image.rows-1? orig_right_bot_y : image.rows-1;

	roi_width = roi_right_bot_x - roi_left_up_x + 1;
	roi_height = roi_right_bot_y - roi_left_up_y + 1;

	float new_x = kp.pt.x - roi_left_up_x;	//feature loc in ROI
	float new_y = kp.pt.y - roi_left_up_y;	//feature loc in ROI

	Rect roiRect(roi_left_up_x, roi_left_up_y, roi_width, roi_height);
	const Mat& roi = image(roiRect);

	Mat tempImg;
	//mrScale*sqrt(m_axis[0]*m_axis[1]) / (normPatchWidth/2)
	double kFactor = (mrScale*sqrt(axis[0]*axis[1]))/normPatchRadius;
	if(kFactor > 1.0)//when measureRegion.size >　normPatch.size
	{
		//double kernel = sqrt(kFactor*kFactor-1)*params.initSigma;
		double kernel = sqrt(kFactor*kFactor-initSigma*initSigma);
		GaussianBlur(roi,tempImg,  Size(0, 0),kernel);
	}
	else
	{
		roi.copyTo(tempImg);
	}

	int out_step = outPatch.step1();
	float *out_data = (float*)outPatch.data;

	int flag_step = flagPatch.step1();
	uchar *flag_data = (uchar*)flagPatch.data;

	bool isInBound = true;

	for (y=-outRadius; y<=outRadius; y++)
	{
		for (x=-outRadius; x<=outRadius; x++)
		{
			float x1 = trans[0] * x + trans[1] * y + new_x;
			float y1 = trans[2] * x + trans[3] * y + new_y;


			if (x1 < 0 || x1 > (tempImg.cols- 1) || y1 < 0 || y1 > (tempImg.rows - 1))
			{
				out_data [(y + outRadius) * out_step  + (x + outRadius)] = 0.0;
				flag_data[(y + outRadius) * flag_step + (x + outRadius)] = 0;
				isInBound = false;
			}
			else
			{
				out_data [(y + outRadius) * out_step  + (x + outRadius)] = BilinearInterU2F(x1, y1, tempImg);
				flag_data[(y + outRadius) * flag_step + (x + outRadius)] = 255;
			}
		}
	}

	return isInBound;
}


/********************************************************************
   To avoid linear illumination change, we normalize the descriptor.
   To avoid non-linear illumination change, we threshold the value 
   of each descriptor element to 'illuThresh', then normalize again.
 ********************************************************************/
void ThreshNorm(float* des, int dim, float thresh)
{
	// Normalize the descriptor, and threshold 
	// value of each element to 'illuThresh'.

	float norm = 0.f;
	int i;

	for (i=0; i<dim; ++i)
	{
		norm += des[i] * des[i];
	}

	norm = sqrt(norm);


	if (thresh <  1.f)
	{
		for (i=0; i<dim; ++i)
		{
			des[i] /= norm;

			if (des[i] > thresh)
			{
				des[i] = thresh;
			}
		}

		// Normalize again.

		norm = 0.f;

		for (i=0; i<dim; ++i)
		{
			norm += des[i] * des[i];
		}

		norm = sqrt(norm);
	}

	for (i=0; i<dim; ++i)
	{
		des[i] /= norm;
	}
}

inline float BilinearInterU2F(float x, float y, const Mat& image)
{
	assert((x >= 0 && y >= 0 && x<=image.cols-1 && y<=image.rows-1));

	int x1 = (int)x;
	int y1 = (int)y;

	int x2, y2;
	if( x1 == image.cols-1)
		x2 = x1;
	else
		x2 = x1+1;

	if(y1 == image.rows-1)
		y2 = y1;
	else
		y2 = y1+1;

	int step = image.step;
	uchar* data = (uchar*)image.data;

	float val =
			(float)((x2 - x) * (y2 - y) * data[y1*step+x1] +
					(x - x1) * (y2 - y) * data[y1*step+x2] +
					(x2 - x) * (y - y1) * data[y2*step+x1] +
					(x - x1) * (y - y1) * data[y2*step+x2]) / 255.f;

	return val;
}

inline float BilinearInterF2F(float x, float y, const Mat& image)
{
	assert((x >= 0 && y >= 0 && x<=image.cols-1 && y<=image.rows-1));

	int x1 = (int)x;
	int y1 = (int)y;

	int x2, y2;
	if( x1 == image.cols-1)
		x2 = x1;
	else
		x2 = x1+1;

	if(y1 == image.rows-1)
		y2 = y1;
	else
		y2 = y1+1;

	int step = image.step / sizeof(float);
	float* data = (float*)image.data;

	float val = 
			(x2 - x) * (y2 - y) * data[y1*step+x1] +
			(x - x1) * (y2 - y) * data[y1*step+x2] +
			(x2 - x) * (y - y1) * data[y2*step+x1] +
			(x - x1) * (y - y1) * data[y2*step+x2];

	return val;
}

bool BilinearInterF2FValid(float* val, float x, float y, const Mat& image, const Mat& flagImage)
{
	*val = 0.0;
	if(!(x >= 0 && y >= 0 && x<=image.cols-1 && y<=image.rows-1))
		return false;

	int x1 = (int)x;
	int y1 = (int)y;

	int x2, y2;
	if( x1 == image.cols-1)
		x2  = x1;
	else
		x2 = x1+1;

	if(y1 == image.rows-1 )
		y2 = y1;
	else
		y2 = y1+1;

	int step = image.step1();
	float* data = (float*)image.data;

	int flag_step = flagImage.step1();
	uchar* flag_data = (uchar*)flagImage.data;

	if(flag_data[y1*flag_step+x1] == 0 || flag_data[y1*flag_step+x2] == 0 || flag_data[y2*flag_step+x1] == 0 || flag_data[y2*flag_step+x2] == 0)
		return false;

	*val = 
			(x2 - x) * (y2 - y) * data[y1*step+x1] +
			(x - x1) * (y2 - y) * data[y1*step+x2] +
			(x2 - x) * (y - y1) * data[y2*step+x1] +
			(x - x1) * (y - y1) * data[y2*step+x2];
	return true;

}

//Compute the transformation for normalization from the ellips
void CalTrans(const AffineKeyPoint& kp, float trans[4], float axis[2])
{
	Mat A(2,2, CV_32FC1);
	Mat eigenVals, eigenVects;

	float* A_data = (float*)A.data;
	A_data[0] = kp.a;
	A_data[1] = kp.b;
	A_data[2] = kp.b;
	A_data[3] = kp.c;

	eigen(A, eigenVals, eigenVects);
	float e1 = eigenVals.at<float>(0);//larger
	float e2 = eigenVals.at<float>(1);//smaller

	Mat eigenVals_sqrt_inv(2,2, CV_32FC1);
	float* eigenVals_sqrt_inv_data = (float*)eigenVals_sqrt_inv.data;
	eigenVals_sqrt_inv_data[0] = 1/sqrt(e1);
	eigenVals_sqrt_inv_data[1] = 0.f;
	eigenVals_sqrt_inv_data[2] = 0.f;
	eigenVals_sqrt_inv_data[3] = 1/sqrt(e2);

	axis[0] = eigenVals_sqrt_inv_data[3];	//long axis
	axis[1] = eigenVals_sqrt_inv_data[0];	//short axis

	A = eigenVects.t() * eigenVals_sqrt_inv * eigenVects;

	trans[0] = A_data[0];
	trans[1] = A_data[1];
	trans[2] = A_data[2];
	trans[3] = A_data[3];

}

bool ReadKpts(const string& fileName, vector<AffineKeyPoint>&kpts)
{
	// Open the file.
	ifstream file(fileName.c_str());

	if (!file.is_open())
		FatalError("Invalid file name!");

	// Read format version.
	float version;
	file >> version;

	// Read the number
	int num;
	file >> num;
	kpts.reserve(num);

	if (fabs(version - 1.f) < 0.000001 )
	{

		for (int i=0; i<num; i++)
		{
			AffineKeyPoint kpt;
			file >> kpt.pt.x >> kpt.pt.y>> kpt.a >> kpt.b >> kpt.c;
			kpts.push_back(kpt);
		}
	}
	else
	{
		string buf;
		for (int i=0; i<num; i++)
		{
			AffineKeyPoint kpt;
			file >> kpt.pt.x >> kpt.pt.y>> kpt.a >> kpt.b >> kpt.c;
			kpts.push_back(kpt);
			for(int j=0; j<version; j++)
				file >> buf;

		}
	}


	// Close the file.
	file.close();
	return true;
}

void WriteDess(const string& fileName, const vector<AffineKeyPoint>&kpts, const Mat& dess, DES_FORMAT desFormat)
{


	// Open the file.
	ofstream file(fileName.c_str());
	if (!file.is_open())
	{
		FatalError("Invalid file name ");
	}

	int num = kpts.size();
	int dim = dess.cols;

	file << dim << endl<< num <<endl;

	if (desFormat == DES_FLOAT)
	{
		int dess_step = dess.step1();
		float* dess_data = (float*)dess.data;
		for (int i=0; i<num; i++)
		{
			const AffineKeyPoint& kpt = kpts[i];

			file << setiosflags(ios::fixed) << setprecision(2) << 
					kpt.pt.x << " " << kpt.pt.y << " " << setiosflags(ios::fixed) << setprecision(6) << kpt.a << " " << kpt.b << " " << kpt.c << " ";

			for(int j=0; j<dim; j++)
			{
				file << dess_data[i*dess_step+j] << " ";
			}
			file <<endl;
		}
	}
	else if (desFormat == DES_INT)
	{
		int dess_step = dess.step1();
		float* dess_data = (float*)dess.data;
		for (int i=0; i<num; i++)
		{
			const AffineKeyPoint& kpt = kpts[i];

			file << setiosflags(ios::fixed) << setprecision(2) << 
					kpt.pt.x << " " << kpt.pt.y << " " << setiosflags(ios::fixed) << setprecision(6) << kpt.a << " " << kpt.b << " " << kpt.c << " ";

			for(int j=0; j<dim; j++)
			{
				file << (int)(dess_data[i*dess_step+j]*255+0.5) << " ";
			}
			file <<endl;
		}
	}

	// Close the file.
	file.close();

}


//Generating the pattern mapping table for LIOP
void GeneratePatternMap(std::map<int,int>**pattern_map, int** pos_weight, int n)
{
	int i, key, count;

	if(*pattern_map != NULL)
	{
		delete *pattern_map;
	}
	*pattern_map = new map<int,int>();
	map<int,int>* temp_pattern_map = *pattern_map;

	if (*pos_weight != NULL)
	{
		delete [] *pos_weight;
	}
	*pos_weight = new int[n];
	int* temp_pos_weight = *pos_weight;


	//do the job
	int *p = new int[n];
	temp_pos_weight[0] = 1;
	for (i=1; i<n;i++)
	{
		temp_pos_weight[i] = temp_pos_weight[i-1]*10;
	}

	count=0;
	key = 0;
	for (i = 0; i < n; i++)
	{
		p[i] = i + 1;
		key += p[i]*temp_pos_weight[n-i-1];
	}
	temp_pattern_map->insert(map<int,int>::value_type(key,count));
	count++;


	while(NextPermutation(p, n))
	{
		key = 0;
		for (i = 0; i < n; i++)
		{
			key += p[i]*temp_pos_weight[n-i-1];
		}
		temp_pattern_map->insert(map<int,int>::value_type(key, count));
		count++;
	}
	delete[] p;
}

bool NextPermutation(int *p, int n)
{
	int last = n - 1;
	int i, j, k;


	i = last;
	while (i > 0 && p[i] < p[i - 1])
		i--;

	if (i == 0)
		return false;

	k = i;
	for (j = last; j >= i; j--)
		if (p[j] > p[i - 1] && p[j] < p[k])
			k = j;
	Swap(p[k], p[i - 1]);
	for (j = last, k = i; j > k; j--, k++)
		Swap(p[j], p[k]);

	return true;

}

bool fGrayComp(Pixel p1, Pixel p2)
{
	return p1.f_gray < p2.f_gray;
}

//non-descending order
void SortGray(float* dst, int* idx, float* src, int len)
{
	int i, j;

	for (i=0; i<len; i++)
	{
		dst[i] = src[i];
		idx[i] = i;
	}

	for (i=0; i<len; i++)
	{
		float temp = dst[i];
		int tempIdx = idx[i];
		for (j=i+1; j<len;j++)
		{
			if (dst[j]<temp)
			{
				temp = dst[j];
				dst[j] = dst[i];
				dst[i] = temp;

				tempIdx = idx[j];
				idx[j] = idx[i];
				idx[i] = tempIdx;
			}
		}

	}

}

//Read the matchInfo file of Patch Dataset
void ReadMatchInfo(const string& match_file, vector<MATCH_INFO>& match_info_vec)
{
	int no_use;


	ifstream ifs(match_file.c_str());
	if (!ifs.is_open())
	{
		cerr << "Can NOT open file"<<match_file<<endl;
		exit(-1);
	}

	while(1)
	{

		MATCH_INFO match_info;
		ifs >> match_info.m_patchId1 ;

		if (ifs.eof())
			break;

		ifs >> match_info.m_3dPointId1 >> no_use >> match_info.m_patchId2 >> match_info.m_3dPointId2 >> no_use >> match_info.isCorrect;

		match_info_vec.push_back(match_info);



	}
	ifs.close();
}




bool ReadImgList(const string& img_list_file, vector<string>& img_file_list)
{
	ifstream ifs(img_list_file.c_str());
	if (!ifs.is_open())
	{
		cout<<"Can not read img_kp_list_file: "<<img_list_file<<endl;
		return false;
	}

	img_file_list.clear();

	string img_file;

	int num;
	ifs>>num;
	img_file_list.clear();
	img_file_list.reserve(num);
	for(int i=0; i<num; i++)
	{
		ifs>>img_file;
		img_file_list.push_back(img_file);
	}

	ifs.close();
	return true;
}

void GetKpList(const string& kp_ext, const vector<string> img_file_list, vector<string>& kp_file_list)
{
	int num = img_file_list.size();
	kp_file_list.clear();
	kp_file_list.reserve(num);
	for (int i=0; i<num; i++)
	{
		const string& img = img_file_list[i];
		int idx = img.find_last_of('.');
		string kp_file(img.substr(0,idx+1)+kp_ext);
		kp_file_list.push_back(kp_file);
	}
}

void ReadMatrix(ifstream& ifs, Mat& mat)
{
	int dim, num;
	ifs>>dim>>num;

	mat.create(dim, num, CV_32FC1);	
	float* mat_data  = (float*)mat.data;
	int mat_step = mat.step/sizeof(float);
	for (int i=0; i<dim; i++)
	{
		for (int j=0; j<num; j++)
		{
			ifs>>mat_data[i*mat_step+j];
		}
	}
	return;
}

void WriteMatrix(ofstream& ofs, const Mat& mat)
{
	int dim = mat.rows;
	int num = mat.cols;
	ofs<<dim<<" "<<num<<endl;
	ofs<<setiosflags(ios::fixed) << setprecision(6);
	float* mat_data = (float*)mat.data;
	int mat_step = mat.step/sizeof(float);
	for (int i=0; i<mat.rows; i++)
	{
		for (int j=0; j<mat.cols; j++)
		{
			ofs<<setw(10)<<mat_data[i*mat_step+j]<<" ";
		}
		ofs<<endl;
	}	
	return;
}
