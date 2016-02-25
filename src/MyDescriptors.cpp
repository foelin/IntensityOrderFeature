#include <MyDescriptors.h>
#include "Utils.h"
#include <algorithm>

//Below are learned normalized quantization locations for OIOP
// when oiopNum = 3
const float oiop_fence33_o3[3*2] = {	
		0.169, 0.405,
		0.372, 0.623,
		0.592, 0.828};


const float oiop_fence34_o3[3*3] = {
		0.124, 0.273, 0.492,
		0.302, 0.498, 0.694,
		0.504, 0.725, 0.874};


const float oiop_fence43_o3[4*2] = { 
		0.149, 0.372,
		0.300, 0.545,
		0.451, 0.697,
		0.625, 0.850};

const float oiop_fence44_o3[4*3] = {	
		0.107, 0.243, 0.459,
		0.238, 0.417, 0.622,
		0.374, 0.579, 0.760,
		0.537, 0.755, 0.891};


const float oiop_fence53_o3[5*2] = {	
		0.136, 0.352,
		0.254, 0.495,
		0.376, 0.619,
		0.502, 0.744,
		0.645, 0.863 };

const float oiop_fence54_o3 [5*3] = {	
		0.098, 0.226, 0.440,
		0.198, 0.366, 0.577,
		0.307, 0.498, 0.688,
		0.420, 0.632, 0.800,
		0.555, 0.772, 0.901};

////////////////////
// when oiopNum = 2
const float oiop_fence33_o2[3*2] = {	
		0.169, 0.427,
		0.372, 0.641,
		0.593, 0.840 };


const float oiop_fence34_o2[3*3] = {	
		0.122, 0.278, 0.526,
		0.298, 0.506, 0.714,
		0.495, 0.737, 0.885};


const float oiop_fence43_o2[4*2] = { 
		0.146, 0.391,
		0.302, 0.570,
		0.446, 0.709,
		0.629, 0.862};


const float oiop_fence44_o2[4*3] = {	
		0.104, 0.247, 0.492,
		0.239, 0.428, 0.652,
		0.364, 0.586, 0.772,
		0.530, 0.768, 0.903	};


const float oiop_fence53_o2[5*2] = {
		0.132, 0.369,
		0.258, 0.522,
		0.375, 0.636,
		0.496, 0.753,
		0.651, 0.876};

const float oiop_fence54_o2[5*3] = {
		0.093, 0.228, 0.471,
		0.200, 0.376, 0.611,
		0.303, 0.506, 0.710,
		0.407, 0.639, 0.808,
		0.551, 0.787, 0.913};


const int MAX_REGION_NUM = 10;
const int MAX_SAMPLE_NUM = 10;

MyDescriptors::MyDescriptors(Params& params)
:m_params(params)
{

	if (m_params.isApplyPCA)
	{
		if(!readPCA(m_params.PCAFile))
		{
			cout<<"Can NOT read the PCA file"<<endl;
			exit(0);
		}
	}

	m_computeDomiOri = true;
	switch(m_params.des_type)
	{


	case LIOP:
	{
		int patternWidth = m_params.liopNum == 3 ? 6 : 24;
		m_dim = patternWidth*m_params.liopRegionNum;
		m_bytes = m_dim * sizeof(float);
		m_dataType = CV_32FC1;
		m_computeDomiOri = false;

		m_ptrCreateFeatFunc = &MyDescriptors::createLIOP;
	}
	break;

	case OIOP:
	{
		int patternWidth = pow((float)params.oiopQuantLevel,m_params.oiopNum);
		m_dim = patternWidth*m_params.oiopRegionNum;

		m_fenceRatio = NULL;
		if (m_params.oiopType == 1)
		{
			if (m_params.oiopRegionNum == 4)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence44_o2;
					else
						m_fenceRatio =  oiop_fence44_o3;

				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence43_o2;
					else
						m_fenceRatio =  oiop_fence43_o3;
				}
			}
			else if (m_params.oiopRegionNum == 3)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence34_o2;
					else
						m_fenceRatio =  oiop_fence34_o3;
				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence33_o2;
					else
						m_fenceRatio =  oiop_fence33_o3;
				}
			}
			else if (m_params.oiopRegionNum == 5)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence54_o2;
					else
						m_fenceRatio =  oiop_fence54_o3;
				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence53_o2;
					else
						m_fenceRatio =  oiop_fence53_o3;
				}
			}
		}


		m_bytes = m_dim * sizeof(float);
		m_dataType = CV_32FC1;
		m_computeDomiOri = false;
		m_ptrCreateFeatFunc = &MyDescriptors::createOIOP;


	}
	break;

	case MIOP:
	case MIOP_FAST:
	{
		int liopPatternWidth = m_params.liopNum == 3 ? 6 : 24;
		int liopDim = liopPatternWidth*m_params.liopRegionNum;


		int oiopPatternWidth = pow((float)params.oiopQuantLevel,m_params.oiopNum);
		int oiopDim = oiopPatternWidth*m_params.oiopRegionNum;

		m_fenceRatio = NULL;
		if (m_params.oiopType == 1)
		{
			if (m_params.oiopRegionNum == 4)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence44_o2;
					else
						m_fenceRatio =  oiop_fence44_o3;

				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence43_o2;
					else
						m_fenceRatio =  oiop_fence43_o3;
				}
			}
			else if (m_params.oiopRegionNum == 3)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence34_o2;
					else
						m_fenceRatio =  oiop_fence34_o3;
				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence33_o2;
					else
						m_fenceRatio =  oiop_fence33_o3;
				}
			}
			else if (m_params.oiopRegionNum == 5)
			{
				if (m_params.oiopQuantLevel == 4)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence54_o2;
					else
						m_fenceRatio =  oiop_fence54_o3;
				}
				else if(m_params.oiopQuantLevel  == 3)
				{
					if (m_params.oiopNum == 2)
						m_fenceRatio = oiop_fence53_o2;
					else
						m_fenceRatio =  oiop_fence53_o3;
				}
			}
		}

		m_dim = liopDim + oiopDim;
		m_bytes = m_dim * sizeof(float);
		m_dataType = CV_32FC1;
		m_computeDomiOri = false;


		m_ptrCreateFeatFunc = m_params.des_type == MIOP ? &MyDescriptors::createMIOP : &MyDescriptors::createMIOP_FAST;
	}
	break;

	default:
		FatalError("Undefined descriptor type!");
		break;
	}


	m_dim *= m_params.srNum;	//multiple support region

}


MyDescriptors::~MyDescriptors(void)
{
}

void MyDescriptors::compute( const Mat& image, vector<AffineKeyPoint>& keypoints, Mat& descriptors ) const
{
	if( image.empty() || keypoints.empty() )
	{
		descriptors.release();
		return;
	}

	computeImpl( image, keypoints, descriptors );
}

int MyDescriptors::descriptorSize() const
{
	return m_bytes;
}

int MyDescriptors::descriptorType() const
{
	return m_dataType;
}

//To extract descriptors on affine covariant regions
void MyDescriptors::computeImpl( const Mat& image, vector<AffineKeyPoint>& keypoints, Mat& descriptors ) const
{
#ifdef TEST_TIME
	clock_t start, finish;
	double time;
	start = clock();

#endif
	int num = keypoints.size();
	descriptors = Mat::zeros(num, m_dim, CV_32F);
	float* des_data = (float*)descriptors.data;
	int des_step = descriptors.step1();


	//To add some borders to avoid border effect only. Actually we use the inside  patch to extract descriptor
	int outPatchWidth = m_params.normPatchWidth+16;
	if(outPatchWidth%2 == 0)
		outPatchWidth++;

	for (int i=0; i<num ;i++)
	{

		Mat outPatch(outPatchWidth, outPatchWidth, CV_32FC1);
		Mat flagPatch(outPatchWidth, outPatchWidth, CV_8UC1);

		float trans[4];
		float axis[2];
		CalTrans(keypoints[i], trans, axis);

		for(int k=0; k<m_params.srNum; k++)
		{
			int single_dim = m_dim/m_params.srNum;

			float domiOrien = 0.f;
			NormalizePatch_ROI(image, outPatch, flagPatch, keypoints[i],trans, axis, m_params.srScaleStep*k+m_params.mrScale,
					m_params.normPatchWidth, m_params.initSigma, domiOrien);
			if (m_params.nSigma > 0)
			{
				GaussianBlur(outPatch, outPatch, Size(0, 0), m_params.nSigma);
			}

			(this->*m_ptrCreateFeatFunc)(outPatch, flagPatch, m_params.normPatchWidth, des_data+i*des_step+k*single_dim);

		}
	}


#ifdef TEST_TIME
	finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "used " << time << " s for describe " <<num<<" features, "<<time/num*1000<< " ms for each one."<<endl<<endl;
#endif

	if ((m_params.des_type == MIOP || m_params.des_type == MIOP_FAST ) && m_params.isApplyPCA )
	{
		if(m_params.srNum == 1)
		{
			Mat desMean;
			repeat(m_PCAMean,num,1,desMean);
			descriptors = (descriptors - desMean)*m_PCABasis;
		}
		else
		{
			Mat desMean;
			repeat(m_PCAMean,num,1,desMean);

			Mat pcaDescriptors = Mat::zeros(num, m_PCABasis.cols*m_params.srNum, CV_32F);

			int single_dim = m_dim/m_params.srNum;
			for(int k=0; k<m_params.srNum; k++)
			{
				Mat srcDes = descriptors.colRange(k*single_dim, (k+1)*single_dim);
				Mat desDes = pcaDescriptors.colRange(k*m_PCABasis.cols, (k+1)*m_PCABasis.cols);
				desDes = (srcDes - desMean)*m_PCABasis;

			}

			pcaDescriptors.copyTo(descriptors);
		}

	}

	return;
}

//To extract descriptors for Patch Dataset
void MyDescriptors:: computePatchImage(const Mat& image, int patch_per_row, int patch_per_col, int patch_length, int max_patch_num,  Mat& descriptors) const
{
	if (max_patch_num < 0)
	{
		max_patch_num = patch_per_col*patch_per_row;
	}


	descriptors = Mat::zeros(max_patch_num, m_dim, CV_32F);
	float* des_data = (float*)descriptors.data;
	int des_step = descriptors.step1();

	int inPatchSz = patch_length - m_params.lsRadius*2;

	Mat outPatch(Size(patch_length, patch_length), CV_32FC1);
	Mat flagPatch = Mat::ones(Size(patch_length, patch_length), CV_8UC1)*255;

	int count=0;
	for(int r=0; r<patch_per_col; r++)
	{
		for(int c=0; c<patch_per_row; c++)
		{
			if (count >= max_patch_num)
				break;

			Rect roi_rec(c*patch_length,r*patch_length,patch_length,patch_length);
			image(roi_rec).convertTo(outPatch, CV_32FC1,1/255.0);
			if (m_params.nSigma > 0)
			{
				GaussianBlur(outPatch, outPatch, Size(0, 0), m_params.nSigma);
			}


			(this->*m_ptrCreateFeatFunc)(outPatch, flagPatch, inPatchSz, des_data);

			des_data += des_step;

			count++;
		}

		if (count >= max_patch_num)
			break;

	}


	if (m_params.isApplyPCA)
	{
		if(m_params.srNum == 1)
		{
			Mat desMean;
			repeat(m_PCAMean,max_patch_num,1,desMean);
			descriptors = (descriptors - desMean)*m_PCABasis;
		}
		else
		{
			Mat desMean;
			repeat(m_PCAMean,max_patch_num,1,desMean);

			Mat pcaDescriptors = Mat::zeros(max_patch_num, m_PCABasis.cols*m_params.srNum, CV_32F);

			int single_dim = m_dim/m_params.srNum;
			for(int k=0; k<m_params.srNum; k++)
			{
				Mat srcDes = descriptors.colRange(k*single_dim, (k+1)*single_dim);
				Mat desDes = pcaDescriptors.colRange(k*m_PCABasis.cols, (k+1)*m_PCABasis.cols);
				desDes = (srcDes - desMean)*m_PCABasis;

			}

			pcaDescriptors.copyTo(descriptors);
		}

	}

}


void MyDescriptors::removeOutBound( const Mat& image, const vector<AffineKeyPoint>& keypoints,  vector<AffineKeyPoint>& keypointsInBounds) const
{
	int num = keypoints.size();
	keypointsInBounds.reserve(num);
	for (int i=0; i<num ;i++)
	{
		const AffineKeyPoint& kp = keypoints[i];

		//To add some borders to avoid border effect only. Actually we use the inside  patch to extract descriptor
		int outPatchWidth = m_params.normPatchWidth+16;
		if(outPatchWidth%2 == 0)
			outPatchWidth++;


		Mat outPatch(outPatchWidth, outPatchWidth, CV_32FC1);
		Mat flagPatch(outPatchWidth, outPatchWidth, CV_8UC1);

		float trans[4];
		float axis[2];
		CalTrans(kp, trans, axis);

		float domiOrien = 0.f;
		float maxScale = m_params.srScaleStep*(m_params.srNum-1)+m_params.mrScale;
		bool isInBound = NormalizePatch_ROI(image, outPatch, flagPatch, kp, trans, axis, maxScale,
				m_params.normPatchWidth, m_params.initSigma, domiOrien);
		if (isInBound)
		{
			keypointsInBounds.push_back(kp);
		}
	}
}

/*
case 1 uniform weight used in PAMI paper
case 2 weight function used in ICCV paper
 */
void MyDescriptors:: createLIOP(const Mat& outPatch, const Mat& flagPatch, int inPatchSz, float* des) const
{
	int i,k,x,y;

	float* out_data = (float *)outPatch.data;
	int out_step = outPatch.step1();
	uchar* flag_data = (uchar*)flagPatch.data;
	int flag_step = flagPatch.step1();

	int inRadius_left = inPatchSz/2;
	int inRadius_right = inPatchSz - inRadius_left - 1;
	float inRadius2 = (float)(inRadius_left*inRadius_left);

	int outRadius = outPatch.cols / 2;

	Pixel* pixel = new Pixel[inPatchSz*inPatchSz];
	int pixelCount = 0;

	int idx[MAX_SAMPLE_NUM];
	float dst[MAX_SAMPLE_NUM];
	float src[MAX_SAMPLE_NUM];

	float theta = 2*CV_PI/m_params.liopNum;

	switch (m_params.liopType)
	{
	case 1:
	{
		for (y=-inRadius_left; y<=inRadius_right; ++y)
		{
			for (x=-inRadius_left; x<=inRadius_right; ++x)
			{

				float dis2 = (float)(x*x + y*y);
				if(dis2 > inRadius2)
				{
					continue;
				}

				float cur_gray = out_data [(y+outRadius)*out_step +x+outRadius];
				uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
				if (cur_flag == 0)
				{
					continue;
				}

				float nDirX = x;
				float nDirY = y;
				float nOri = atan2(nDirY, nDirX);
				if (fabs(nOri - CV_PI) < INFMIN_F)
				{
					nOri = -CV_PI;
				}

				bool isInBound = true;
				for (k=0; k<m_params.liopNum; k++)
				{
					float deltaX = m_params.lsRadius*cos(nOri+k*theta);
					float deltaY = m_params.lsRadius*sin(nOri+k*theta);

					float sampleX = x+deltaX;
					float sampleY = y+deltaY;
					float gray;

					if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
					{
						isInBound = false;
						break;
					}

					src[k] = gray;
				}

				if (!isInBound)
				{
					continue;
				}

				int key = 0;
				SortGray(dst, idx, src, m_params.liopNum);
				for (k=0; k<m_params.liopNum; k++)
				{
					key += (idx[k]+1)*m_params.pLiopPosWeight[m_params.liopNum-k-1];
				}
				map<int, int>::iterator iter;
				iter = m_params.pLiopPatternMap->find(key);

				Pixel pix;
				pix.x = x;
				pix.y = y;
				pix.f_gray = cur_gray;
				pix.i_gray = (int)(pix.f_gray*255+0.5f);
				pix.weight = 1;
				pix.l_pattern = iter->second;
				pixel[pixelCount++] = pix;
			}
		}


		sort(pixel, pixel+pixelCount, fGrayComp);

		int l_patternWidth = m_params.liopNum == 3 ? 6 : 24;
		int dim = l_patternWidth*m_params.liopRegionNum;

		if (pixelCount >= m_params.liopRegionNum)
		{

			int curId = 0;
			int lastId = 0;
			for (i=0; i<m_params.liopRegionNum; i++)
			{
				int fenceId = pixelCount*(i+1)/m_params.liopRegionNum-1;
				float fenceGray = pixel[fenceId].f_gray;
				int regionId = i;
				curId = lastId;

				while (true)
				{
					if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
					{
						lastId = curId;
						break;
					}

					int id = regionId*l_patternWidth+pixel[curId].l_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}

				while (true)
				{
					if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
						break;

					int id = regionId*l_patternWidth+pixel[curId].l_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}
			}

			ThreshNorm(des,dim,1.f);
		}

	}
	break;

	case 2:
	{

		// normalize the gray values
		float max_gray = 0.f;
		float min_gray = 1.f;

		for (y=-outRadius; y<=outRadius; ++y)
		{
			for (x=-outRadius; x<=outRadius; ++x)
			{
				float cur_gray = out_data [(y+outRadius)*out_step +x+outRadius];
				uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
				if (cur_flag == 0)
				{
					continue;
				}

				if (cur_gray > max_gray)
				{
					max_gray = cur_gray;
				}

				if (cur_gray < min_gray)
				{
					min_gray = cur_gray;
				}

			}
		}

		float dif_gray = max_gray - min_gray;
		if (dif_gray < INFMIN_F)
		{
			dif_gray = INFMIN_F;
		}

		for (y=-inRadius_left; y<=inRadius_right; ++y)
		{
			for (x=-inRadius_left; x<=inRadius_right; ++x)
			{

				float dis2 = (float)(x*x + y*y);
				if(dis2 > inRadius2)
				{
					continue;
				}

				float cur_gray = out_data [(y+outRadius)*out_step +x+outRadius];
				uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
				if (cur_flag == 0)
				{
					continue;
				}

				float nDirX = x;
				float nDirY = y;
				float nOri = atan2(nDirY, nDirX);
				if (fabs(nOri - CV_PI) < INFMIN_F)
				{
					nOri = -CV_PI;
				}


				bool isInBound = true;
				for (k=0; k<m_params.liopNum; k++)
				{
					float deltaX = m_params.lsRadius*cos(nOri+k*theta);
					float deltaY = m_params.lsRadius*sin(nOri+k*theta);

					float sampleX = x+deltaX;
					float sampleY = y+deltaY;
					float gray;

					if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
					{
						isInBound = false;
						break;
					}

					src[k] = (gray-min_gray) / dif_gray;
				}

				if (!isInBound)
				{
					continue;
				}

				int key = 0;
				SortGray(dst, idx, src, m_params.liopNum);
				for (k=0; k<m_params.liopNum; k++)
				{
					key += (idx[k]+1)*m_params.pLiopPosWeight[m_params.liopNum-k-1];
				}
				map<int, int>::iterator iter;
				iter = m_params.pLiopPatternMap->find(key);

				float thre = m_params.liopThre/255.0;
				float dif = 0.0;
				int count = 0;
				for (i=m_params.liopNum-1; i>0; i--)
				{
					for (k=i-1; k>=0; k--)
					{
						dif = fabs(dst[i] - dst[k]);
						if (dif>thre)
						{
							count ++;
						}
					}
				}

				float weight = count+1;

				Pixel pix;
				pix.x = x;
				pix.y = y;
				pix.f_gray = cur_gray;
				pix.i_gray = (int)(pix.f_gray*255+0.5f);
				pix.weight = weight;
				pix.l_pattern = iter->second;
				pix.id = pixelCount;
				pixel[pixelCount++] = pix;
			}
		}


		sort(pixel, pixel+pixelCount, fGrayComp);

		int l_patternWidth = m_params.liopNum == 3 ? 6 : 24;
		int dim = l_patternWidth*m_params.liopRegionNum;

		if (pixelCount >= m_params.liopRegionNum)
		{
			int curId = 0;
			int lastId = 0;
			for (i=0; i<m_params.liopRegionNum; i++)
			{
				int fenceId = pixelCount*(i+1)/m_params.liopRegionNum-1;
				float fenceGray = pixel[fenceId].f_gray;
				int regionId = i;
				curId = lastId;

				while (true)
				{
					if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
					{
						lastId = curId;
						break;
					}

					int id = regionId*l_patternWidth+pixel[curId].l_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}

				while (true)
				{
					if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
						break;

					int id = regionId*l_patternWidth+pixel[curId].l_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}
			}

			ThreshNorm(des,dim, 1.f);
		}

	}
	break;

	default:
		printf("undefined case!\n");
		exit(0);
	}

	delete [] pixel;
}


/*
case 1 learning based quantization as described in PAMI paper
case 2 standard quantization
 */

void MyDescriptors::createOIOP(const Mat& outPatch, const Mat& flagPatch, int inPatchSz, float* des) const
{

	int i,j,k,x,y;

	float* out_data = (float *)outPatch.data;
	int out_step = outPatch.step1();
	uchar* flag_data = (uchar*)flagPatch.data;
	int flag_step = flagPatch.step1();

	int inRadius_left = inPatchSz/2;
	int inRadius_right = inPatchSz - inRadius_left - 1;
	float inRadius2 = (float)(inRadius_left*inRadius_left);

	int outRadius = outPatch.cols / 2;

	Pixel* pixel = new Pixel[inPatchSz*inPatchSz];

	switch (m_params.oiopType)
	{

	case 1:
		{

			int sampleNum = m_params.oiopNum;
			Mat pixelSample(sampleNum,inPatchSz*inPatchSz,CV_32FC1);
			float* pixelSample_data = (float*)pixelSample.data;
			int pixelSample_step = pixelSample.step1();

			float theta = 2*CV_PI/m_params.oiopNum;
			int pixelCount = 0;
			for (y=-inRadius_left; y<=inRadius_right; ++y)
			{
				for (x=-inRadius_left; x<=inRadius_right; ++x)
				{
					float dis2 = (float)(x*x + y*y);
					if(dis2 > inRadius2)
					{
						continue;
					}

					float cur_gray = out_data[(y+outRadius)*out_step+x+outRadius];
					uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
					if (cur_flag == 0)
					{
						continue;
					}

					float nDirX = x;
					float nDirY = y;
					float nOri = atan2(nDirY, nDirX);
					if (fabs(nOri - CV_PI) < INFMIN_F)
					{
						nOri = -CV_PI;
					}

					bool isInBound = true;
					int radius = m_params.lsRadius;

					for (k=0; k<m_params.oiopNum; k++)
					{
						float deltaX = radius*cos(nOri+k*theta);
						float deltaY = radius*sin(nOri+k*theta);

						float sampleX = x+deltaX;
						float sampleY = y+deltaY;
						float gray;

						if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
						{
							isInBound = false;
							break;
						}

						pixelSample_data[k*pixelSample_step+pixelCount] = gray;
					}

					if (!isInBound)
					{
						continue;
					}

					Pixel pix;
					pix.x = x;
					pix.y = y;
					pix.f_gray = cur_gray;
					pix.i_gray = (int)(pix.f_gray*255+0.5);
					pix.weight = 1;
					pix.id = pixelCount;
					pix.o_pattern = 0;
					pixel[pixelCount++] = pix;
				}
			}


			sort(pixel, pixel+pixelCount, fGrayComp);

			float subFenceGray[MAX_REGION_NUM][MAX_REGION_NUM];
			int fenceNum = m_params.oiopQuantLevel-1;
			int subFenceId;
			for (k=0; k<m_params.oiopRegionNum; k++)
			{
				for (i=0; i<fenceNum; i++)
				{
					subFenceId = int(m_fenceRatio[k*fenceNum+i]*pixelCount);
					subFenceGray[k][i] = pixel[subFenceId].f_gray;
				}
			}

			//compute oiop pattern
			int posWeight = 1;
			for (i=0; i<sampleNum; i++)
			{
				for (j=0; j<pixelCount; j++)
				{

					Pixel& pix = pixel[j];


					int regionId = float(j)/pixelCount*m_params.oiopRegionNum;
					int origId = pix.id;
					float gray = pixelSample_data[i*pixelSample_step+origId];
					for (k=0;k<fenceNum;k++)
					{
						if (gray<= subFenceGray[regionId][k])
							break;
					}
					pix.o_pattern += k*posWeight;
				}
				posWeight*=m_params.oiopQuantLevel;
			}

			int o_patternWidth = pow((float)m_params.oiopQuantLevel,sampleNum);
			int dim = o_patternWidth*m_params.oiopRegionNum;

			int curId = 0;
			int lastId = 0;
			if (pixelCount >= m_params.oiopRegionNum)
			{


				for (i=0; i<m_params.oiopRegionNum; i++)
				{
					int fenceId = pixelCount*(i+1)/m_params.oiopRegionNum-1;
					float fenceGray = pixel[fenceId].f_gray;
					int regionId = i;
					curId = lastId;

					while (true)
					{
						if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
						{
							lastId = curId;
							break;
						}

						int id = regionId*o_patternWidth+pixel[curId].o_pattern;
						des[id] += pixel[curId].weight;
						curId++;
					}

					while (true)
					{
						if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
							break;

						int id = regionId*o_patternWidth+pixel[curId].o_pattern;
						des[id] += pixel[curId].weight;
						curId++;
					}
				}
				ThreshNorm(des,dim, 1.f);
			}

		}
		break;



	case 2:
	{

		int sampleNum = m_params.oiopNum;
		Mat pixelSample(sampleNum,inPatchSz*inPatchSz,CV_32FC1);
		float* pixelSample_data = (float*)pixelSample.data;
		int pixelSample_step = pixelSample.step1();

		float theta = 2*CV_PI/m_params.oiopNum;
		int pixelCount = 0;

		for (y=-inRadius_left; y<=inRadius_right; ++y)
		{
			for (x=-inRadius_left; x<=inRadius_right; ++x)
			{
				float dis2 = (float)(x*x + y*y);
				if(dis2 > inRadius2)
				{
					continue;
				}

				float cur_gray = out_data[(y+outRadius)*out_step+x+outRadius];
				uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
				if (cur_flag == 0)
				{
					continue;
				}

				float nDirX = x;
				float nDirY = y;
				float nOri = atan2(nDirY, nDirX);
				if (fabs(nOri - CV_PI) < INFMIN_F)
				{
					nOri = -CV_PI;
				}

				bool isInBound = true;
				int radius = m_params.lsRadius;
				for (k=0; k<m_params.oiopNum; k++)
				{
					float deltaX = radius*cos(nOri+k*theta);
					float deltaY = radius*sin(nOri+k*theta);

					float sampleX = x+deltaX;
					float sampleY = y+deltaY;
					float gray;

					if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
					{
						isInBound = false;
						break;
					}

					pixelSample_data[k*pixelSample_step+pixelCount] = gray;
				}

				if (!isInBound)
				{
					continue;
				}

				Pixel pix;
				pix.x = x;
				pix.y = y;
				pix.f_gray = cur_gray;
				pix.i_gray = (int)(pix.f_gray*255+0.5);
				pix.weight = 1;
				pix.id = pixelCount;	//记录id
				pix.o_pattern = 0;
				pixel[pixelCount++] = pix;
			}
		}


		sort(pixel, pixel+pixelCount, fGrayComp);

		float subFenceGray[MAX_REGION_NUM];
		for (i=0; i<m_params.oiopQuantLevel-1; i++)
		{
			int fenceId = float(i+1)/m_params.oiopQuantLevel*pixelCount-1;
			subFenceGray[i] = pixel[fenceId].f_gray;
		}

		//compute the oiop pattern
		int posWeight = 1;
		for (i=0; i<sampleNum; i++)
		{
			for (j=0; j<pixelCount; j++)
			{

				Pixel& pix = pixel[j];

				int origId = pix.id;
				float gray = pixelSample_data[i*pixelSample_step+origId];
				for (k=0;k<m_params.oiopQuantLevel-1;k++)
				{
					if (gray<= subFenceGray[k])
						break;
				}
				pix.o_pattern += k*posWeight;
			}
			posWeight*=m_params.oiopQuantLevel;
		}

		int o_patternWidth = pow((float)m_params.oiopQuantLevel,sampleNum);
		int dim = o_patternWidth*m_params.oiopRegionNum;

		if (pixelCount >= m_params.oiopRegionNum)
		{

			int curId = 0;
			int lastId = 0;
			for (i=0; i<m_params.oiopRegionNum; i++)
			{
				int fenceId = pixelCount*(i+1)/m_params.oiopRegionNum-1;
				float fenceGray = pixel[fenceId].f_gray;
				int regionId = i;
				curId = lastId;

				while (true)
				{
					if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
					{
						lastId = curId;
						break;
					}

					int id = regionId*o_patternWidth+pixel[curId].o_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}

				while (true)
				{
					if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
						break;

					int id = regionId*o_patternWidth+pixel[curId].o_pattern;
					des[id] += pixel[curId].weight;
					curId++;
				}
			}

			ThreshNorm(des,dim, 1.f);
		}
	}
	break;


	}

	delete [] pixel;
}

//simply implementation of MIOP without considering speed issue
void MyDescriptors::createMIOP(const Mat& outPatch, const Mat& flagPatch, int inPatchSz, float* des)  const
{
	int l_patternWidth = m_params.liopNum == 3 ? 6 : 24;
	int liop_dim = (l_patternWidth*m_params.liopRegionNum);

	createLIOP(outPatch, flagPatch, inPatchSz, des);
	createOIOP(outPatch, flagPatch, inPatchSz, des+liop_dim);
}


//liopType=1:LIOP + oiopType=12:OIOP, fast implementation, share the global order
void MyDescriptors::createMIOP_FAST(const Mat& outPatch, const Mat& flagPatch, int inPatchSz, float* des)  const
{
	int i,j,k,x,y;

	float* out_data = (float *)outPatch.data;
	int out_step = outPatch.step1();
	uchar* flag_data = (uchar*)flagPatch.data;
	int flag_step = flagPatch.step1();

	int inRadius_left = inPatchSz/2;
	int inRadius_right = inPatchSz - inRadius_left - 1;
	float inRadius2 = (float)(inRadius_left*inRadius_left);

	int outRadius = outPatch.cols / 2;

	Pixel* pixel = new Pixel[inPatchSz*inPatchSz];

	//for liop
	int		liop_idx[MAX_SAMPLE_NUM];
	float	liop_dst[MAX_SAMPLE_NUM];
	float	liop_src[MAX_SAMPLE_NUM];
	float	liop_theta = 2*CV_PI/m_params.liopNum;

	//for oiop
	float oiop_theta = 2*CV_PI/m_params.oiopNum;

	Mat pixelSample(m_params.oiopNum,inPatchSz*inPatchSz,CV_32FC1);
	float* pixelSample_data = (float*)pixelSample.data;
	int pixelSample_step = pixelSample.step1();


	int pixelCount = 0;
	for (y=-inRadius_left; y<=inRadius_right; ++y)
	{
		for (x=-inRadius_left; x<=inRadius_right; ++x)
		{

			float dis2 = (float)(x*x + y*y);
			if(dis2 > inRadius2)
			{
				continue;
			}

			float cur_gray = out_data [(y+outRadius)*out_step +x+outRadius];
			uchar cur_flag = flag_data[(y+outRadius)*flag_step+x+outRadius];
			if (cur_flag == 0)
			{
				continue;
			}

			float nDirX = x;
			float nDirY = y;
			float nOri = atan2(nDirY, nDirX);
			if (fabs(nOri - CV_PI) < INFMIN_F)
			{
				nOri = -CV_PI;
			}

			//for liop
			bool isInBound = true;
			for (k=0; k<m_params.liopNum; k++)
			{
				float deltaX = m_params.lsRadius*cos(nOri+k*liop_theta);
				float deltaY = m_params.lsRadius*sin(nOri+k*liop_theta);

				float sampleX = x+deltaX;
				float sampleY = y+deltaY;
				float gray;

				if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
				{
					isInBound = false;
					break;
				}

				liop_src[k] = gray;
			}

			if (!isInBound)
			{
				continue;
			}

			//one problem: this requires the samples of both oiop and liop are inside the image
			//for oiop
			for (k=0; k<m_params.oiopNum; k++)
			{
				float deltaX = m_params.lsRadius*cos(nOri+k*oiop_theta);
				float deltaY = m_params.lsRadius*sin(nOri+k*oiop_theta);

				float sampleX = x+deltaX;
				float sampleY = y+deltaY;
				float gray;

				if (!BilinearInterF2FValid(&gray, sampleX+outRadius,sampleY+outRadius,outPatch, flagPatch) )
				{
					isInBound = false;
					break;
				}

				pixelSample_data[k*pixelSample_step+pixelCount] = gray;
			}

			if (!isInBound)
			{
				continue;
			}



			//for liop
			int key = 0;
			SortGray(liop_dst, liop_idx, liop_src, m_params.liopNum);
			for (k=0; k<m_params.liopNum; k++)
			{
				key += (liop_idx[k]+1)*m_params.pLiopPosWeight[m_params.liopNum-k-1];
			}
			map<int, int>::iterator iter;
			iter = m_params.pLiopPatternMap->find(key);

			Pixel pix;
			pix.x = x;
			pix.y = y;
			pix.f_gray = cur_gray;
			pix.i_gray = (int)(pix.f_gray*255+0.5f);
			pix.weight = 1;
			pix.l_pattern = iter->second;
			pix.id = pixelCount;
			pix.o_pattern = 0;
			pixel[pixelCount++] = pix;
		}
	}

	sort(pixel, pixel+pixelCount, fGrayComp);

	//for liop
	int l_patternWidth = m_params.liopNum == 3 ? 6 : 24;
	int liop_dim = l_patternWidth*m_params.liopRegionNum;

	if (pixelCount >= m_params.liopRegionNum)
	{
		int curId = 0;
		int lastId = 0;
		for (i=0; i<m_params.liopRegionNum; i++)
		{
			int fenceId = pixelCount*(i+1)/m_params.liopRegionNum-1;
			float fenceGray = pixel[fenceId].f_gray;
			int regionId = i;
			curId = lastId;

			while (true)
			{
				if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
				{
					lastId = curId;
					break;
				}

				int id = regionId*l_patternWidth+pixel[curId].l_pattern;
				des[id] += pixel[curId].weight;
				curId++;
			}

			while (true)
			{
				if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
					break;

				int id = regionId*l_patternWidth+pixel[curId].l_pattern;
				des[id] += pixel[curId].weight;
				curId++;
			}
		}

		ThreshNorm(des,liop_dim,1.f);
	}

	//for oiop
	float subFenceGray[MAX_REGION_NUM][MAX_REGION_NUM];
	int fenceNum = m_params.oiopQuantLevel-1;

	int subFenceId;
	for (k=0; k<m_params.oiopRegionNum; k++)
	{
		for (i=0; i<fenceNum; i++)
		{
			subFenceId = int(m_fenceRatio[k*fenceNum+i]*pixelCount);
			subFenceGray[k][i] = pixel[subFenceId].f_gray;
		}
	}

	int posWeight = 1;
	for (i=0; i<m_params.oiopNum; i++)
	{
		for (j=0; j<pixelCount; j++)
		{

			Pixel& pix = pixel[j];


			int regionId = float(j)/pixelCount*m_params.oiopRegionNum;
			int origId = pix.id;
			float gray = pixelSample_data[i*pixelSample_step+origId];
			for (k=0;k<fenceNum;k++)
			{
				if (gray<= subFenceGray[regionId][k])
					break;
			}
			pix.o_pattern += k*posWeight;
		}
		posWeight*=m_params.oiopQuantLevel;
	}

	int o_patternWidth = pow((float)m_params.oiopQuantLevel,m_params.oiopNum);
	int oiop_dim = o_patternWidth*m_params.oiopRegionNum;

	int curId = 0;
	int lastId = 0;
	float* oiop_des = des+liop_dim;
	if (pixelCount >= m_params.oiopRegionNum)
	{
		for (i=0; i<m_params.oiopRegionNum; i++)
		{
			int fenceId = pixelCount*(i+1)/m_params.oiopRegionNum-1;
			float fenceGray = pixel[fenceId].f_gray;
			int regionId = i;
			curId = lastId;

			while (true)
			{
				if (fabs(pixel[curId].f_gray-fenceGray) < INFMIN_F)
				{
					lastId = curId;
					break;
				}

				int id = regionId*o_patternWidth+pixel[curId].o_pattern;
				oiop_des[id] += pixel[curId].weight;
				curId++;
			}

			while (true)
			{
				if(curId==pixelCount || pixel[curId].f_gray>fenceGray)
					break;

				int id = regionId*l_patternWidth+pixel[curId].o_pattern;
				oiop_des[id] += pixel[curId].weight;
				curId++;
			}
		}

		ThreshNorm(oiop_des,oiop_dim, 1.f);
	}

	delete [] pixel;

}


bool MyDescriptors::readPCA(const string& file)
{
	ifstream ifs(file.c_str());
	if (!ifs.is_open())
	{
		cout<<"Can not read feature projection matrix: "<<file<<endl;
		return false;
	}

	ReadMatrix(ifs, m_PCAMean);

	Mat PCABasis;
	ReadMatrix(ifs, PCABasis);
	assert(m_params.PCABasisNum <= PCABasis.cols);
	Mat tmp = PCABasis.colRange(0,m_params.PCABasisNum);
	tmp.copyTo(m_PCABasis);

	ifs.close();
	return true;

}

