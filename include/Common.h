//////////////////////////////////////////////////////////////////////////
// 
// NAME
//  Common.h -- common definitions and functions.
// 
// DESCRIPTION
//  This file includes some common used definitions and functions, and 
//  some necessary header files for this project.
// 
//////////////////////////////////////////////////////////////////////////

#define TEST_TIME

#ifndef COMMON_H
#define COMMON_H


#include <cfloat>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <bitset>
#include "assert.h"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

class AffineKeyPoint
{
public:

	AffineKeyPoint() : pt(0,0), a(1.f), b(0.f), c(1.f), angle(-1), response(0), octave(0), class_id(-1)  {}

	AffineKeyPoint(Point2f _pt, float _a, float _b, float _c, float _size=1, float _angle=-1,
		float _response=0, int _octave=0, int _class_id=-1)
		: pt(_pt), a(_a), b(_b), c(_c), angle(_angle),
		response(_response), octave(_octave), class_id(_class_id){}

	AffineKeyPoint(float x, float y, float _a, float _b, float _c, float _angle=-1,
		float _response=0, int _octave=0, int _class_id=-1)
		: pt(x, y), a(_a), b(_b), c(_c), angle(_angle),
		response(_response), octave(_octave), class_id(_class_id) {}

	Point2f pt; //!< coordinates of the keypoints

	//parameters for the ellipse
	float a;
	float b;
	float c;


	 float angle; //!< computed orientation of the keypoint (-1 if not applicable);
	//!< it's in [0,360) degrees and measured relative to
	//!< image coordinate system, ie in clockwise.
	 float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
	 int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
	 int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)



};


typedef unsigned char uchar;
enum DES_TYPE{	UNDEFINE,
	LIOP,
	OIOP,
	MIOP,
	MIOP_FAST
};

//vgg affine  x,y,a,b,c
//lowe affine y,x,scal,orien,a,b,c
enum DES_FORMAT{DES_INT,DES_FLOAT};

const float INFMIN_F = FLT_EPSILON;
const double INFMIN_D = DBL_EPSILON;

const double MAX_DIS = 1000000.0;


template <class T>
inline void Swap(T &v1, T &v2)
{
	T temp = v1;
	v1 = v2;
	v2 = temp;
}

template <class T>
inline T Max(T x, T y)
{
	return (x > y) ? x : y;
}

template <class T>
inline T Min(T x, T y)
{
	return (x < y) ? x : y;
}


// Simple error handling
inline void FatalError(const char *msg)
{
	cerr << msg << endl;
	exit(1);
}

void GeneratePatternMap(std::map<int,int>**pattern_map, int** pos_weight, int n);
bool NextPermutation(int *p, int n);

typedef struct _Params{
	double initSigma;

	float mrScale;			// the ratio between Measure Region and Detected Retion
	int normPatchWidth;		// the size of the normalized patch, set to 41 as in the paper

	DES_FORMAT desFormat;	//
	DES_TYPE des_type;		// type of the descriptors, LIOP, OIOP, MIOP, MIOP_FAST

	int liopType;			// weighting type of LIOP
	int liopRegionNum;		// number of ordinal bins
	int liopNum;			// number of sampling points around each pixel
	double liopThre;			//threshold for weighting function used in ICCV paper
	map<int,int>* pLiopPatternMap;	// LIOP index tabel
	int * pLiopPosWeight;	// weights for encoding LIOP to  a decimal number


	int oiopType;			// quantization strategy of OIOP
	int oiopRegionNum;		// number of ordinal bins
	int oiopQuantLevel;		// number of quantization level
	int oiopNum;			// number of sampling points around each pixel

	int lsRadius;	//local sample radius;
	double nSigma;	//Gaussian sigma after normalization


	int srNum;	//support region number
	float srScaleStep;	//scale step between mr

	bool isAffine;		// affine covariant region or not

	string PCAFile;		// pca parameters for MIOP
	int PCABasisNum;	// the dimension after dimension reduction
	int isApplyPCA;		// apply pca on MIOP or not

	_Params()
	{
		this->initSigma = 0.0;
		this->mrScale = 3.0;
		this->normPatchWidth = 41;

		this->desFormat = DES_INT;

		this->oiopType = 1;
		this->oiopRegionNum = 4;
		this->oiopQuantLevel = 4;
		this->oiopNum = 3;

		this->liopType = 1;
		this->liopRegionNum = 6;
		this->liopNum = 4;
		this->pLiopPatternMap = NULL;
		this->pLiopPosWeight = NULL;
		GeneratePatternMap(&this->pLiopPatternMap, &this->pLiopPosWeight, this->liopNum);

		this->lsRadius = 6;
		this->nSigma = 1.2;
		this->liopThre = 5.0;
		this->des_type = LIOP;
		this->srNum = 1;
		this->srScaleStep = 1.5f;

		this->isAffine = true;

		this->PCAFile = "pca_miop.txt";
		this->PCABasisNum = 128;
		this->isApplyPCA = 0;
	}

	~_Params()
	{
		if (this->pLiopPatternMap!=NULL)
		{
			delete this->pLiopPatternMap;
			pLiopPatternMap = NULL;
		}
		if(this->pLiopPosWeight!=NULL)
		{
			delete [] this->pLiopPosWeight;
			pLiopPosWeight = NULL;
		}
	}

}Params;

typedef struct _Pixel{
	float x;
	float y;
	float angle;	//angle of this pixel relative to the center of the patch
	float f_gray;
	int i_gray;
	float weight;
	int o_pattern;	//overall pattern
	int l_pattern;	//local pattern
	int id;

	_Pixel()
	{
		x = 0.f;
		y = 0.f;
		angle = 0.f;
		f_gray = 0.f;
		i_gray = 0;
		weight = 0.f;
		o_pattern = 0;
		l_pattern = 0;
		id = 0;
	}


	inline _Pixel& operator=(const _Pixel& p)
	{
		this->x = p.x;
		this->y = p.y;
		this->angle = p.angle;
		this->f_gray = p.f_gray;
		this->i_gray = p.i_gray;
		this->o_pattern = p.o_pattern;
		this->l_pattern = p.l_pattern;
		this->id = p.id;
		this->weight = p.weight;
		return *this;
	}
}Pixel;

//////////////////////////////////////////////////////////
// Sture for Patch Dataset
typedef struct _DIR_INFO{
	string dir_name;
	int img_num;
	int last_img_patch_num;
	int patch_num;
}DIR_INFO;

typedef struct _MATCH_INFO
{
	int m_patchId1;
	int m_3dPointId1;
	int m_patchId2;
	int m_3dPointId2;
	bool isCorrect;
}MATCH_INFO;



const int IMG_LENGTH = 1024;
const int PATCH_LENGTH = 64;
const int PATCH_PER_ROW = IMG_LENGTH / PATCH_LENGTH;
const int PATCH_PER_COL = IMG_LENGTH / PATCH_LENGTH;
const int PATCH_PER_IMG = PATCH_PER_ROW * PATCH_PER_COL;

const string ALL_MATCH_FILES [9] = {"m50_1000_1000_0.txt",    "m50_2000_2000_0.txt",     "m50_5000_5000_0.txt",
	"m50_10000_10000_0.txt",  "m50_20000_20000_0.txt",   "m50_50000_50000_0.txt",
	"m50_100000_100000_0.txt","m50_200000_200000_0.txt", "m50_500000_500000_0.txt"};

#endif

