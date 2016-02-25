#ifndef _UTILS_H_
#define _UTILS_H_


#include "Common.h"
#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

bool NormalizePatch_ROI(const Mat& image, Mat& outPatch, Mat& flagPatch, const AffineKeyPoint& kp, const float origTrans[4], const float axis[2], float mrScale, int normPatchWidth, float initSigma, float domiOrien);
void ThreshNorm(float* des, int dim, float thresh);

inline float BilinearInterU2F(float x, float y, const Mat& image);
inline float BilinearInterF2F(float x, float y, const Mat& image);
bool BilinearInterF2FValid(float* val, float x, float y, const Mat& image, const Mat& flagImage);

void CalTrans(const AffineKeyPoint& kp, float trans[4], float axis[2]);

bool ReadKpts(const string& region_file, vector<AffineKeyPoint>&kpts);
void WriteDess(const string& des_file, const vector<AffineKeyPoint>&kpts, const Mat& dess, DES_FORMAT desFormat);


bool fGrayComp(Pixel p1, Pixel p2);
void SortGray(float* dst, int* idx, float* src, int len);


void ReadMatchInfo(const string& match_file, vector<MATCH_INFO>& match_info_vec);
bool ReadImgList(const string& img_list_file, vector<string>& img_file_list);
void GetKpList(const string& kp_ext, const vector<string> img_file_list, vector<string>& kp_file_list);

void ReadMatrix(ifstream& ifs, Mat& mat);
void WriteMatrix(ofstream& ofs, const Mat& mat);

#endif
