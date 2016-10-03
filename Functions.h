#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "imgproc/imgproc.hpp"

#include <iostream>
#include <string>

#include "mclmcrrt.h"
#include "mclcppclass.h"

using namespace cv;
using namespace std;

extern "C" {

bool MW_CALL_CONV OMPInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

bool MW_CALL_CONV TrainDLInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

bool MW_CALL_CONV OMPInitialize(void);

bool MW_CALL_CONV TrainDLInitialize(void);

bool MW_CALL_CONV DispDictInitialize(void);

void MW_CALL_CONV OMPTerminate(void);

void MW_CALL_CONV TrainDLTerminate(void);

void MW_CALL_CONV DispDictTerminate(void);

void MW_CALL_CONV OMPPrintStackTrace(void);

void MW_CALL_CONV TrainDLPrintStackTrace(void);

void MW_CALL_CONV DispDictPrintStackTrace(void);

bool MW_CALL_CONV mlxOMP(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

bool MW_CALL_CONV mlxMexTrainDL(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

bool MW_CALL_CONV mlxChapter_12_DispDict(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

}

extern  __declspec(dllimport) void MW_CALL_CONV OMP(int nargout, mwArray& CoefMatrix, const mwArray& Data, const mwArray& Dictionary, const mwArray& param);

extern  __declspec(dllimport) void MW_CALL_CONV mexTrainDL(int nargout, mwArray& varargout, const mwArray& varargin);

extern __declspec(dllimport) void MW_CALL_CONV Chapter_12_DispDict(int nargout, mwArray& I, const mwArray& D, const mwArray& numRows, const mwArray& numCols, const mwArray& X, const mwArray& Y, const mwArray& sortVarFlag);


Mat CreateTrainingSet(Mat I, int n);

void save2txt(Mat data,string str);

Mat readData(int block_size, int data_size, string xpath);

Mat ImageRecover(int n, int N1, int N2, Mat Dalpha);

Mat HOG(Mat atom, Mat Var);

int is_file_exists(string fname);

Mat bfltGray(Mat A,int w,double sigma_d,double sigma_r);

Mat OMP_Cholesky(Mat D, Mat X,double eps0);

void DictPartition(Mat &dict, Mat &Dict_rain, Mat &Dict_geometry);

Mat LibOMP(Mat dict, Mat X, int l, double eps0);

Mat SpMultiply(Mat D, Mat SpMatrix);

void DictLearn_KSVD(Mat & A, Mat y,int codebook_size,int ksvd_iter);

Mat LibDictLearn(Mat data, int K, double lambda, int iter);

Mat DispDict(Mat dict, int rows, int cols, int x, int y, float sort);