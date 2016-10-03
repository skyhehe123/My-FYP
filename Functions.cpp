#include "stdafx.h"
#include "Functions.h"

Mat CreateTrainingSet(Mat I, int n)
{
	int N1=I.rows;
	int N2=I.cols;
	int cnt=0;
	Mat data=Mat::zeros(n*n, (N1 - n + 1) * (N2 - n + 1),CV_32F);
	Mat patch;

	for (int j = 0;j<(N2-n+1);j++)
       for (int i = 0;i<(N1-n+1);i++)
       {
		   /*patch = I(Rect(j,i,n,n)).clone();
		   patch.reshape(0,n*n).copyTo(patches.col(cnt)); 
           cnt++;*/ 
		   for (int k =0; k<n;k++){
			   for(int l=0;l<n;l++)
			        data.at<float>(l+n*k,i+j*(N1-n+1))=I.at<float>(i+l,j+k);
					
		   }
	   }
    //Demean and normalization 
	Mat mean;
	reduce(data, mean, 0, CV_REDUCE_AVG);
	data = data - repeat(mean, data.rows, 1);
	for(int i = 0; i < data.cols; i++)
    {	
		data.col(i)=data.col(i)/cv::norm(data.col(i),NORM_L2);
    }
	return data;
}

void save2txt(Mat data,string str){
   
    FILE *pOut = fopen(str.c_str(), "w");
    for(int i=0; i<data.rows; i++){
        for(int j=0; j<data.cols; j++){
            fprintf(pOut, "%lf", data.at<float>(i, j));
            if(j == data.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}
Mat  readData(int rows, int cols, string xpath){
    Mat data = Mat::zeros(rows, cols,CV_32F);
    FILE *streamX;
    streamX = fopen(xpath.c_str(), "r");
    double tpdouble;
    int counter = 0;
    while(1){
        if(fscanf(streamX, "%lf", &tpdouble) == EOF) break;
        data.at<float>(counter / cols, counter % cols) = tpdouble;
		
        ++ counter;
    }
    fclose(streamX);
	return data;

}
void DictPartition(Mat &dict, Mat &Dict_rain, Mat &Dict_geometry){
	   Mat PHOG(9,dict.cols,CV_32F);         //9 bins of histogram, 1024 atoms
	   Mat var = Mat(1,dict.cols,CV_32F);
	   Mat col,atom;
	   for(int i=0;i<dict.cols;i++)
	   { 
		   col=dict.col(i).clone();
		   atom=col.reshape(0,std::sqrt((double)dict.rows));
		   atom=atom.t();
		   HOG(atom, var.col(i)).copyTo(PHOG.col(i));
	   }

	   PHOG=PHOG.t();
	   Mat labels,centers;

	   kmeans(PHOG,2, labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.001), 3, KMEANS_PP_CENTERS, centers);

	   Mat subDict1, subDict2; 
	   vector<float> var1, var2;
	   for(int j=0;j<dict.cols;j++)
	   {
		   if(labels.at<int>(j)==1)
		   {
			   var1.push_back(var.at<float>(0,j));
			   if(subDict1.empty())
				   subDict1=dict.col(j);
			   else
			       hconcat(subDict1,dict.col(j),subDict1);
		   }
		   else 
		   {
			   var2.push_back(var.at<float>(0,j));
			   if(subDict2.empty())
				   subDict2=dict.col(j);
			   else
			       hconcat(subDict2,dict.col(j),subDict2);
		   }
	   }
	   
	   if(mean(var1)[0]>mean(var2)[0])
	   {
		   //Dict_rain = subDict2;
		   //Dict_geometry = subDict1;
		   Dict_rain = subDict1;
		   Dict_geometry = subDict2;
	   }
	   else
	   {
		   //Dict_rain = subDict1;
		   //Dict_geometry = subDict2;
		   Dict_rain = subDict2;
		   Dict_geometry = subDict1;
	   }
	   //concatenat two dictionary
	   hconcat(Dict_rain,Dict_geometry,dict);

}
Mat ImageRecover(int n, int N1, int N2, Mat Dalpha)
{
	Mat yout=Mat::zeros(N1, N2,CV_32F);
    Mat Weight=Mat::zeros(N1,N2,CV_32F); 
	int i=0, j=0;
    for (int k=0;k<(N1-n+1)*(N2-n+1);k++)
    {
	  Mat patch;
	  Mat kcol=Dalpha.col(k).clone();
	  kcol.reshape(0,n).copyTo(patch);
	  patch=patch.t();
      yout(Rect(j,i,n,n))=yout(Rect(j,i,n,n))+patch; 
      Weight(Rect(j,i,n,n))=Weight(Rect(j,i,n,n))+1; 
   
	  if (i<N1-n) 
         i=i+1; 
      else
	  {
		 i=0; 
		 j=j+1; 
	  }
   
    }
    yout=yout/(Weight); 
    return yout;
}
Mat HOG(Mat atom, Mat Var)
{
	 // cout<<atom<<endl;
	  copyMakeBorder(atom,atom,1,1,1,1,BORDER_CONSTANT,Scalar(0.0));
 
      Mat grad_xr,grad_yu;
      //Mat kern_x = (Mat_<float>(1,3) <<  -1, 0,  1);
	  //Mat kern_y = (Mat_<float>(3,1) <<  -1, 0,  1);
	  Mat kern_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
	  Mat kern_y = (Mat_<float>(3,3) <<  -1,-2,-1, 0,0,0, 1,2, 1);
	  filter2D(atom, grad_xr, atom.depth(), kern_x);
	  filter2D(atom, grad_yu, atom.depth(), kern_y);
	  grad_xr=grad_xr(Range(1,grad_xr.rows-1),Range(1,grad_xr.cols-1));
	  grad_yu=grad_yu(Range(1,grad_yu.rows-1),Range(1,grad_yu.cols-1));
	  

	  Mat thetas(grad_xr.rows,grad_xr.cols,CV_32F),angles(grad_xr.rows,grad_xr.cols,CV_32F),magnit(grad_xr.rows,grad_xr.cols,CV_32F);
	  for(int r=0;r<grad_xr.rows;r++)
		  for(int c=0;c<grad_xr.cols;c++)
		  {
			  thetas.at<float>(r,c)=atan2(grad_yu.at<float>(r,c),grad_xr.at<float>(r,c));		
			  magnit.at<float>(r,c)=sqrt((grad_yu.at<float>(r,c)*grad_yu.at<float>(r,c)+grad_xr.at<float>(r,c)*grad_xr.at<float>(r,c)));
		  }
		

	   angles = thetas*180/(atan(1.0)*4);
	   
	   atom=atom(Range(1,atom.rows-1),Range(1,atom.cols-1));
	   //double min,max;
	   //minMaxLoc(atom, &min, &max, NULL, NULL );
	   //Mat Strech_atom=(atom-min)/max;
       //Mat stdDev;
	   //meanStdDev (Strech_atom, noArray(), stdDev);
	   //Var.at<float>(0,0)=stdDev.dot(stdDev);
	  

	   Mat hist;
	   hist=Mat::zeros(9,1,CV_32F);
	   int bin=0;
	   for (float ang_lim=-140;ang_lim<=180;ang_lim=ang_lim+40){
        
             for(int r=0;r<angles.rows;r++){
		        for(int c=0;c<angles.cols;c++)
                 if (angles.at<float>(r,c)<ang_lim)
				 {
                    angles.at<float>(r,c)=9999;
                    hist.at<float>(bin,0)=hist.at<float>(bin,0)+magnit.at<float>(r,c);
				 }}
		    bin=bin+1;		
	   }
     
		hist=hist/cv::norm(hist,NORM_L2);
		//cout<<"histogram:"<<hist<<endl;
		double min,max;
	    minMaxLoc(hist, &min, &max, NULL, NULL );
		Var.at<float>(0,0)=max-min;
		return hist;


}
int is_file_exists(string fname)
{
    FILE *file;
	if (file = fopen(fname.c_str(), "r"))
    {
        fclose(file);
        return 1;
    }
    return 0;
}
Mat bfltGray(Mat A,int w,double sigma_d,double sigma_r)
{
// Pre-compute Gaussian distance weights.

    Mat X, Y;  
	//meshgrid(-w:w,-w:w)
	std::vector<float> t_x, t_y;  
	for(int i = Range(-w, w).start; i <= Range(-w, w).end; i++) t_x.push_back(i);  
    for(int j = Range(-w, w).start; j <= Range(-w, w).end; j++) t_y.push_back(j);  
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);  
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y); 

    pow(X, 2.0, X);
    pow(Y, 2.0, Y);
    Mat G,B; 
    exp((-(X+Y))/(2*sigma_d*sigma_d),G);
	
	int iMin,iMax,jMin,jMax;
    Mat I,H,F;
    B = Mat::zeros(A.rows, A.cols,CV_32F);
    for (int i = 0;i<A.rows;i++)
    {
    for (int j = 0;j<A.cols;j++)
    {
         // Extract local region.
         iMin = max(i-w+1,1);
		 iMax = min(i+w+1,A.rows);
         jMin = max(j-w+1,1);
         jMax = min(j+w+1,A.cols);
		 I=A(Rect(jMin-1,iMin-1,jMax-jMin+1,iMax-iMin+1)).clone();
         
         // Compute Gaussian intensity weights.
		 Mat temp=I-A.at<float>(i,j);
		 pow(temp, 2.0, temp);
		 exp((-temp/(2*sigma_r*sigma_r)),H);
         
         // Calculate bilateral filter response.
		 F =H.mul(G(Rect(jMin-j+w-1,iMin-i+w-1,jMax-jMin+1,iMax-iMin+1)));
         B.at<float>(i,j) =float( sum(F.mul(I))(0)/sum(F)(0));     
     }
     }
	return B;
}
Mat OMP_Cholesky(Mat D, Mat X,double eps0)
{
int P=X.cols;
Mat G=D.t()*D;
//Sparse codes matrix A
Mat A = Mat::zeros(D.cols,P,CV_32F);
cout<<"Orthoganal Matching Pursuit with no parameter"<<endl;
cout<<"Dictionary size: "<<D.cols<<endl;
cout<<"Number of signals: "<<P<<endl;
cout<<"signal dimension: "<<X.rows<<endl;
cout<<"****Orthoganal Matching Pursuit***"<<endl;
for (int k=0;k<P;k++)
{
   Mat x=X.col(k);
   double eps=x.dot(x);
   Mat L = (Mat_<float>(1,1) <<  1);
   Mat alpha0=D.t()*x;
   Mat alpha=alpha0.clone();
   double pre_sigma=0;
   int j=0;
   int pos=0;
   vector<int> indx;
   while (1)
   {
      Point maxLoc;
	  minMaxLoc(abs(alpha), NULL, NULL, NULL, &maxLoc );
	  pos=maxLoc.y;

      if(j>0)
	  {
	      //solve Lw=G(indx,pos) for w (forward substitution)
		  Mat w(L.rows,1,CV_32F);
	      for (int i = 0; i < indx.size(); i++){
            float a = G.at<float>(indx[i],pos);
			for(int j=0;j<i;j++)
			   a = a - L.at<float>(i,j)*w.at<float>(j,0);
		    w.at<float>(i,0)=a/L.at<float>(i,i);
          }
		  
		  //Update L
          Mat z=Mat::zeros(L.rows+1,1,CV_32F);
		  z.at<float>(L.rows,0)=sqrt(1-w.dot(w));
		  Mat w_t=w.t();
	      L.push_back(w_t);
	      hconcat(L, z, L);
      }

      indx.push_back(pos);  

	  //solve LL^t *gamma = alpha0(indx,:) for gamma (forward & backward substitution)
	  Mat gamma(L.cols,1,CV_32F);
	  //forward substitution
	  Mat y(L.rows,1,CV_32F);
	  for (int i = 0; i < indx.size(); i++){
            float a = alpha0.at<float>(indx[i],0);
			for(int j=0;j<i;j++)
			   a = a - L.at<float>(i,j)*y.at<float>(j,0);
		    y.at<float>(i,0)=a/L.at<float>(i,i);
	  }
	  //backward substitution
	  for (int i=y.rows-1;i>=0;i--){
			float a=y.at<float>(i,0);
			for(int j=i+1;j<=y.rows-1;j++)
			   a = a - L.at<float>(j,i)*gamma.at<float>(j,0);
            gamma.at<float>(i,0)=a/L.at<float>(i,i);
      }

	  //beta=G(:,indx)*gamma
	  Mat beta = Mat::zeros(G.rows,1,CV_32F);
	  for (int i = 0; i < indx.size(); i++){
            beta += G.col(indx[i])*gamma.row(i);
      }
	  
      alpha=alpha0-beta;

	  //sigma = gamma^t * beta(indx,:)
	  double sigma=0;
	  for (int i = 0; i < indx.size(); i++){
          sigma +=  beta.at<float>(indx[i],0) * gamma.at<float>(i,0);
      }
	  
      eps=eps-sigma+pre_sigma;

      j=j+1;
      pre_sigma=sigma;
	  
      if(eps<eps0)
	  {
		  for (int i = 0; i < indx.size(); i++)
             A.at<float>(indx[i],k)=  gamma.at<float>(i,0);
          break; 
	  }
   }
   
}
return A;
} 
Mat LibOMP(Mat dict, Mat X, int l, double eps0){

       mwArray D(X.rows,X.cols,mxSINGLE_CLASS);
       X=X.t();
       X=X.reshape(0,X.cols);
       D.SetData((float *)X.data,X.rows*X.cols);

	   mwArray L(l);
	   const char* fieldnames[] = {
           "L","eps"
       };
	   mwArray eps(eps0);
	   mwArray param2(2,1,2,fieldnames);
	   param2.Get("L",1,1).Set(L);
	   param2.Get("eps",1,1).Set(eps);
	  
	   mwArray Dictionary(dict.rows,dict.cols,mxSINGLE_CLASS);
	   cout<<dict.rows<<" "<<dict.cols<<endl;
	   dict=dict.t();
	   dict=dict.reshape(0,dict.cols);
	   
	   Dictionary.SetData((float *)dict.data,dict.rows*dict.cols);
	
	   Mat SparseCodes(X.cols,dict.cols,CV_32F);
		
	   mwArray CoefMat;
	   OMP(1,CoefMat,D,Dictionary,param2);
	   CoefMat.GetData((float *)SparseCodes.data,dict.cols*X.cols);
	   SparseCodes=SparseCodes.t();
	   
	   return SparseCodes;

}
Mat SpMultiply(Mat D, Mat SpMatrix){
	
	SparseMat Sp=cv::SparseMat(SpMatrix);

	Mat Result = Mat::zeros(D.rows,SpMatrix.cols,CV_32F);
	
	SparseMatConstIterator_<float>
    it = Sp.begin<float>(),
    it_end = Sp.end<float>();

	
	for(; it != it_end; ++it)
    {
   
    const SparseMat::Node* n = it.node();
	int x=n->idx[0],y=n->idx[1];
	float val=it.value<float>();
	   for(int i=0;i<D.rows;i++){
		   //Result.at<float>(i,y)+=D.at<float>(i,x)*val;
		   Result.ptr<float>(i)[y] += val * D.ptr<float>(i)[x];
	   }
	}

	return Result;
}
void DictLearn_KSVD(Mat & A, Mat y,int codebook_size,int ksvd_iter)
{
//==============================
//input parameter
// y - input signal
// codebook_size - count of atoms
//output parameter
// A - dictionary
// x - coefficent
//reference:K-SVD:An Algorithm for Designing of Overcomplete Dictionaries
// for Sparse Representation,Aharon M.,Elad M.etc
//==============================
if(y.cols<codebook_size)
{
   cout<<"codebook_size is too large or training samples is too small"<<endl;
   return;
}
// initialization
if(A.empty())
{
   A = Mat(y.rows,codebook_size,CV_32F);
   std::vector <int> seeds;

   for (int j = 0; j < codebook_size; j++)
        seeds.push_back(j);

   cv::randShuffle(seeds);

   for (int j = 0; j < codebook_size; j++)
	    A.col(j)=y.col(seeds[j]);
}
//normalizing atoms
for(int i = 0; i < A.cols; i++)
{	
		A.col(i)=A.col(i)/cv::norm(A.col(i),NORM_L2);
}
//main iteration
for (int k=1;k<=ksvd_iter;k++)
{
  cout<<"iteration "<<k<<endl;
  // sparse coding
  Mat x = LibOMP(A,y,10,0.1);
  Mat R=y-A*x;
  // update dictionary
  for (int m=0;m<codebook_size;m++)
  {
	vector<int> indx;
	for(int j=0;j<x.cols;j++)
	   if(x.at<float>(m,j)!=0)
		  indx.push_back(j);
	Mat Ri;
	for(int j=0;j<indx.size();j++)
	{
        Ri.col(indx[j]) = R.col(indx[j]) + A.col(m)*x.at<float>(m,indx[j]);
	}
	cv::SVD homographySVD(Ri,cv::SVD::FULL_UV);

	A.col(m)=homographySVD.u.col(0);
    for(int j=0;j<indx.size();j++)
	{
		x.at<float>(m,indx[j])=homographySVD.w.at<float>(0,0)* homographySVD.vt.at<float>(0,indx[j]);
		R.col(indx[j]) = Ri.col(indx[j])-A.col(m)*x.at<float>(m,indx[j]);
	}
    
    
  }
}
}
Mat LibDictLearn(Mat data, int dict_size, double lambda_d, int no_iter)
{

	 int rows = data.rows;
     int cols = data.cols;

	 mwArray Input(1, 2, mxCELL_CLASS);
	 mwArray Data(rows,cols,mxSINGLE_CLASS);
	

	 Mat tmp=data.t();
	 tmp=tmp.reshape(0,cols);
	 Data.SetData((float *)tmp.data,rows*cols);

     const char* fieldnames[] = {
        "K","lambda","iter"
     };
	 mwArray K(dict_size);
	 mwArray lambda(lambda_d);
     mwArray iter(no_iter);

	 mwArray param(3,1,3,fieldnames);
	 param.Get("iter",1,1).Set(iter);
	 param.Get("lambda",1,1).Set(lambda);
	 param.Get("K",1,1).Set(K);

	 Input.Get(1,1).Set(Data);
     Input.Get(1,2).Set(param);

	 Mat dict_t=Mat(dict_size,rows,CV_32F); 

	 mwArray Dict;
	 mexTrainDL(1,Dict,Input);

	 Dict.Get(1,1).GetData((float *)dict_t.data,rows*dict_size);

	 return dict_t.t();
}
Mat DispDict(Mat dict, int rows, int cols, int x, int y, float sort)
{
	 mwArray numRows(rows),numCols(cols),X(x),Y(y),sortVarFlag(sort);
	 mwArray dictionary(256,rows*cols,mxSINGLE_CLASS);
	 	
	 mwArray display;
	 Mat dict_temp=dict.t();
	 dict_temp=dict_temp.reshape(0,rows*cols);
	 dictionary.SetData((float*)dict_temp.data,256*rows*cols);
	
	 Chapter_12_DispDict(1, display, dictionary, numRows, numCols, X, Y, sortVarFlag);

	 int size_r=rows*(x+1)+1;
	 int size_c=cols*(y+1)+1;
	 Mat dictionary_t=Mat(size_r,size_c,CV_32F); 
	 display.GetData((float *)dictionary_t.data, size_r*size_c);
		
	Mat  dd = dictionary_t.t();
	//imshow("dictionary",dd);
	//waitKey(-1);
	return dd;
}