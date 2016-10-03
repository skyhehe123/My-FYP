
// MFC_DEMODlg.cpp : 实现文件
//

#include "stdafx.h"
#include "MFC_DEMO.h"
#include "MFC_DEMODlg.h"
#include "afxdialogex.h"
#include "Functions.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();
	
// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFC_DEMODlg 对话框




CMFC_DEMODlg::CMFC_DEMODlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CMFC_DEMODlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFC_DEMODlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, m_cbExamble);
	DDX_Control(pDX, IDC_COMBO2, m_cbAtoms);
}

BEGIN_MESSAGE_MAP(CMFC_DEMODlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CMFC_DEMODlg::OnBnClickedOpenImg)
	ON_BN_CLICKED(IDC_BUTTON2, &CMFC_DEMODlg::OnBnClickedButton2)
	ON_CBN_SELCHANGE(IDC_COMBO1, &CMFC_DEMODlg::OnCbnSelchangeCombo1)
	ON_BN_CLICKED(IDC_BUTTON3, &CMFC_DEMODlg::OnBnClickedButton3)
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BUTTON4, &CMFC_DEMODlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CMFC_DEMODlg::OnBnClickedButton5)
END_MESSAGE_MAP()


// CMFC_DEMODlg 消息处理程序

BOOL CMFC_DEMODlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	mclInitializeApplication(NULL,0);
    OMPInitialize();
	TrainDLInitialize();
	DispDictInitialize();
	SetDlgItemText(IDC_TOTAL_CASES,"Ready");
	GetDlgItem(IDC_EDIT1)->SetWindowTextA("6");
	GetDlgItem(IDC_EDIT2)->SetWindowTextA("0.2");
	GetDlgItem(IDC_EDIT3)->SetWindowTextA("100");

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CMFC_DEMODlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFC_DEMODlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFC_DEMODlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CMFC_DEMODlg::DrawPicToHDC(Mat image, UINT ID)
{
	CDC *pDC = GetDlgItem(ID)->GetDC();
	HDC hDC= pDC->GetSafeHdc();
	CRect rect;
	GetDlgItem(ID)->GetClientRect(&rect);
	IplImage *img=& IplImage (image);
	CvvImage cimg;
	CvSize s=cvGetSize(img);
	cimg.CopyOf( img );				// 复制图片
	cimg.DrawToHDC( hDC, &rect );	// 将图片绘制到显示控件的指定区域内
	ReleaseDC( pDC );
	
}

void CMFC_DEMODlg::OnBnClickedOpenImg()
{
	// TODO: 在此添加控件通知处理程序代码

	
	CFileDialog dlg(
		TRUE, _T("*.bmp"), NULL,
		OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY,
		_T("image files (*.bmp; *.jpg) |*.bmp; *.jpg | All Files (*.*) |*.*||"), NULL
		);
	
	// 打开文件对话框的标题名
	dlg.m_ofn.lpstrTitle = _T("Open Image");
	// 判断是否获得图片
	if( dlg.DoModal() != IDOK )	
		return;
	// 获取图片路径	
	CString mPath = dlg.GetPathName();
	fTitle = dlg.GetFileTitle();
	
	
	img = imread(string(mPath), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH); 

	Size size(256,256);
    resize(img,img,size);
	img.convertTo(img,CV_32F,1/255.0);
	DrawPicToHDC(img, IDC_STATIC);
	
	
	
}

//Bilateral filter
void CMFC_DEMODlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码]
	CString value1,value2;
	GetDlgItemTextA(IDC_EDIT1,value1);
    GetDlgItemTextA(IDC_EDIT2,value2);
	if(img.empty())
	{
		AfxMessageBox( "Please open an image first" );
		return;
	}
	SetDlgItemText(IDC_TOTAL_CASES,"Busy");
    img_lf=bfltGray(img,5,atof(value1),atof(value2));
	img_hf=img-img_lf;
	if(m_cbExamble.FindString(-1,"High-freq")==CB_ERR)
	{
	m_cbExamble.AddString("High-Freq");
	m_cbExamble.AddString("Low-Freq");
	}
	SetDlgItemText(IDC_TOTAL_CASES,"Ready");
}


void CMFC_DEMODlg::OnCbnSelchangeCombo1()
{
	// TODO: 在此添加控件通知处理程序代码
	CString str;
    int idx = m_cbExamble.GetCurSel();
    if( idx < 0 ) return;

    m_cbExamble.GetLBText( idx, str );
    //CString Out;
    //Out.Format( "Drop List Selection => index %d\n%s", idx, str );
    //AfxMessageBox( Out );
	if(idx==0)
		DrawPicToHDC(img, IDC_STATIC);
	else if(idx==2)
		DrawPicToHDC(img_lf, IDC_STATIC);
	else if(idx==1)
	{
		Mat img_hf_show;
		img_hf.convertTo(img_hf_show,CV_32F,1/2.0,1/2.0);
		DrawPicToHDC(img_hf_show, IDC_STATIC);
	}
	else if(idx==3) //Dict
	{
		 int size=sqrt(dict.cols);
		 
		 Mat dict_show = DispDict(dict, size, size, 16, 16, 0.0);
		 DrawPicToHDC(dict_show, IDC_STATIC);
	}
	else if(idx==4) //Rain SubDict
	{
		 //show rain subdict
		Mat Dict_rain_show;
		int size=sqrt(dict.cols);
		copyMakeBorder(Dict_rain,Dict_rain_show,0,0,0,dict.cols-Dict_rain.cols,BORDER_CONSTANT,Scalar(0));
	    Dict_rain_show = DispDict(Dict_rain_show, size, size, 16, 16, 0.0);
		DrawPicToHDC(Dict_rain_show, IDC_STATIC);
	}
	else if(idx==5) //Geometry SubDict
	{
		 //show rain subdict
		Mat Dict_geometry_show;
		int size=sqrt(dict.cols);
	    copyMakeBorder(Dict_geometry,Dict_geometry_show,0,0,dict.cols-Dict_geometry.cols,0,BORDER_CONSTANT,Scalar(0));
	    Dict_geometry_show = DispDict(Dict_geometry_show, size, size, 16, 16, 0.0);
		DrawPicToHDC(Dict_geometry_show, IDC_STATIC);
	}
	else if(idx==6) //Geometry High-Freq
	{
		 //show rain subdict
		Mat img_hf_recover_show;
		img_hf_recover.convertTo(img_hf_recover_show,CV_32F,1/2.0,1/2.0);
		DrawPicToHDC(img_hf_recover_show, IDC_STATIC);
	}
	else if(idx==7) //Derain
	{
		 //show rain subdict
		DrawPicToHDC(img_recover, IDC_STATIC);
	}
	
}

//Dictionary traning
void CMFC_DEMODlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	
	if(img_hf.empty())
	{
		AfxMessageBox( "Please Pre-process the image first" );
		return;
	}

	CString s_atoms,s_iter;
    int idx = m_cbAtoms.GetCurSel();
    if( idx < 0 ) return;

    m_cbAtoms.GetLBText( idx, s_atoms );
	GetDlgItemTextA(IDC_EDIT3,s_iter);
	

	
	SetDlgItemText(IDC_TOTAL_CASES,"Busy");
	data = CreateTrainingSet(img_hf, 16);
	
	//Obtain dictionary
	string str="Dictionaries/"+fTitle+"_"+s_atoms+".txt";
	
	if(is_file_exists(str)){
		   AfxMessageBox( "Dictionary already exists" );
	       dict = readData(data.rows,atoi(s_atoms),str);
    }
    else{
		   dict= LibDictLearn(data,atoi(s_atoms),0.15,atoi(s_iter));
		   save2txt(dict,str);
	}

	if(m_cbExamble.FindString(-1,"Dictionary")==CB_ERR)
	m_cbExamble.AddString("Dictionary");

	SetDlgItemText(IDC_TOTAL_CASES,"Ready");
	
}


void CMFC_DEMODlg::OnDestroy()
{
	CDialogEx::OnDestroy();

	// TODO: 在此处添加消息处理程序代码
	TrainDLTerminate(); 
	OMPTerminate(); 
	DispDictTerminate(); 
    mclTerminateApplication();
	 
}

//Dictionary partition
void CMFC_DEMODlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	if(dict.empty())
	{
		AfxMessageBox( "Have a dictionary first" );
		return;
	}
	SetDlgItemText(IDC_TOTAL_CASES,"Busy");

	DictPartition(dict, Dict_rain, Dict_geometry);
	if(m_cbExamble.FindString(-1,"Rain SubDict")==CB_ERR)
	{
	m_cbExamble.AddString("Rain SubDict");
	m_cbExamble.AddString("Geom SubDict");
	}

	SetDlgItemText(IDC_TOTAL_CASES,"Ready");
}


void CMFC_DEMODlg::OnBnClickedButton5()
{
	// TODO: 在此添加控件通知处理程序代码
	if(Dict_geometry.empty())
	{
		AfxMessageBox( "Have a cartoon dictionary first" );
		return;
	}
	SetDlgItemText(IDC_TOTAL_CASES,"Busy");
	Mat SparseCoef=LibOMP(dict,data,10,0.1);
	Mat CoefMatrix_Geo=SparseCoef.rowRange(Dict_rain.cols,dict.cols); 
	Mat Dalpha=SpMultiply(dict.colRange(Dict_rain.cols,dict.cols), CoefMatrix_Geo);
	
	img_hf_recover=ImageRecover(16, img.rows, img.cols, Dalpha);
	img_recover=img_hf_recover+img_lf;

	if(m_cbExamble.FindString(-1,"Geometry High-Freq")==CB_ERR)
	{
	m_cbExamble.AddString("Geometry High-Freq");
	m_cbExamble.AddString("Derain");
	}
	SetDlgItemText(IDC_TOTAL_CASES,"Ready");
}
