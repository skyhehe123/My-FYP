
// MFC_DEMODlg.h : 头文件
//

#pragma once
#include "cv.h"
#include "highgui.h"
#include "CvvImage.h"
#include "c:\opencv\build\include\opencv2\core\core.hpp"
#include "Functions.h"
#include "afxwin.h"
#include "afxcmn.h"

// CMFC_DEMODlg 对话框
class CMFC_DEMODlg : public CDialogEx
{
// 构造
public:
	CMFC_DEMODlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_MFC_DEMO_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpenImg();

	void DrawPicToHDC(Mat img, UINT ID);
	afx_msg void OnBnClickedButton2();

	Mat img,data,img_hf,img_lf,dict,Dict_rain, Dict_geometry, img_hf_recover, img_recover;
	CString fTitle;
	afx_msg void OnCbnSelchangeCombo1();
	CComboBox m_cbExamble;
	afx_msg void OnBnClickedButton3();
	afx_msg void OnDestroy();
	afx_msg void OnBnClickedButton4();
	CComboBox m_cbAtoms;
	afx_msg void OnBnClickedButton5();
};
