
// MFC_DEMODlg.h : ͷ�ļ�
//

#pragma once
#include "cv.h"
#include "highgui.h"
#include "CvvImage.h"
#include "c:\opencv\build\include\opencv2\core\core.hpp"
#include "Functions.h"
#include "afxwin.h"
#include "afxcmn.h"

// CMFC_DEMODlg �Ի���
class CMFC_DEMODlg : public CDialogEx
{
// ����
public:
	CMFC_DEMODlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_MFC_DEMO_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
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
