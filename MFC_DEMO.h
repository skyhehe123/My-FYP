
// MFC_DEMO.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CMFC_DEMOApp:
// �йش����ʵ�֣������ MFC_DEMO.cpp
//

class CMFC_DEMOApp : public CWinApp
{
public:
	CMFC_DEMOApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CMFC_DEMOApp theApp;