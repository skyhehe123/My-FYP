This project shows how to remove the rain streak from a single image. It is implemented in C++ with MFC GUI design. 

Prerequirement
	Matlab
	OpenCV 3.0
	64-bit windows 
Directories and files included
	/Images - The dataset  

	/Dictionaries - A folder contains .txt file that contains  	dictionaries, the format 

      Function.h - The header file contains all the functions used.

 	Function.cpp - The C++ file contains all the functions used.

      CvvImage.h, CvvImage.cpp, resource.h, stdafx.h, stdafx.cpp, targ	etver.h - files for MFC configuration, do not modify it.

	MFC_DEMO.h, MFC_DEMO.cpp, MFC_DEMODlg.h, MFC_DEMODlg.cpp - files 	for 	GUI design. 

To run the program 
	you can directly find the .exe from debug/release.
To build the program 
	you have to config as follows:
	1. Choose Configuration Manager and add x64 platform

	2. At Configuration Properties -> VC++ -> 
	a. Include Directories: 		
		C:\MATLAB_ROOT\extern\include
		C:\OPENCV_ROOT\build\install\include
		C:\OPENCV_ROOT\opencv\build\install\include\opencv
		C:\OPENCV_ROOT.0\opencv\build\install\include\opencv2

	b. Library Directories:
		C:\OPENCV_ROOT\x64\vc11\lib
		C:\MATLAB_ROOT\extern\lib\win64\microsoft

	3. At Configuration Properties -> Linker ->
	Additional dependencies:
		mclmcrrt.lib
		mclmcr.lib
		opencv_ts300d.lib (for debug mode)
            opencv_ts300.lib (for release mode)
		opencv_world300d.lib (for debug mode)
		opencv_world300.lib (for release mode)
      4. Add matlab run time libs to system environmental variables PATH:
     		PATH = C:\MATLAB_ROOT\runtime\win64
GUI Usage
	The GUI is easy to use, the processing steps has to follow the order:	Apply bilateral filter->Train/Load the dictionary->	Dictionary 	partition->Restore. You can select different source to be displayed 	from 	combobox bellow.
      The trained dictionary for an image will be saved to directory 	"/dictionaries", with filename format "IMAGE NAME_NUM OF ATOMS.txt.	
