// test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Windows.h"
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
HINSTANCE OcrDetectorDll;

int main()
{
	const char* CS_DLLName;
	std::string strConfigPath, strImgPath;
	std::vector<std::vector<std::vector<int>>>vecBoxResult;
	std::vector<std::string> vecResult;
	strConfigPath = "D:\\4_code\\Github\\PaddleOCR\\deploy\\cpp_infer\\tools\\config.txt";
	strImgPath = "D:\\4_code\\Github\\PaddleOCR\\doc\\imgs\\1.jpg";
	cv::Mat srcimg = cv::imread(strImgPath, cv::IMREAD_COLOR);
	CS_DLLName = "ocr_system.dll";
	OcrDetectorDll = ::LoadLibrary(CS_DLLName);
	//初始化
	typedef int(*OCRDetectorInit)(std::string strConfig);
	OCRDetectorInit OCRDetectorInit_;
//	OCRDetectorInit_ = (OCRDetectorInit)GetProcAddress(OcrDetectorDll, MAKEINTRESOURCE(3));
	OCRDetectorInit_ = (OCRDetectorInit)GetProcAddress(OcrDetectorDll, "OCRInit");
	int i = OCRDetectorInit_(strConfigPath);
	//执行
	typedef int(*OCRDetectorRun)(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::string> &labels);
	OCRDetectorRun OCRDetectorRun_;
	//OCRDetectorRun_ = (OCRDetectorRun)GetProcAddress(OcrDetectorDll, MAKEINTRESOURCE(2));
	OCRDetectorRun_ = (OCRDetectorRun)GetProcAddress(OcrDetectorDll, "OCRDetector");
	int k = 0;
	while (k<1)
	{
		auto start = std::chrono::system_clock::now();
		int j = OCRDetectorRun_(srcimg, vecBoxResult, vecResult);
		auto end = std::chrono::system_clock::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "predict result:" << vecResult.size()<<"\t";
		for (int i=0; i<vecResult.size(); i++)
		{
			std::cout << vecResult.at(i);
		}
		
		std::cout << "predict time:" << double(duration.count()) *std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;
		k++;
	}
	//释放
	typedef int(*OCRDetectorFree)();
	OCRDetectorFree OCRDetectorFree_;
	OCRDetectorFree_ = (OCRDetectorFree)GetProcAddress(OcrDetectorDll, "OCRFree");
	OCRDetectorFree_();
	std::cout << "释放成功" << std::endl;
	system("pause");
    return 0;
}

