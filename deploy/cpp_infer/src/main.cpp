// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/config.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>

using namespace std;
using namespace cv;
using namespace PaddleOCR;

class OCRDetectorClass
{
public:
	OCRDetectorClass();
	OCRDetectorClass(std::string strConfig);
	bool Init(std::string strConfig);
	bool Run(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::string> &labels);
	void Free();
private:
	DBDetector *det = nullptr;
	Classifier *cls = nullptr;
	CRNNRecognizer *rec = nullptr;

};
extern "C"  _declspec(dllexport) int OCRInit(std::string strConfig);
extern "C" _declspec(dllexport) int OCRDetector(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::string> &labels);
extern "C" _declspec(dllexport) void OCRFree();
OCRDetectorClass::OCRDetectorClass()
{

}
OCRDetectorClass::OCRDetectorClass(std::string strConfig)
{
	Config config(strConfig);
	config.PrintConfigInfo();
	det = new DBDetector(config.det_model_dir, config.use_gpu, config.gpu_id, config.gpu_mem,
		config.cpu_math_library_num_threads, config.use_mkldnn,
		config.use_zero_copy_run, config.max_side_len, config.det_db_thresh,
		config.det_db_box_thresh, config.det_db_unclip_ratio, config.visualize);
	//DBDetector det(
	//	config.det_model_dir, config.use_gpu, config.gpu_id, config.gpu_mem,
	//	config.cpu_math_library_num_threads, config.use_mkldnn,
	//	config.use_zero_copy_run, config.max_side_len, config.det_db_thresh,
	//	config.det_db_box_thresh, config.det_db_unclip_ratio, config.visualize);
	if (config.use_angle_cls == true) {
		cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
			config.gpu_mem, config.cpu_math_library_num_threads,
			config.use_mkldnn, config.use_zero_copy_run,
			config.cls_thresh);
	}
	rec = new CRNNRecognizer(config.rec_model_dir, config.use_gpu, config.gpu_id,
		config.gpu_mem, config.cpu_math_library_num_threads,
		config.use_mkldnn, config.use_zero_copy_run,
		config.char_list_file);
	/*CRNNRecognizer rec(config.rec_model_dir, config.use_gpu, config.gpu_id,
		config.gpu_mem, config.cpu_math_library_num_threads,
		config.use_mkldnn, config.use_zero_copy_run,
		config.char_list_file);*/
}
bool OCRDetectorClass::Init(std::string strConfig)
{
	try
	{
		//std::cout << "初始化开始 :" << std::endl;
		Config config(strConfig);
		config.PrintConfigInfo();
		//std::cout << "参数设置成功 :" << std::endl;
		det = new DBDetector(config.det_model_dir, config.use_gpu, config.gpu_id, config.gpu_mem,
			config.cpu_math_library_num_threads, config.use_mkldnn,
			config.use_zero_copy_run, config.max_side_len, config.det_db_thresh,
			config.det_db_box_thresh, config.det_db_unclip_ratio, config.visualize);
		//std::cout << "det生成 :" << std::endl;
		if (config.use_angle_cls == true) {
			cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
				config.gpu_mem, config.cpu_math_library_num_threads,
				config.use_mkldnn, config.use_zero_copy_run,
				config.cls_thresh);
			//std::cout << "cls生成 :" << std::endl;
		}
		rec = new CRNNRecognizer(config.rec_model_dir, config.use_gpu, config.gpu_id,
			config.gpu_mem, config.cpu_math_library_num_threads,
			config.use_mkldnn, config.use_zero_copy_run,
			config.char_list_file);
		//std::cout << "rec生成 :" << std::endl;
	}
	catch (const std::exception&)
	{
		return false;
	}
	return true;
}
bool OCRDetectorClass::Run(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::string> &labels)
{
	try
	{
		//auto start = std::chrono::system_clock::now();
		//std::cout << "开始 :" << std::endl;
		det->Run(imgMat, boxes);
		//std::cout << "执行1:" << std::endl;
		rec->Run(boxes, imgMat, cls);
		rec->GetLabel(labels);
		//std::cout << "执行2 :" << std::endl;
		//auto end = std::chrono::system_clock::now();
		/*auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "Run time:" << double(duration.count()) *std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;*/
	}
	catch (const std::exception&)
	{
		return false;
	}
	return true;
}
void OCRDetectorClass::Free()
{
	if (det)
	{
		delete det;
		det = nullptr;
	}
	if (cls)
	{
		delete cls;
		cls = nullptr;
	}
	if (rec)
	{
		delete rec;
		rec = nullptr;
	}
}
/*int main(int argc, char **argv) */
OCRDetectorClass* myOcrDetector = new OCRDetectorClass;
_declspec(dllexport) int OCRInit(std::string strConfig)
{
	//std::cout << "进入OCRInit :" << std::endl;
	myOcrDetector->Init(strConfig);
	return 0;
}
_declspec(dllexport) int OCRDetector(cv::Mat imgMat,  std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::string> &labels)
{
 /* if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }*/
	//std::cout << "进入OCRDetector :" << std::endl;
	myOcrDetector->Run(imgMat, boxes, labels);
    return 0;
}
_declspec(dllexport) void OCRFree()
{
	myOcrDetector->Free();
	delete myOcrDetector;
	myOcrDetector = nullptr;
}