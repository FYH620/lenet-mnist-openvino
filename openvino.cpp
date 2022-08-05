#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;


int main()
{
	// 设置设备为CPU推理，在LoadNetwork时指定挂载的设备
	std::string device_name = "CPU";
	// 输入图像读取路径与输入IR模型文件读取路径
	std::string test_dir = "MNIST/";
	std::string input_image_path = test_dir + "img_test/" + "28-test-gray.jpg";


	// 初始化核心类Core类对象
	Core ie;
	// 核心对象利用ReadNetwork读取IR格式的模型
	// 参数这里只需要传入xml文件即可，对应的bin文件会在同级目录下自己找与xml同名bin文件
	// 这里第二个参数也可以自己指定bin文件路径，这里我已经把xml与bin放在同一目录了，写了一个参数
	CNNNetwork network = ie.ReadNetwork(test_dir + "openvino_IR/" + "lenet.xml");
	// 设置推理时的batch-size
	network.setBatchSize(1);


	// network.getInputsInfo()返回一个map格式的数据，实际是openvino实现InputsDataMap类
	// 源码用了 using InputsDataMap = std::map<std::string, InputInfo::Ptr> 起别名
	// 这个map里面保存了很多输入（对于我这个手写数字识别任务只有一个图像输入，onnx设置的也只有一个输入）

	// 如果自己设置了很多输入，可以用for(auto& input:network.getInputsInfo())遍历获取first与
	// second成员来代表这些输入数据的名称与信息指针

	// 因此用begin()迭代器调出第一个数据输入，获取输入数据名字与输入数据信息InputInfo类对象指针
	InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;
	// 设置输入信息，预处理时加入自动使用双线性插值resize，后续发现读取的图像尺寸不对会自行resize
	input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
	// 设置输入信息，设置为NCHW读取，与onnx导出IR格式相同，与下面的infer时blob内存写入的方式相同
	input_info->setLayout(Layout::NCHW);
	// 设置输入数据为FP32格式（这里要看用的什么方式训练的，训练用了FP16/INT8量化这里也要对应选择格式）
	input_info->setPrecision(Precision::FP32);
	// 同理配置输出数据格式（与输入数据格式同理）
	DataPtr output_info = network.getOutputsInfo().begin()->second;
	std::string output_name = network.getOutputsInfo().begin()->first;
	output_info->setPrecision(Precision::FP32);


	// 利用核心类对象的LoadNetwork方法创建可执行网络
	ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);


	// 创建推理请求（引擎），可以根据需要在这创建多个随后处理不同数据，设置为不同的对象名即可
	// 如 InferRequest infer_request1 = executable_network.CreateInferRequest();
	//    InferRequest infer_request2 = executable_network.CreateInferRequest();
	//    ........ 后续推理代码 ..........
	InferRequest infer_request = executable_network.CreateInferRequest();


	// 读取uchar8（0-255）单通道数据并归一化
	// 这里归一化的原因是torch训练时使用了/255操作，而我们只在onnx转ir时使用了mean与std，因此这里要补	 
	// 一个除以255的操作，并转化为CV_32FC1格式的cv::Mat数据
	cv::Mat img = cv::imread(input_image_path, CV_8UC1);
	cv::Mat new_img;
	img.convertTo(new_img, CV_32FC1, 1.0 / 255);
	

	// 利用buffer函数获取输入数据的Blob类在内存中的地址准备写入
	InferenceEngine::Blob::Ptr input = infer_request.GetBlob(input_name);
	auto buffer = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
	// 这里写入的方法要与读取格式与训练格式对应，如按照NCHW写入，因为这个是灰度图因此通道数为1
	// 在buffer内存计算偏移（相当于“多维数组压平放入一维内存”）写入该点的像素值
	// 如果是三通道NCHW读取则还需要在最外层套一层channels数目的循环，这里由于指定为灰度图所以不再写
	size_t img_height = new_img.rows;
	size_t img_width = new_img.cols;
	for (int row = 0; row < img_height; ++row)
	{
		for (int col = 0; col < img_width; ++col)
			buffer[row * img_width + col] = new_img.at<float>(row, col);
	}

	// 特别的对于上述三通道图像的处理方式除了利用循环也可以像下方注释代码一样切分后写入，都可以
	// 把像素写入内存并作为Blob类对象格式数据送入引擎推理（引擎无法处理Mat类数据，需要是Blob类数据对象）
	/*std::vector<cv::Mat> planes(3);
	for (size_t pId = 0; pId < planes.size(); pId++) {
		planes[pId] = cv::Mat(cv::Size(img_width,img_height), CV_32FC1, buffer + pId * 			cv::Size(img_width, img_height).area());
	}
	cv::split(new_img, planes);*/
	

	// 推理引擎执行对已经利用指针写入内存的Blob类对象进行推理
	infer_request.Infer();


	// 获取推理结果的Blob类数据对象（利用output_name层名获取）
	Blob::Ptr output = infer_request.GetBlob(output_name);
	// 利用buffer函数获取输出Blob类数据对象的首地址
	// 用as函数强制转化为模板内PrecisionTrait<Precision::FP32>对应的value_type类型，本例即float
	// 获取输出结果为FP32格式的float指针，这里代表输出数据第一个数据在内存的地址
	float* logits = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
	
	
	// 因为这里是手写数字识别，共计10类，按照首地址依次读取后面内存中的10个概率值即可
	for (int i = 0; i < 10; ++i)
		std::cout << "数字为" << i << "的概率为:" << logits[i] << std::endl;
	system("pause");
	return 0;
}
