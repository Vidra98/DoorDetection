#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>


std::vector<torch::jit::IValue> load_image(const std::string &imagePath, bool plot=false, 
                                           int img_res=512);
std::vector<torch::jit::IValue> im_to_torch(cv::Mat image);
torch::Tensor gets_pred(torch::Tensor score_maps, int out_res[], int model_res);


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: main <path-to-exported-script-module>\n";
    return -1;
  }

  std::string imagePath;
  imagePath = "/home/victor/Desktop/python/deep_learning/pytorch-pose-7ce6642f777e9da6249bd5b05330d57fa09ea37a/data/spsDoor/train2/images/door_6.jpg";

  std::vector<torch::jit::IValue>input;
  

  torch::Tensor out_tensor, kp;
  int out_res[2] = {1280, 720};
  int model_res = 256;


  auto start = std::chrono::high_resolution_clock::now();

  input=load_image(imagePath);
  auto im_loading_time = std::chrono::high_resolution_clock::now();

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto model_loading_time = std::chrono::high_resolution_clock::now();

  auto outputTensor = model.forward(input);
  auto model_execution_time = std::chrono::high_resolution_clock::now();

  auto TensorList = outputTensor.toTensorList();
  out_tensor = TensorList[0];
  kp = gets_pred(out_tensor, out_res, model_res);
  std::cout<<kp<<std::endl;
  //int print = kp[0][0].item<int>();


  auto kp_extraction_time = std::chrono::high_resolution_clock::now();

  //create Mat from output tensor
  cv::Mat im_keypoint0 = cv::Mat::eye(256, 256, CV_32F);
  cv::Mat im_keypoint1 = cv::Mat::eye(256, 256, CV_32F);
  cv::Mat im_keypoint2 = cv::Mat::eye(256, 256, CV_32F);
  cv::Mat im_keypoint3 = cv::Mat::eye(256, 256, CV_32F);

  std::memcpy((void*)im_keypoint0.data, out_tensor[0][0].data_ptr(), sizeof(float)*out_tensor[0][0].numel());
  std::memcpy((void*)im_keypoint1.data, out_tensor[0][1].data_ptr(), sizeof(float)*out_tensor[0][1].numel());
  std::memcpy((void*)im_keypoint2.data, out_tensor[0][2].data_ptr(), sizeof(float)*out_tensor[0][2].numel());
  std::memcpy((void*)im_keypoint3.data, out_tensor[0][3].data_ptr(), sizeof(float)*out_tensor[0][3].numel());
  auto stop = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  auto loading_image_time = std::chrono::duration_cast<std::chrono::milliseconds>(im_loading_time - start);
  auto loading_model_time = std::chrono::duration_cast<std::chrono::milliseconds>(model_loading_time- im_loading_time);
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(model_execution_time - model_loading_time);
  auto kp_time = std::chrono::duration_cast<std::chrono::milliseconds>(kp_extraction_time - model_execution_time);
  auto gen_plot_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - kp_extraction_time);

  std::cout<<"The total execution took :"<<total_time.count()<<" milliseconds with :"<<std::endl;
  std::cout<<"image loading time :"<<loading_image_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"model loading time :"<<loading_model_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"inference time :"<<inference_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"keypoint extraction time :"<<kp_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"model plot gen time :"<<gen_plot_time.count()<<" microseconds:"<<std::endl;

  cv::imshow("keypoint0", im_keypoint0);
  cv::imshow("keypoint1", im_keypoint1);
  cv::imshow("keypoint2", im_keypoint2);
  cv::imshow("keypoint3", im_keypoint3);

  cv::waitKey(0);

  std::cout << "ok\n";
}

std::vector<torch::jit::IValue> load_image(const std::string &imagePath, bool plot, int img_res){
  cv::Mat image, inp;
  image = cv::imread(imagePath);
  cv::imshow("image", image);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::resize(image, inp, cv::Size(img_res, img_res), 0, 0, cv::INTER_CUBIC);

  return im_to_torch(inp);
}

std::vector<torch::jit::IValue> im_to_torch(cv::Mat image){
  /*
  * Convert an RGB (H x W x C) input image to a IValue of shape (1 x C x H x W)
  */
  torch::DeviceType device_type = c10::DeviceType::CPU; //at::kCUDA;
  auto tensor_image = torch::from_blob(image.data, {image.rows, image.cols, image.channels() }, at::kByte);
  tensor_image = tensor_image.permute({2,0,1});
  tensor_image.unsqueeze_(0);
  tensor_image = tensor_image.toType(c10::kFloat).div(255);
  tensor_image.to(device_type);
  std::vector<torch::jit::IValue>input;

  input.push_back(tensor_image);
  return input;
}

torch::Tensor gets_pred(torch::Tensor score_maps, int out_res[], int model_res){
  /*
  * get the score maps for each keypoint, extract the location of the keypoint and rescale it to 
  * the desired resolution.
  */
   auto out_size = score_maps.sizes();
  //gets_pred(out_tensor);
  torch::Tensor kp;
  kp = torch::randn({4,2});

  for(int im_idx = 0; im_idx < out_size[0]; im_idx++){
    for(int kp_idx = 0; kp_idx < out_size[1]; kp_idx++){
      kp[kp_idx][0] = torch::argmax(score_maps[im_idx][kp_idx])%out_size[2];
      kp[kp_idx][1] = (torch::argmax(score_maps[im_idx][kp_idx])/out_size[2]).to(torch::kInt);
      
      kp[kp_idx][0] = kp[kp_idx][0]*out_res[0]/model_res;
      kp[kp_idx][1] = kp[kp_idx][1]*out_res[1]/model_res;

    }
  }
  return kp;
}