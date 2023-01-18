#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>


std::vector<torch::jit::IValue>   load_image_from_path(const std::string &imagePath, bool plot=false, 
                                           int img_res=512);
std::vector<torch::jit::IValue>   load_image_from_mat(const cv::Mat image, bool plot=false, 
                                           int img_res=512);
std::vector<torch::jit::IValue> im_to_torch(cv::Mat image);
torch::Tensor gets_pred(torch::Tensor score_maps, int out_res[], int model_res);


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: main <path-to-exported-script-module>\n";
    return -1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<torch::jit::IValue>input;
  
  torch::Tensor out_tensor, kp;
  int out_res[2];
  int model_res = 256;

  std::string imagePath;
  cv::VideoCapture cap("/home/victor/Desktop/python/Visp_3D_model/build/model/door_front/my_phone/video_2.mp4");

  // Check if camera opened successfully
  if(!cap.isOpened()){
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }


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
  auto loading_model_time = std::chrono::duration_cast<std::chrono::milliseconds>(model_loading_time - start);
  std::cout<<"model loading time :"<<loading_model_time.count()<<" milliseconds:"<<std::endl;

  cv::Mat frame;
  // Capture frame-by-frame
  cap >> frame;

  cv::Size s = frame.size();
  out_res[1] = s.height + 1;
  out_res[0] = s.width + 1;

  int frame_count = 0;
  int img_rate = 5;

  while(frame.empty()==false){
    if(frame_count%img_rate == false){
    
      auto loop_start = std::chrono::high_resolution_clock::now();

      // Display the resulting frame
      
      input = load_image_from_mat(frame);

      auto outputTensor = model.forward(input);
      auto model_execution_time = std::chrono::high_resolution_clock::now();

      auto TensorList = outputTensor.toTensorList();
      out_tensor = TensorList[0];
      kp = gets_pred(out_tensor, out_res, model_res);
      std::cout<<kp<<std::endl;

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

      auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(model_execution_time - loop_start);
      auto kp_time = std::chrono::duration_cast<std::chrono::milliseconds>(kp_extraction_time - model_execution_time);
      auto gen_plot_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - kp_extraction_time);

      std::cout<<"inference time :"<<inference_time.count()<<" milliseconds:"<<std::endl;
      std::cout<<"keypoint extraction time :"<<kp_time.count()<<" milliseconds:"<<std::endl;
      std::cout<<"model plot gen time :"<<gen_plot_time.count()<<" microseconds:"<<std::endl;

      // cv::imshow("keypoint0", im_keypoint0);
      // cv::imshow("keypoint1", im_keypoint1);
      // cv::imshow("keypoint2", im_keypoint2);
      // cv::imshow("keypoint3", im_keypoint3);

      for (int i = 0; i < 4; i++) {
            cv::Point ftpt(kp[i][0].item<int>(), kp[i][1].item<int>());
            cv::circle(frame, ftpt, 3, cv::Scalar(255, 0, 255), 1, cv::FILLED);
            cv::circle(frame, ftpt, 1, cv::Scalar(0, 0, 255), 1, cv::FILLED);
      }
      cv::imshow( "Frame", frame );
      //cv::waitKey(0);


      // Press  ESC on keyboard to exit
      char c=(char)cv::waitKey(0);
      if(c==27)
        break;
    }

    frame_count += 1;
    cap >> frame;
  }
  std::cout << "ok\n";

  imagePath = "/home/victor/Desktop/simulation/gazeboroboticsimulator/src/sps/worlds/gate/1.png.001.png";
  cv::Mat image = cv::imread(imagePath);
  s = image.size();
  out_res[1] = s.height + 1;
  out_res[0] = s.width + 1;
  std::cout<<"out_res: "<<out_res[0]<<", "<<out_res[1]<<std::endl;
  input= load_image_from_path(imagePath);

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
  auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(model_execution_time - model_loading_time);
  auto kp_time = std::chrono::duration_cast<std::chrono::milliseconds>(kp_extraction_time - model_execution_time);
  auto gen_plot_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - kp_extraction_time);

  std::cout<<"The total execution took :"<<total_time.count()<<" milliseconds with :"<<std::endl;
  std::cout<<"model loading time :"<<loading_model_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"inference time :"<<inference_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"keypoint extraction time :"<<kp_time.count()<<" milliseconds:"<<std::endl;
  std::cout<<"model plot gen time :"<<gen_plot_time.count()<<" microseconds:"<<std::endl;

  cv::imshow("keypoint0", im_keypoint0);
  cv::imshow("keypoint1", im_keypoint1);
  cv::imshow("keypoint2", im_keypoint2);
  cv::imshow("keypoint3", im_keypoint3);
  for (int i = 0; i < 4; i++) {
    cv::Point ftpt(kp[i][0].item<int>(), kp[i][1].item<int>());
    cv::circle(image, ftpt, 3, cv::Scalar(255, 0, 255), 1, cv::FILLED);
    cv::circle(image, ftpt, 1, cv::Scalar(0, 0, 255), 1, cv::FILLED);
  }
  cv::imshow( "image", image );
  cv::waitKey(0);

  std::cout << "ok\n";
}

std::vector<torch::jit::IValue> load_image_from_mat(const cv::Mat image, bool plot, int img_res){
  cv::Mat inp;
  //image = cv::imread(imagePath);
  //cv::imshow("image", image);
  cv::cvtColor(image, inp, cv::COLOR_BGR2RGB);
  cv::resize(inp, inp, cv::Size(img_res, img_res), 0, 0, cv::INTER_CUBIC);

  return im_to_torch(inp);
}

std::vector<torch::jit::IValue> load_image_from_path(const std::string &imagePath, bool plot, int img_res){
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
      kp[kp_idx][0] = torch::argmax(score_maps[im_idx][kp_idx])%out_size[2]+1;
      kp[kp_idx][1] = (torch::argmax(score_maps[im_idx][kp_idx])/out_size[2]).to(torch::kInt)+1;
      
      kp[kp_idx][0] = kp[kp_idx][0]*out_res[0]/model_res;
      kp[kp_idx][1] = kp[kp_idx][1]*out_res[1]/model_res;

    }
  }
  return kp;
}