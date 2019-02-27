#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int imageWidth = 640;								//摄像头的分辨率
const int imageHeight = 480;
const int boardWidth = 11;								//横向的角点数目
const int boardHeight = 8;								//纵向的角点数据
const int boardCorner = boardWidth * boardHeight;		//总的角点数据						
const int squareSize = 20;								//标定板黑白格子的大小 单位mm
const Size boardSize = Size(boardWidth, boardHeight);   //棋盘大小
Size imageSize = Size(imageWidth, imageHeight);         //图片大小

char* name_l = (char*)malloc(sizeof(char) * 200);
char* name_r = (char*)malloc(sizeof(char) * 200);
int frameNumber;                                      	//相机标定时需要采用的图像帧数

Mat R, T, E, F;                                         //R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵 
Mat cameraMatrix_l, cameraMatrix_r;					    //相机内参数
Mat distCoeff_l, distCoeff_r; 			            	//相机畸变参数
vector<Mat> rvecs_l, rvecs_r;						    //旋转向量
vector<Mat> tvecs_l, tvecs_r;							//平移向量

vector<vector<Point2f>> corners_l, corners_r;			//各个图像找到的角点的集合 和objRealPoint 一一对应
//vector<vector<Point2f>> imagePoint_l;                 //左边摄像机所有照片角点的坐标集合  
//vector<vector<Point2f>> imagePoint_r;                 //右边摄像机所有照片角点的坐标集合 
vector<vector<Point3f>> objRealPoint_r;                 //左边摄像机所有照片角点的坐标集合 
vector<vector<Point3f>> objRealPoint_l;                 //右边摄像机所有照片角点的坐标集合 
vector<vector<Point3f>> objRealPoint;					//各副图像的角点的实际物理坐标集合
vector<Point2f> corner_l, corner_r;						//某一副图像找到的角点

Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）   
Mat mapLx, mapLy, mapRx, mapRy;                         //映射表  
Rect validROIL, validROIR;                              //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Mat xyz;              //三维坐标

Mat rgbImage_l, grayImage_l, rgbImage_r, grayImage_r;
Mat rectifyImage_l, rectifyImage_r;                    //左右相机reamp后的图
Point origin;         //鼠标按下的起始点
Point originn;
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);

Mat disp8,_3dImage;
const double max_z = 1.0e4;