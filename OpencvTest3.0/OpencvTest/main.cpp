// opencv_test.cpp : 定义控制台应用程序的入口点。
//

#include <tchar.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <highgui.hpp>
#include "cv.h"
#include <cv.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "global.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/ml.hpp"

using namespace std;
using namespace cv;

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	//	Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//	imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

/*输出和保存数据*/
void outputCameraParam(void)
{
	/*保存数据*/
	cv::FileStorage fs;//初始化 
	fs.open("xml\\cameraMatrix_l.xml", FileStorage::WRITE);
	fs << "cameraMatrix_l" << cameraMatrix_l;
	fs.open("xml\\cameraDistoration_l.xml", FileStorage::WRITE);//初始化
	fs << "distCoeff_l" << distCoeff_l;
	fs.open("xml\\rotatoVector_l.xml", FileStorage::WRITE);//初始化
	fs << "rvecs_l" << rvecs_l;
	fs.open("xml\\translationVector_l.xml", FileStorage::WRITE);//初始化
	fs << "tvecs_l" << tvecs_l;

	fs.open("xml\\cameraMatrix_r.xml", FileStorage::WRITE);
	fs << "cameraMatrix_l" << cameraMatrix_r;
	fs.open("xml\\cameraDistoration_r.xml", FileStorage::WRITE);//初始化
	fs << "distCoeff_r" << distCoeff_r;
	fs.open("xml\\rotatoVector_r.xml", FileStorage::WRITE);//初始化
	fs << "rvecs_r" << rvecs_r;
	fs.open("xml\\translationVector_r.xml", FileStorage::WRITE);//初始化
	fs << "tvecs_r" << tvecs_r;
	fs.release();

	/*输出数据*/
	/*fx 0  cx
	  0  fy cy
	  0  0  0
	  */
	cout << "fx_l :" << cameraMatrix_l.at<double>(0, 0) << endl << "fy_l :" << cameraMatrix_l.at<double>(1, 1) << endl;
	cout << "cx_l :" << cameraMatrix_l.at<double>(0, 2) << endl << "cy_l :" << cameraMatrix_l.at<double>(1, 2) << endl;
	cout << "k1_l :" << distCoeff_l.at<double>(0, 0) << endl;
	cout << "k2_l :" << distCoeff_l.at<double>(0, 1) << endl;
	cout << "p1_l :" << distCoeff_l.at<double>(0, 2) << endl;
	cout << "p2_l :" << distCoeff_l.at<double>(0, 3) << endl;
	cout << "p3_l :" << distCoeff_l.at<double>(0, 4) << endl;

	cout << "fx_r :" << cameraMatrix_r.at<double>(0, 0) << endl << "fy_r :" << cameraMatrix_r.at<double>(1, 1) << endl;
	cout << "cx_r :" << cameraMatrix_r.at<double>(0, 2) << endl << "cy_r :" << cameraMatrix_r.at<double>(1, 2) << endl;
	cout << "k1_r :" << distCoeff_r.at<double>(0, 0) << endl;
	cout << "k2_r :" << distCoeff_r.at<double>(0, 1) << endl;
	cout << "p1_r :" << distCoeff_r.at<double>(0, 2) << endl;
	cout << "p2_r :" << distCoeff_r.at<double>(0, 3) << endl;
	cout << "p3_r :" << distCoeff_r.at<double>(0, 4) << endl;
}

//获取图片
int getPicture(void){
	int pic_num = 1;
	VideoCapture cap1;
	VideoCapture cap2;

	cap2.open(1);
	cap1.open(0);


	if (cap1.isOpened() && cap2.isOpened()) {
		double w = 640, h = 480;
		cap1.set(CAP_PROP_FRAME_WIDTH, w);//设置显示界面的宽高  
		cap1.set(CAP_PROP_FRAME_HEIGHT, h);
		cap2.set(CAP_PROP_FRAME_WIDTH, w);
		cap2.set(CAP_PROP_FRAME_HEIGHT, h);


		Mat frame1, frame2;
		namedWindow("Video1");
		namedWindow("Video2");
		while (1)
		{
			cap1 >> frame1;
			imshow("Video1", frame1);
			cap2 >> frame2;
			imshow("Video2", frame2);
			waitKey(30);

			char c = cvWaitKey(100);

			if (c == 's') {
				destroyWindow("Video1");
				destroyWindow("Video2");
				break;
			}
			if (c == ' ') {
				sprintf(name_l, "image\\leftPic%d.jpg", pic_num);
				sprintf(name_r, "image\\rightPic%d.jpg", pic_num);
				cout << "done" << endl;
				pic_num++;

				imwrite(name_l, frame1);
				imwrite(name_r, frame2);

			}
			if (c == 'n') {
				sprintf(name_l, "pic\\leftPic.jpg");
				sprintf(name_r, "pic\\rightPic.jpg");
				imwrite(name_l, frame1);
				imwrite(name_r, frame2);
				cout << "done" << endl;
			}

		}
	return pic_num;
	}
}

/*****单目标定*****/
void SingleCalibration_l() {
	Mat img_l;                      //定义图片
	int goodFrameCount_l= 0;
	namedWindow("ImageL");
	cout << "按Q退出 ..." << endl;
	while (goodFrameCount_l< frameNumber) //frameNumber表示图片数量
	{
		char *filename_l = (char *)malloc(sizeof(char) * 100);
		sprintf(filename_l, "image\\leftPic%d.jpg", goodFrameCount_l + 1);
	
		rgbImage_l = imread(filename_l, CV_LOAD_IMAGE_COLOR); //图片转为3信道
		cvtColor(rgbImage_l, grayImage_l, COLOR_BGR2GRAY);    //转为灰度图后赋给grayImage
		imshow("Camera_l", grayImage_l);

		bool isFind_l = findChessboardCorners(rgbImage_l, boardSize, corner_l, CV_CALIB_CB_NORMALIZE_IMAGE);
		if (isFind_l == true)	//所有角点都被找到 说明这幅图像是可行的
		{
			/*
			Size(5,5) 搜索窗口的一半大小
			Size(-1,-1) 死区的一半尺寸
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
			*/
			cornerSubPix(grayImage_l, corner_l, Size(3, 3), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			//将角点精确到亚像素级
			drawChessboardCorners(rgbImage_l, boardSize, corner_l, isFind_l);
			imshow("chessboard_l", rgbImage_l);
			corners_l.push_back(corner_l);//把所有角点信息保存起来
			goodFrameCount_l++;
			cout << "The image is good" << endl;
		}
		else
		{
			cout << "The image is bad please try again" << endl;
		}

		if (waitKey(10) == 'q')
		{
			break;
		}
	}
	/*
	图像采集完毕 接下来开始摄像头的校正
	calibrateCamera()
	输入参数 
	objectPoints  角点的实际物理坐标
	imagePoints   角点的图像坐标
	imageSize	   图像的大小
	输出参数
	cameraMatrix  相机的内参矩阵
	distCoeffs	   相机的畸变参数
	rvecs		   旋转矢量(外参数)
	tvecs		   平移矢量(外参数）
	*/

	/*设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置*/
	/*计算实际的校正点的三维坐标*/
	calRealPoint(objRealPoint_l, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;
	/*标定摄像头*/
	calibrateCamera(objRealPoint_l, corners_l, Size(imageWidth, imageHeight), cameraMatrix_l, distCoeff_l, rvecs_l, tvecs_l, 0);
	cout << "calibration successful" << endl;
	/*显示畸变校正效果*/
	Mat cImage_l;
	undistort(rgbImage_l, cImage_l, cameraMatrix_l, distCoeff_l);
	imshow("Corret Image_l", cImage_l);
}
void SingleCalibration_r() {
	Mat img_r;                      //定义图片
	int goodFrameCount_r = 0;
	namedWindow("ImageR");
	cout << "按Q退出 ..." << endl;
	while (goodFrameCount_r < frameNumber) //frameNumber表示图片数量
	{
		char *filename_r = (char *)malloc(sizeof(char) * 100);
		sprintf(filename_r, "image\\rightPic%d.jpg", goodFrameCount_r + 1);
		rgbImage_r = imread(filename_r, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImage_r, grayImage_r, COLOR_BGR2GRAY);
		imshow("Camera_r", grayImage_r);

		bool isFind_r = findChessboardCorners(rgbImage_r, boardSize, corner_r, 0);
		if (isFind_r == true)	//所有角点都被找到 说明这幅图像是可行的
		{
			cornerSubPix(grayImage_r, corner_r, Size(3, 3), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			drawChessboardCorners(rgbImage_r, boardSize, corner_r, isFind_r);
			imshow("chessboard_r", rgbImage_r);
			corners_r.push_back(corner_r);
			goodFrameCount_r++;
			cout << "The image is good" << endl;
		}
		else
		{
			cout << "The image is bad please try again" << endl;
		}

		if (waitKey(10) == 'q')
		{
			break;
		}
	}
	/*计算实际的校正点的三维坐标*/
	calRealPoint(objRealPoint_r, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;
	/*标定摄像头*/
	calibrateCamera(objRealPoint_r, corners_r, Size(imageWidth, imageHeight), cameraMatrix_r, distCoeff_r, rvecs_r, tvecs_r, 0);
	cout << "calibration successful" << endl;

	/*显示畸变校正效果*/
	Mat cImage_r;
	undistort(rgbImage_r, cImage_r, cameraMatrix_r, distCoeff_r);
	imshow("Corret Image_r", cImage_r);
}

/*****立体匹配*****/
void stereo_match(int, void*)
{
	bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
	bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
	bm->setSpeckleWindowSize(300);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);
	Mat disp;
	bm->compute(grayImage_l, grayImage_r, disp);//输入图像必须为灰度图
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
	reprojectImageTo3D(disp8, _3dImage, Q, true, -1);
}

/*****特征点检测*****/
/*void FeatureDe() {

	//改变字体颜色
	system("color 2F");
	//显示帮助文字
	ShowHelpText();

	//载入原图片并显示
	Mat scrImage_l = imread("pic\\leftPic.jpg",1);
	Mat scrImage_r = imread("pic\\rightPic.jpg",1);
	imshow("原始图L", scrImage_l);
	imshow("原始图R", scrImage_r);

		//定义需要用到的变量和类
		int minHessian = 400;//定于SURF中的hessian阈值特征点检测算子
	SurfFeatureDetector detector(minHessian);//定义一个SURF特征检测类对象
	std::vector<KeyPoint> keypoints_l, keypoints_r; //vector模板类是能够存放任意类型的动态数组，能够增加和压缩数据

		//调用detect函数检测出SURF特征关键点，保存在vector容器中
		detector.detect(scrImage_l, keypoints_l);
	detector.detect(scrImage_r, keypoints_r);

	//绘制特征关键点
	Mat img_keypoints_l; Mat img_keypoints_r;
	drawKeypoints(scrImage_l, keypoints_l, img_keypoints_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(scrImage_r, keypoints_r, img_keypoints_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//显示效果图
	imshow("特征点检测效果图", img_keypoints_l);
	imshow("特征点检测效果图", img_keypoints_l);

	waitKey(0);

}*/

/*void FeatureDet() {
	//载入原图片并显示
	Mat scrImage_l = imread("pic\\leftPic.jpg", IMREAD_GRAYSCALE);
	Mat scrImage_r = imread("pic\\rightPic.jpg", IMREAD_GRAYSCALE);
	imshow("原始图L", scrImage_l);
	imshow("原始图R", scrImage_r);

	Ptr<SURF> surf;      //创建方式和2中的不一样
	surf = SURF::create(800);

	BFMatcher matcher;
	Mat l, r;
	vector<KeyPoint>key1, key2;
	vector<DMatch> matches;

	surf->detectAndCompute(scrImage_l, Mat(), key1, l);
	surf->detectAndCompute(scrImage_r, Mat(), key2, r);

	matcher.match(l, r, matches);       //匹配

	sort(matches.begin(), matches.end());  //筛选匹配点
	vector< DMatch > good_matches;
	int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
	cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	Mat outimg;
	drawMatches(scrImage_l, key1, scrImage_r, key2, good_matches, outimg, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点

}*/
//保存数据
static void saveXYZ(const char* filename, int xxx, int yyy,const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = yyy; y < (yyy+50); y++)
	{
		for (int x = xxx; x < (xxx+50); x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			//if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}
/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		for (int xx = x - 5; xx < x + 6; xx++) {
			for (int yy = y - 150; yy < y; yy++)
			{
				originn = Point(xx, yy);
				cout << originn << "in world coordinate is: " << xyz.at<Vec3f>(originn) << endl;
			}
		}
		//saveXYZ("3d_data.txt",x,y,xyz);
		printf("文件写入完毕\n");
		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}

/*点击时显示坐标，鼠标移动时不显示*/
/*IplImage* src = 0;
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		CvPoint pt = cvPoint(x, y);
		Vec3f point = _3dImage.at<Vec3f>(y, x);
		char temp[16];
		sprintf(temp, "(%d,%d,%d)", point[0], point[1],point[2]);
		cvPutText(src, temp, pt, &font, cvScalar(255, 255, 255, 0));
		cvCircle(src, pt, 2, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("src", src);
	}
}*/
//int xx, yy;
/*void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		CvPoint pt = cvPoint(x, y);
		xx = pt.x;
		yy = pt.y;
		imshow("3dimg",_3dImage);
		for (int y = 0; y<_3dImage.rows; y++)
		{
			for (int x = 0; x<_3dImage.cols; x++)
			{
				Vec3f point = _3dImage.at<Vec3f>(y, x);// Vec3f 是 template 类定义  
				if (y == yy&&x == xx)
					cout << "坐标[" << point[0] << "," << point[1] << "," << point[2] << "]" << endl;
			}
		}
	}
}
*/
int _tmain(int argc, _TCHAR* argv[])
{   /*棋盘单目标定*/
	//frameNumber=getPicture();//得到图片
	frameNumber = 13;
	SingleCalibration_l();//单目标定左
	SingleCalibration_r();//单目标定右
	outputCameraParam();	/*保存并输出参数*/
	cout << "out successful" << endl;


	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;
	double  rms = stereoCalibrate(objRealPoint, corners_l, corners_r,
		cameraMatrix_l, distCoeff_l,
		cameraMatrix_r, distCoeff_r,
		Size(imageWidth, imageHeight), R, T, E, F,
		CALIB_USE_INTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	cout << "Stereo Calibration done with RMS error = " << rms << endl;

	/*
	立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
	Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差
	*/
	stereoRectify(cameraMatrix_l, distCoeff_l, cameraMatrix_r, distCoeff_r, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
	/*
	根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
	mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
	所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
	*/
	initUndistortRectifyMap(cameraMatrix_l, distCoeff_l, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrix_r, distCoeff_r, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	/*读取图片*/
	rgbImage_l = imread("pic\\leftPic.jpg", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImage_l, grayImage_l, CV_BGR2GRAY);
	rgbImage_r = imread("pic\\rightPic.jpg", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImage_r, grayImage_r, CV_BGR2GRAY);

	cvtColor(grayImage_l, rectifyImage_l, CV_GRAY2BGR);
	cvtColor(grayImage_r, rectifyImage_r, CV_GRAY2BGR);

	imshow("ImageL Before Rectify", rectifyImage_l);
	imshow("ImageR Before Rectify", rectifyImage_r);


	/*
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(rectifyImage_l, rectifyImage_l, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImage_r, rectifyImage_r, mapRx, mapRy, INTER_LINEAR);

	imshow("ImageL", rectifyImage_l);
	imshow("ImageR", rectifyImage_r);

	cvtColor(rectifyImage_l, grayImage_l, CV_BGR2GRAY);
	cvtColor(rectifyImage_r, grayImage_r, CV_BGR2GRAY);

	/*
	把校正结果显示出来
	把左右两幅图像显示到同一个画面上
	这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来
	*/
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);

	/*左图像画到画布上*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rectifyImage_l, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  

	cout << "Painted ImageL" << endl;

	/*右图像画到画布上*/
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rectifyImage_r, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*画上对应的线条*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	/*
	立体匹配*/
	
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
	// 创建SAD窗口 Trackbar
	createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
	// 创建视差唯一性百分比窗口 Trackbar
	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
	// 创建视差窗口 Trackbar
	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
	//鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
	setMouseCallback("disparity", onMouse, 0);
	stereo_match(0, 0);

	//保存三维数据
	FILE*fp=fopen("D://3d_data.txt", "wt");
	for (int y=0;y<_3dImage.rows;y++)
	{
		for (int x=0;x<_3dImage.cols; x++)
		{
			Vec3f point = _3dImage.at<Vec3f>(y, x);// Vec3f 是 template 类定义  
			if (fabs(point[2] - max_z)<FLT_EPSILON||fabs(point[2])>max_z)
				fprintf(fp,"%d %d %d\n", 0, 0, 0);
			else
				fprintf(fp,"%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);

	//CvFont font;
	//cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	//imshow("disparity", disp8);
	//cvSetMouseCallback("3dimg", on_mouse, 0);
	//cvWaitKey(0);
	

	cout << "wait key" << endl;
	waitKey(0);
	system("pause");
	return 0;
}
