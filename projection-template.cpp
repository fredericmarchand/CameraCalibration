#include <stdio.h>
#include <string>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "opencv2/calib3d/calib3d.hpp"


#include <math.h>

#define NUM_POINTS    10
#define RANGE         100.00

#define MAX_CAMERAS   100 
#define MAX_POINTS    3000

#define DEBUG_COMPUTE 0
#define DEBUG_DECOMPOSE 0

float projection[3][4] = {
0.902701, 0.051530, 0.427171, 10.0,
0.182987, 0.852568, -0.489535, 15.0,
-0.389418, 0.520070, 0.760184, 20.0,
};

float intrinsic[3][3] = {
-1000.000000, 0.000000, 0.000000, 
0.000000, -2000.000000, 0.000000, 
0.000000, 0.000000,     1.000000,
};

float all_object_points[NUM_POINTS][3] = {
0.1251, 56.3585, 19.3304, 
80.8741, 58.5009, 47.9873,
35.0291, 89.5962, 82.2840,
74.6605, 17.4108, 85.8943,
71.0501, 51.3535, 30.3995,
1.4985, 9.1403, 36.4452,
14.7313, 16.5899, 98.8525,
44.5692, 11.9083, 0.4669,
0.8911, 37.7880, 53.1663,
57.1184, 60.1764, 60.7166
};

// you write this routine
void decomposeprojectionmatrix(CvMat* projection_matrix, CvMat* rotation_matrix, CvMat* translation, CvMat* camera_matrix)
{
	CvMat* normProjMat = cvCreateMat(3, 4, CV_32F);
	
	float addPowers = pow(cvmGet(projection_matrix, 2, 0), 2) + pow(cvmGet(projection_matrix, 2, 1), 2) + pow(cvmGet(projection_matrix, 2, 2), 2);
	float scaleFactor = sqrt(addPowers);

#if DEBUG_DECOMPOSE == 1
	printf("%f\n", scaleFactor);
#endif

	for (int i = 0; i < 3; ++i) 
	{
		for (int j = 0; j < 4; ++j)
		{
			cvmSet(normProjMat, i, j, cvmGet(projection_matrix, i, j) / scaleFactor);
#if DEBUG_DECOMPOSE
			printf("%f ", cvmGet(normProjMat, i, j));
#endif
		}
#if DEBUG_DECOMPOSE == 1
		printf("\n");
#endif
	}
	
	float sigma;
	if (cvmGet(projection_matrix, 2, 3) < 0)
	{
		sigma = -1;
	}
	else
	{
		sigma = 1;
	}

	float Tz = sigma * cvmGet(normProjMat, 2, 3);

#if DEBUG_DECOMPOSE == 1
	printf("Tz: %f\n", Tz);
#endif

	for (int i = 0; i < 3; ++i) 
	{
		cvmSet(rotation_matrix, 2, i, sigma * cvmGet(normProjMat, 2, i));
	}	

	float Ox = ((cvmGet(projection_matrix, 0, 0) * cvmGet(projection_matrix, 2, 0)) + (cvmGet(projection_matrix, 0, 1) * cvmGet(projection_matrix, 2, 1)) + (cvmGet(projection_matrix, 0, 2) * cvmGet(projection_matrix, 2, 2)));
	float Oy = ((cvmGet(projection_matrix, 1, 0) * cvmGet(projection_matrix, 2, 0)) + (cvmGet(projection_matrix, 1, 1) * cvmGet(projection_matrix, 2, 1)) + (cvmGet(projection_matrix, 1, 2) * cvmGet(projection_matrix, 2, 2)));
	float fx = sqrt((pow(cvmGet(normProjMat, 0, 0), 2) + pow(cvmGet(normProjMat, 0, 1), 2) + pow(cvmGet(normProjMat, 0, 2), 2)) - pow(Ox, 2));
	float fy = sqrt((pow(cvmGet(normProjMat, 1, 0), 2) + pow(cvmGet(normProjMat, 1, 1), 2) + pow(cvmGet(normProjMat, 1, 2), 2)) - pow(Oy, 2));

#if DEBUG_DECOMPOSE == 1
	printf ("Ox = %f, Oy = %f, fx = %f, fy = %f\n", Ox, Oy, fx, fy);
#endif

	for (int i = 0; i < 3; ++i) 
	{
		cvmSet(rotation_matrix, 0, i, sigma * (Ox * cvmGet(normProjMat, 2, i) - cvmGet(normProjMat, 0, i))/fx);
		cvmSet(rotation_matrix, 1, i, sigma * (Oy * cvmGet(normProjMat, 2, i) - cvmGet(normProjMat, 1, i))/fy);
	}

	float Tx = (sigma * ((Ox * Tz) - cvmGet(normProjMat, 0, 3)) / fx);
	float Ty = (sigma * ((Oy * Tz) - cvmGet(normProjMat, 1, 3)) / fy);

	cvmSet(translation, 0, 0, Tx);
	cvmSet(translation, 1, 0, Ty);
	cvmSet(translation, 2, 0, Tz);

	cvmSet(camera_matrix, 0, 0, -fx);
	cvmSet(camera_matrix, 1, 1, -fy);
	cvmSet(camera_matrix, 0, 2, Ox);
	cvmSet(camera_matrix, 1, 2, Oy);
	cvmSet(camera_matrix, 2, 2, 1);
}

// you write this routine
void computeprojectionmatrix(CvMat* image_points, CvMat* object_points, CvMat* projection_matrix)
{
	CvMat* eVects = cvCreateMat(12, 12, CV_64F);
	CvMat* eVals =  cvCreateMat(12, 1, CV_64F);

	CvMat* A     = cvCreateMat(NUM_POINTS*2, 12, CV_32F);
	CvMat* AT    = cvCreateMat(12, NUM_POINTS*2, CV_32F);
	CvMat* ATA = cvCreateMat(12, 12, CV_32F);

	int pointIndex = 0;
	for (int i = 0; i < (NUM_POINTS*2); i+=2, pointIndex++) 
	{
		float Xi = cvmGet(object_points, pointIndex, 0);
		float Yi = cvmGet(object_points, pointIndex, 1);
		float Zi = cvmGet(object_points, pointIndex, 2);
		float xi = cvmGet(image_points, pointIndex, 0);
		float yi = cvmGet(image_points, pointIndex, 1);
		//even rows
		cvmSet(A, i, 0, Xi);
		cvmSet(A, i, 1, Yi);
		cvmSet(A, i, 2, Zi);
		cvmSet(A, i, 3, 1);
		cvmSet(A, i, 4, 0);
		cvmSet(A, i, 5, 0);
		cvmSet(A, i, 6, 0);
		cvmSet(A, i, 7, 0);
		cvmSet(A, i, 8,  -(xi * Xi));
		cvmSet(A, i, 9,  -(xi * Yi));
		cvmSet(A, i, 10, -(xi * Zi));
		cvmSet(A, i, 11, -xi);

		//odd rows
		cvmSet(A, i+1, 0, 0);
		cvmSet(A, i+1, 1, 0);
		cvmSet(A, i+1, 2, 0);
		cvmSet(A, i+1, 3, 0);
		cvmSet(A, i+1, 4, Xi);
		cvmSet(A, i+1, 5, Yi);
		cvmSet(A, i+1, 6, Zi);
		cvmSet(A, i+1, 7, 1);
		cvmSet(A, i+1, 8,  -(yi * Xi));
		cvmSet(A, i+1, 9,  -(yi * Yi));
		cvmSet(A, i+1, 10, -(yi * Zi));
		cvmSet(A, i+1, 11, -yi);
	}
#if DEBUG_COMPUTE == 1
	printf("A\n");
	for (int i = 0; i < (NUM_POINTS*2); i++) 
	{
		for (int j = 0; j < 12; ++j)
		{
			printf("%f ", cvmGet(A, i, j));
		}
		printf("\n");
	}
	printf("\n");
#endif
	cvTranspose(A, AT);

#if DEBUG_COMPUTE == 1
	printf("A Transpose\n");
	for (int i = 0; i < 12; i++) 
	{
		for (int j = 0; j < (NUM_POINTS*2); ++j)
		{
			printf("%f ", cvmGet(AT, i, j));
		}
		printf("\n");
	}
	printf("\n");
#endif

	cvMatMul(AT, A, ATA);	

#if DEBUG_COMPUTE == 1
	printf("ATA\n");
	for (int i = 0; i < 12; i++) 
	{
		for (int j = 0; j < 12; ++j)
		{
			printf("%f ", cvmGet(ATA, i, j));
		}
		printf("\n");
	}
	printf("\n");
#endif

	cvEigenVV(ATA, eVects, eVals);

#if DEBUG_COMPUTE == 1
	printf("Eigen Values\n");
	for (int i = 0; i < 12; i++) 
	{
		printf("%f ", cvmGet(eVals, i, 0));
		printf("\n");
	}
	printf("\n");

	printf("Eigen Vectors\n");
	for (int i = 0; i < 12; i++) 
	{
		for (int j = 0; j < 12; ++j)
		{
			printf("%f ", cvmGet(eVects, i, j));
		}
		printf("\n");
	}
	printf("\n");
#endif
	int index = 0;
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 4; j++) 
		{
			cvmSet(projection_matrix, i, j, cvmGet(eVects, 11, index++));
		}
	}

	printf("Projection Matrix\n");
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 4; j++) 
		{
			printf("%f ", cvmGet(projection_matrix, i, j));
		}
		printf("\n");
	}
	printf("\n");

}

int main() {
    CvMat* camera_matrix, *computed_camera_matrix; //
    CvMat* rotation_matrix, *computed_rotation_matrix; //
    CvMat* translation, *computed_translation;
	CvMat* image_points, *transp_image_points;
    CvMat* rot_vector;
    CvMat* object_points, *transp_object_points;
	CvMat* computed_projection_matrix;
	CvMat *final_projection;
	CvMat temp_projection, temp_intrinsic;
	FILE *fp;

	cvInitMatHeader(&temp_projection, 3, 4, CV_32FC1, projection);
	cvInitMatHeader(&temp_intrinsic, 3, 3, CV_32FC1, intrinsic);

	final_projection = cvCreateMat(3, 4, CV_32F);

	object_points = cvCreateMat(NUM_POINTS, 4, CV_32F);
	transp_object_points = cvCreateMat(4, NUM_POINTS, CV_32F);

	image_points = cvCreateMat(NUM_POINTS, 3, CV_32F);
	transp_image_points = cvCreateMat(3, NUM_POINTS, CV_32F);

	rot_vector = cvCreateMat(3, 1, CV_32F);
    camera_matrix = cvCreateMat(3, 3, CV_32F);
    rotation_matrix = cvCreateMat(3, 3, CV_32F);
    translation = cvCreateMat(3, 1, CV_32F);

    computed_camera_matrix = cvCreateMat(3, 3, CV_32F);
    computed_rotation_matrix = cvCreateMat(3, 3, CV_32F);
    computed_translation = cvCreateMat(3, 1, CV_32F);
	computed_projection_matrix = cvCreateMat(3, 4, CV_32F);

	fp = fopen("assign3-out","w");

	fprintf(fp, "Rotation matrix\n");
	for (int i=0; i<3; i++) {
   		for (int j=0; j<3; j++) {
    		cvmSet(camera_matrix,i,j, intrinsic[i][j]);
    		cvmSet(rotation_matrix,i,j, projection[i][j]);
    	}
		fprintf(fp, "%f %f %f\n", 
			cvmGet(rotation_matrix,i,0), cvmGet(rotation_matrix,i,1), cvmGet(rotation_matrix,i,2));
	}
    for (int i=0; i<3; i++)
    	cvmSet(translation, i, 0, projection[i][3]);

	fprintf(fp, "\nTranslation vector\n");
	fprintf(fp, "%f %f %f\n", 
		cvmGet(translation,0,0), cvmGet(translation,1,0), cvmGet(translation,2,0));

	fprintf(fp, "\nCamera Calibration\n");
	for (int i=0; i<3; i++) {
		fprintf(fp, "%f %f %f\n", 
			cvmGet(camera_matrix,i,0), cvmGet(camera_matrix,i,1), cvmGet(camera_matrix,i,2));
	}

	fprintf(fp,"\n");
	for (int i = 0; i < NUM_POINTS; i++) {
		cvmSet(object_points, i, 0, all_object_points[i][0]);
		cvmSet(object_points, i, 1, all_object_points[i][1]);
		cvmSet(object_points, i, 2, all_object_points[i][2]);
		cvmSet(object_points, i, 3, 1.0);
		fprintf(fp, "Object point %d x %f y %f z %f\n", 
			i, all_object_points[i][0], all_object_points[i][1], all_object_points[i][2]);
	}
	fprintf(fp, "\n");
	cvTranspose(object_points, transp_object_points);

	cvMatMul(&temp_intrinsic, &temp_projection, final_projection);
	cvMatMul(final_projection, transp_object_points, transp_image_points);
	//cvTranspose(transp_image_points, image_points);


	for (int i=0; i<NUM_POINTS; i++) {
		cvmSet(image_points, i, 0, cvmGet(transp_image_points, 0, i)/cvmGet(transp_image_points, 2, i));
		cvmSet(image_points, i, 1, cvmGet(transp_image_points, 1, i)/cvmGet(transp_image_points, 2, i));
		fprintf(fp, "Image point %d x %f y %f\n", 
			i, cvmGet(image_points, i, 0), cvmGet(image_points, i, 0));
	}

	computeprojectionmatrix(image_points, object_points, computed_projection_matrix);
//#if DEBUG_COMPUTE == 1
	printf("Ratios\n");
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 4; j++) 
		{
			printf("%f ", cvmGet(computed_projection_matrix, i, j) / cvmGet(final_projection, i, j));
		}
		printf("\n");
	}
//#endif
	decomposeprojectionmatrix(computed_projection_matrix, computed_rotation_matrix, computed_translation, computed_camera_matrix);

	fprintf(fp, "\nComputed Rotation matrix\n");
	for (int i=0; i<3; i++) {
		fprintf(fp, "%f %f %f\n", 
			cvmGet(computed_rotation_matrix,i,0), cvmGet(computed_rotation_matrix,i,1), cvmGet(computed_rotation_matrix,i,2));
	}

	fprintf(fp, "\nComputed Translation vector\n");
	fprintf(fp, "%f %f %f\n", 
		cvmGet(computed_translation,0,0), cvmGet(computed_translation,1,0), cvmGet(computed_translation,2,0));

	fprintf(fp, "\nComputed Camera Calibration\n");
	for (int i=0; i<3; i++) {
		fprintf(fp, "%f %f %f\n", 
			cvmGet(computed_camera_matrix,i,0), cvmGet(computed_camera_matrix,i,1), cvmGet(computed_camera_matrix,i,2));
	}

	fclose(fp);
	//getchar();
    return 0;
}
