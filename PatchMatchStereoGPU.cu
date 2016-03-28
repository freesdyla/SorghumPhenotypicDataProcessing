#include "PatchMatchStereoGPU.h"
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <sstream>
#include "lodepng.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

int devCount;
double* init_time;
double* main_time;
double* post_time; 
double average = 0.0;
double sd = 0.0;

#define POST_PROCESSING 1

// CPU timing
struct timeval timerStart;
void StartTimer();
double GetTimer();

__device__ const float BAD_COST = 1e6f;//0.9*10.+0.1*2.;
__device__ const float alpha_c = 0.1f;
__device__ const float alpha_g = 1.f - alpha_c;
__device__ const float weight_c = 1.f/10.f; 
__device__ const float MIN_NZ = 0.5f;


__device__ float theta_sigma_d = 0.f;
__device__ float theta_sigma_n = 0.f;

// GPU timing
void StartTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop);
float GetTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop);

void loadPNG(float* img_ptr, float* R, float* G, float* B, std::string file_name, int* cols, int* rows);
void savePNG(unsigned char* disp, std::string fileName, int cols, int rows);
void timingStat(double* time, int nt, double* average, double* sd);
int imgCharToFloat(unsigned char* imgCharPtr, float* imgFloatPtr, bool reverse, unsigned int imgSize, float scale);

// kernels
// evaluate window-based disimilarity unary cost
__device__ float evaluateCost(cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to, cudaTextureObject_t rG_to,
				cudaTextureObject_t rB_to, cudaTextureObject_t rGrad_to,
				float u, float v, int x, int y, float disp, int cols, int rows,
                                int winRadius, float nx, float ny, float nz, int base) // base 0 left, 1 right
{
        float cost = 0.0f;
        float weight;
        float af, bf, cf;

        float xf = (float)x;
        float yf = (float)y;

/*	nx = 0.f;
	ny = 0.f;
	nz = 1.f;
*/

        // af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz
        af = nx/nz*(-1.0f);
        bf = ny/nz*(-1.0f);
        cf = (nx*xf + ny*yf + nz*disp)/nz;



        if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0/* || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0*/ )
                return BAD_COST;

        float tmp_disp;
//        float weight_sum = 0.0f;


        float r, g, b, color_L2;
        if(base == 1)
        {
                for(int h=-winRadius; h<=winRadius; h++)
                {
                        for(int w=-winRadius; w<=winRadius; w++)
                        {
                                tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf);

                               // if( isinf(tmp_disp)!=0 || isnan(tmp_disp)!=0 )
                                 //       return BAD_COST;

                                tmp_disp = tmp_disp/(float)(cols);

                                float wn = (float)w/(float)(cols);
                                float hn = (float)h/(float)(rows);

                                r = fabsf(tex2D<float>(rR_to, u, v)-tex2D<float>(rR_to, u+wn, v+hn));
                                g = fabsf(tex2D<float>(rG_to, u, v)-tex2D<float>(rG_to, u+wn, v+hn));
                                b = fabsf(tex2D<float>(rB_to, u, v)-tex2D<float>(rB_to, u+wn, v+hn));

                                weight = expf(-(r+b+g)*weight_c);
			
/*                                r = fabsf( tex2D<float>(lR_to, u + tmp_disp, v) - tex2D<float>(lR_to, u + tmp_disp + wn, v + hn));
                                g = fabsf( tex2D<float>(lG_to, u + tmp_disp, v) - tex2D<float>(lG_to, u + tmp_disp + wn, v + hn));
                                b = fabsf( tex2D<float>(lB_to, u + tmp_disp, v) - tex2D<float>(lB_to, u + tmp_disp + wn, v + hn));

				weight *= expf(-(r+b+g)*0.1f);
*/
                               // weight = expf(-sqrtf(r*r+g*g+b*b)*0.1f);

                      //          weight_sum += weight;

                                r = fabsf( tex2D<float>(rR_to, u + wn, v + hn) - tex2D<float>(lR_to, u + tmp_disp + wn, v + hn));
                                g = fabsf( tex2D<float>(rG_to, u + wn, v + hn) - tex2D<float>(lG_to, u + tmp_disp + wn, v + hn));
                                b = fabsf( tex2D<float>(rB_to, u + wn, v + hn) - tex2D<float>(lB_to, u + tmp_disp + wn, v + hn));

                                color_L2 = (r+g+b);
                               // color_L2 = sqrtf(r*r+g*g+b*b);

                                cost += weight * (alpha_c*min(color_L2, 10.0f)
                                                + alpha_g*min(fabsf( tex2D<float>(rGrad_to, u + wn, v + hn)
                                                - tex2D<float>(lGrad_to, u + tmp_disp + wn, v + hn)), 2.0f));
                        }
                }
        }
        else
        {
                for(int h=-winRadius; h<=winRadius; h++)
                {
                        for(int w=-winRadius; w<=winRadius; w++)
                        {
                                tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf);

                            //    if( isinf(tmp_disp)!=0 || isnan(tmp_disp)!=0 )
                              //          return BAD_COST;

                                tmp_disp = tmp_disp/(float)(cols);

                                float wn = (float)w/(float)(cols);
                                float hn = (float)h/(float)(rows);

                                r = fabsf(tex2D<float>(lR_to, u, v)-tex2D<float>(lR_to, u + wn, v + hn));
                                g = fabsf(tex2D<float>(lG_to, u, v)-tex2D<float>(lG_to, u + wn, v + hn));
                                b = fabsf(tex2D<float>(lB_to, u, v)-tex2D<float>(lB_to, u + wn, v + hn));

                                weight = expf(-(r+b+g)*weight_c);
                               // weight = expf(-sqrtf(r*r+b*b+g*g)*0.1f);

  /*                              r = fabsf(tex2D<float>(rR_to, u - tmp_disp, v) - tex2D<float>(rR_to, u - tmp_disp + wn, v + hn));
                                g = fabsf(tex2D<float>(rG_to, u - tmp_disp, v) - tex2D<float>(rG_to, u - tmp_disp + wn, v + hn));
                                b = fabsf(tex2D<float>(rB_to, u - tmp_disp, v) - tex2D<float>(rB_to, u - tmp_disp + wn, v + hn));
                                weight *= expf(-(r+b+g)*0.1f);
    */                //            weight_sum += weight;

                                r = fabsf(tex2D<float>(lR_to, u + wn, v + hn) - tex2D<float>(rR_to, u - tmp_disp + wn, v + hn));
                                g = fabsf(tex2D<float>(lG_to, u + wn, v + hn) - tex2D<float>(rG_to, u - tmp_disp + wn, v + hn));
                                b = fabsf(tex2D<float>(lB_to, u + wn, v + hn) - tex2D<float>(rB_to, u - tmp_disp + wn, v + hn));

  	                        color_L2 = (r+g+b);
                                //color_L2 = sqrtf(r*r+g*g+b*b);

                                cost += weight * (alpha_c*min(color_L2, 10.0f)
                                                + alpha_g*min(fabsf(tex2D<float>(lGrad_to, u + wn, v + hn)
                                                - tex2D<float>(rGrad_to, u - tmp_disp + wn, v + hn)), 2.0f));
                        }
                }
        }
	
	return cost;//weight_sum;

}

__global__ void stereoMatching(float* dRDispPtr, float* dRPlanes, float* dLDispPtr, float* dLPlanes,
                                float* dLCost, float* dRCost, int cols, int rows, int winRadius,
                                curandState* states, int iteration, float maxDisp, 
				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,
				cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to,
				cudaTextureObject_t rGrad_to)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        // does not need to process borders
        if(x>=cols || y>=rows)
                return;

        float u = ((float)x+0.5f)/(float)(cols);
        float v = ((float)y+0.5f)/(float)(rows);

        int idx = y*cols + x;

        // if 1st iteration, force planes to be fronto-parallel
        if(iteration == 0)
        {
                dRDispPtr[idx] *= maxDisp;
                dLDispPtr[idx] *= maxDisp;
                dLCost[idx] = BAD_COST;
                dRCost[idx] = BAD_COST;
	}

        // evaluate disparity of current pixel (based on right)
        float min_cost;
        float cost;
        float tmp_disp;
        float s;
        int tmp_idx;
        int new_x;
        int best_i;
        int best_j;
        bool VIEW_PROPAGATION = true;
        bool PLANE_REFINE = true;
        //--------------------------------------------
        {
                min_cost = BAD_COST;
                best_i = 0;
                best_j = 0;
                // spatial propagation
                for(int i=-1; i<=1; i++)
                {
                        for(int j=-1; j<=1; j++)
                        {
                                if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows)
                                        continue;

                                tmp_idx = idx + i*cols + j;

                                tmp_disp = dRDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, 
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dRPlanes[tmp_idx], dRPlanes[tmp_idx+1], dRPlanes[tmp_idx+2], 1);

                                // base 0 left, 1 right

                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = i;
                                        best_j = j;
                                }
                        }
                }

/*                if(iteration>0)
                {
                        tmp_min_cost = BAD_COST;

                        for(int i=-winRadius; i<=winRadius; i++)
                        {
                                for(int j=-winRadius; j<=winRadius; j++)
                                {
                                        if( x+j>=0 && x+j<cols && y+i>=0 && y+i<rows)
                                        {
                                                tmp_idx = idx + i*cols + j;

                                                if(dRCost[tmp_idx] < tmp_min_cost)
                                                {
                                                        tmp_min_cost = dRCost[tmp_idx];
                                                        tmp_best_i = i;
                                                        tmp_best_j = j;
                                                }
                                        }
                                }

                        }

                        if(tmp_min_cost < BAD_COST)
                        {

                                tmp_idx = idx + tmp_best_i*cols + tmp_best_j;

                                tmp_disp = dRDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dRPlanes[tmp_idx], dRPlanes[tmp_idx+1], dRPlanes[tmp_idx+2], 1);

                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = tmp_best_i;
                                        best_j = tmp_best_j;
                                }
                        }
                }
*/
                // update best plane
                tmp_idx = idx + best_i*cols + best_j;
                dRDispPtr[idx] = dRDispPtr[tmp_idx];
                tmp_idx *= 3;
                dRPlanes[idx*3] = dRPlanes[tmp_idx];
                dRPlanes[idx*3 + 1] = dRPlanes[tmp_idx + 1];
                dRPlanes[idx*3 + 2] = dRPlanes[tmp_idx + 2];
                dRCost[idx] = min_cost;


                // view propagation
                if(VIEW_PROPAGATION)
                {
                        new_x = (int)lroundf(dRDispPtr[idx]) + x;

                        // check if in range
                        if(new_x>=0 && new_x<cols)
                        {
                                tmp_idx = idx + new_x - x;
                                tmp_disp = dLDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dLPlanes[tmp_idx], dLPlanes[tmp_idx+1], dLPlanes[tmp_idx+2], 1);


                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        dRCost[idx] = min_cost;
                                        dRDispPtr[idx] = tmp_disp;
                                        dRPlanes[3*idx] = dLPlanes[tmp_idx];
                                        dRPlanes[3*idx+1] = dLPlanes[tmp_idx+1];
                                        dRPlanes[3*idx+2] = dLPlanes[tmp_idx+2];
                                }
                        }
                }

                // right plane refinement
                if(PLANE_REFINE)
                {
                        s = 1.0f;

                        for(float delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                        {
                                float cur_disp = dRDispPtr[idx];

                                cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                                if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                                {
                                        s *= 0.5;
                                        continue;
                                }

                                float nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dRPlanes[idx*3];
                                float ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dRPlanes[idx*3+1];
                                float nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dRPlanes[idx*3+2];
                                
				//normalize
                                float norm = sqrtf(nx*nx+ny*ny+nz*nz);

				nx /= norm;
				ny /= norm;
				nz /= norm;

				if(isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 || fabsf(nz) < MIN_NZ/* || isnan(nx)!=0 || isnan(ny)!=0 || isnan(nz)!=0*/ )
				{
					s *= 0.5f;
					continue;
				}

				cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, cur_disp, cols, rows, winRadius,
						nx, ny, nz, 1);


				if(cost < min_cost)
				{
					min_cost = cost;
					dRCost[idx] = min_cost;
					dRDispPtr[idx] = cur_disp;
					dRPlanes[idx*3] = nx;
					dRPlanes[idx*3 + 1] = ny;
					dRPlanes[idx*3 + 2] = nz;

				}

                                s *= 0.5;
                        }
                }

        }

        //------------------------------------------------------------
        {
                min_cost = BAD_COST;
                best_i = 0;
                best_j = 0;

                // spatial  propagation
                for(int i=-1; i<=1; i++)
                {
                        for(int j=-1; j<=1; j++)
                        {
                                if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows)
                                        continue;

                                tmp_idx = idx + i*cols + j;

                                tmp_disp = dLDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dLPlanes[tmp_idx], dLPlanes[tmp_idx+1], dLPlanes[tmp_idx+2], 0);


                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = i;
                                        best_j = j;
                                }
                        }
                }

  /*              if(iteration>0)
                {
                        tmp_min_cost = BAD_COST;

                        for(int i=-winRadius; i<=winRadius; i++)
                        {
                                for(int j=-winRadius; j<=winRadius; j++)
                                {
                                        if( x+j>=0 && x+j<cols && y+i>=0 && y+i<rows)
                                        {

                                                tmp_idx = idx + i*cols + j;

                                                if(dLCost[tmp_idx] < tmp_min_cost)
                                                {
                                                        tmp_min_cost = dLCost[tmp_idx];
                                                        tmp_best_i = i;
                                                        tmp_best_j = j;
                                                }
                                        }
                                }

                        }


                        if(tmp_min_cost != BAD_COST)
                        {
                                tmp_idx = idx + tmp_best_i*cols + tmp_best_j;

                                tmp_disp = dLDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dLPlanes[tmp_idx], dLPlanes[tmp_idx+1], dLPlanes[tmp_idx+2], 0);


                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = tmp_best_i;
                                        best_j = tmp_best_j;
                                }
                        }
                }
*/
                // update best plane
                tmp_idx = idx + best_i*cols + best_j;
                dLDispPtr[idx] = dLDispPtr[tmp_idx];
                tmp_idx *= 3;
                dLPlanes[idx*3] = dLPlanes[tmp_idx];
                dLPlanes[idx*3 + 1] = dLPlanes[tmp_idx + 1];
                dLPlanes[idx*3 + 2] = dLPlanes[tmp_idx + 2];
                dLCost[idx] = min_cost;

                // view propagation
                if(VIEW_PROPAGATION)
                {
                        new_x = x - (int)lroundf(dLDispPtr[idx]);

                        // check if in range
                        if(new_x>=0 && new_x<cols)
                        {
                                tmp_idx = idx + new_x - x;
                                tmp_disp = dRDispPtr[tmp_idx];

                                tmp_idx *= 3;

                                cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                dRPlanes[tmp_idx], dRPlanes[tmp_idx+1], dRPlanes[tmp_idx+2], 0);


                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        dLCost[idx] = min_cost;
                                        dLDispPtr[idx] = tmp_disp;
                                        dLPlanes[3*idx] = dRPlanes[tmp_idx];
                                        dLPlanes[3*idx+1] = dRPlanes[tmp_idx+1];
                                        dLPlanes[3*idx+2] = dRPlanes[tmp_idx+2];
                                }

                        }
                }

                // left plane refinement
                // exponentially reduce disparity search range
                if(PLANE_REFINE)
                {
                        s = 1.0f;

                        for(float delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                        {
                                float cur_disp = dLDispPtr[idx];

                                cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                                if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                                {
                                        s *= 0.5;
                                        continue;
                                }

                                float nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dLPlanes[idx*3];
                                float ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dLPlanes[idx*3+1];
                                float nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dLPlanes[idx*3+2];

                                //normalize
                                float norm = sqrtf(nx*nx+ny*ny+nz*nz);

				nx /= norm;
				ny /= norm;
				nz /= norm;

				if( isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 || fabsf(nz) < MIN_NZ/* || isnan(nx)!=0 || isnan(ny)!=0 || isnan(nz)!=0*/ )
				{
					s *= 0.5f;
					continue;
				}

				cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
							u, v, x, y, cur_disp, cols, rows, winRadius,
						nx, ny, nz, 0);


				if(cost < min_cost)
				{
					min_cost = cost;
					dLCost[idx] = min_cost;
					dLDispPtr[idx] = cur_disp;
					dLPlanes[idx*3] = nx;
					dLPlanes[idx*3 + 1] = ny;
					dLPlanes[idx*3 + 2] = nz;
				}

                                s *= 0.5;
                        }
                }
        }

        return;
}



__global__ void imgGradient_huber( int cols, int rows, cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to, 
				   float* lGradX, float* rGradX, float* lGradY, float* rGradY,
				   float* lGradXY, float* rGradXY, float* lGradYX, float* rGradYX)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        const int idx = y*cols+x;
	const float du = 1.0f/((float)cols);
	const float dv = 1.0f/((float)rows);
        const float u = ((float)x+0.5f)*du;
        const float v = ((float)y+0.5f)*dv;

	// horizontal sobel
	lGradX[idx] = 2.0f*(tex2D<float>(lGray_to, u+du, v)-tex2D<float>(lGray_to, u-du, v))
		      + (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u-du, v+dv))
		      + (tex2D<float>(lGray_to, u+du, v-dv)-tex2D<float>(lGray_to, u-du, v-dv));

	rGradX[idx] = 2.0f*(tex2D<float>(rGray_to, u+du, v)-tex2D<float>(rGray_to, u-du, v))
		      + (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u-du, v+dv))
		      + (tex2D<float>(rGray_to, u+du, v-dv)-tex2D<float>(rGray_to, u-du, v-dv));

	// vertical sobel
	lGradY[idx] = 2.0f*(tex2D<float>(lGray_to, u, v+dv)-tex2D<float>(lGray_to, u, v-dv))
		      + (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u+du, v-dv))
		      + (tex2D<float>(lGray_to, u-du, v+dv)-tex2D<float>(lGray_to, u-du, v-dv));

	rGradY[idx] = 2.0f*(tex2D<float>(rGray_to, u, v+dv)-tex2D<float>(rGray_to, u, v-dv))
		      + (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u+du, v-dv))
		      + (tex2D<float>(rGray_to, u-du, v+dv)-tex2D<float>(rGray_to, u-du, v-dv));

	// central difference 45 deg
	lGradXY[idx] = (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u-du, v-dv))/sqrtf(8.0f);
	rGradXY[idx] = (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u-du, v-dv))/sqrtf(8.0f);

	// central difference 135 deg
	lGradYX[idx] = (tex2D<float>(lGray_to, u-du, v+dv)-tex2D<float>(lGray_to, u+du, v-dv))/sqrtf(8.0f);
	rGradYX[idx] = (tex2D<float>(rGray_to, u-du, v+dv)-tex2D<float>(rGray_to, u+du, v-dv))/sqrtf(8.0f);
	
}

__global__ void gradient(float* lGradPtr, float*rGradPtr, cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows)
                return;

        int idx = y*cols+x;
        float u = ((float)x+0.5f)/(float)(cols);
        float v = ((float)y+0.5f)/(float)(rows);

/*	float tmp;

	tmp =  3.0f*(tex2D<float>(lGray_to, u + 1.0f/(float)(cols), v - 1.0f/(float)rows ) - tex2D<float>(lGray_to, u - 1/(float)(cols), v - 1.0f/(float)rows)) ;
	tmp +=  10.0f*(tex2D<float>(lGray_to, u + 1.0f/(float)(cols), v ) - tex2D<float>(lGray_to, u - 1/(float)(cols), v) );
	tmp +=  3.0f*(tex2D<float>(lGray_to, u + 1.0f/(float)(cols), v + 1.0f/(float)rows ) - tex2D<float>(lGray_to, u - 1/(float)(cols), v + 1.0f/(float)rows)) ;

	lGradPtr[idx] = tmp;

	tmp =  3.0f*(tex2D<float>(rGray_to, u + 1.0f/(float)(cols), v - 1.0f/(float)rows ) - tex2D<float>(rGray_to, u - 1/(float)(cols), v - 1.0f/(float)rows)) ;
	tmp +=  10.0f*(tex2D<float>(rGray_to, u + 1.0f/(float)(cols), v ) - tex2D<float>(rGray_to, u - 1/(float)(cols), v) );
	tmp +=  3.0f*(tex2D<float>(rGray_to, u + 1.0f/(float)(cols), v + 1.0f/(float)rows ) - tex2D<float>(rGray_to, u - 1/(float)(cols), v + 1.0f/(float)rows)) ;

	rGradPtr[idx] = tmp;
*/
        lGradPtr[idx] = 0.5f*(tex2D<float>(lGray_to, u + 1/(float)(cols), v ) - tex2D<float>(lGray_to, u - 1/(float)(cols), v) );
        rGradPtr[idx] = 0.5f*(tex2D<float>(rGray_to, u + 1/(float)(cols), v ) - tex2D<float>(rGray_to, u - 1/(float)(cols), v) );

/*        float left_y = 0.5f*(tex2D<float>(lGray_to, u, v + 1/(float)(rows) ) - tex2D<float>(lGray_to, u, v - 1/(float)(rows)) );
        float right_y = 0.5f*(tex2D<float>(rGray_to, u, v + 1/(float)(rows) ) - tex2D<float>(rGray_to, u, v - 1/(float)(rows)) );

	lGradPtr[idx] = 0.5f*( fabsf(left_x) + fabsf(left_y) );
	rGradPtr[idx] = 0.5f*( fabsf(right_x) + fabsf(right_y) );
*/

        return;
}

// initialize random states
__global__ void init(unsigned int seed, curandState_t* states, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows)
                return;

        int idx = y*cols+x;
        curand_init(seed, idx, 0, &states[idx]);
}

// initialize random uniformally distributed plane normals
__global__ void init_plane_normals(float* dRPlanes, curandState_t* states, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows)
                return;

        int idx = y*cols+x;

        float x1, x2;
	
	while(true)
	{
		while(true)
		{
		        x1 = curand_uniform(&states[idx])*2.0f - 1.0f;
		        x2 = curand_uniform(&states[idx])*2.0f - 1.0f;

		        if( x1*x1 + x2*x2 < 1.0f )
		                break;
		}

		int i = idx*3;
		dRPlanes[i] = 2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2);
		dRPlanes[i+1] = 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2);
		dRPlanes[i+2] = (1.0f - 2.0f*(x1*x1 + x2*x2));
	  //      dRPlanes[i+2] = fabsf(1.0f - 2.0f*(x1*x1 + x2*x2));
	
		if(fabsf(dRPlanes[i+2]) > MIN_NZ)
			break; 
	}
}

__global__ void leftRightCheck(float* dRDispPtr, float* dLDispPtr, int* dLOccludeMask, int* dROccludeMask, int cols, int rows, float minDisp, float maxDisp, int iter)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows)
		return;	

	int idx = y*cols+x;

	const float thresh = minDisp;

	float tmp_disp = dLDispPtr[idx];

	int tmp_idx = x - (int)lroundf(tmp_disp);

	const float d_thresh = 1.0f;

	if( (tmp_disp<0) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || fabsf(tmp_disp - dRDispPtr[idx + tmp_idx - x]) > d_thresh
		|| tmp_disp < thresh )
	{
		if(iter == 1)		
		dLDispPtr[idx] = 0.0f;
		dLOccludeMask[idx] = 1;
	}
	else
		dLOccludeMask[idx] = 0;

	tmp_disp = dRDispPtr[idx];
	tmp_idx = x + (int)lroundf(tmp_disp);

	if( (tmp_disp<0) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || fabsf(tmp_disp - dLDispPtr[idx + tmp_idx - x]) > d_thresh
		|| tmp_disp < thresh )
	{
		if(iter == 1)
		dRDispPtr[idx] = 0.0f;
		dROccludeMask[idx] = 1;
	}
	else
		dROccludeMask[idx] = 0;
	
}

__global__ void fillInOccluded(float* dLDisp, float* dRDisp, float* dLPlanes, float* dRPlanes,
				float* dLCost, float* dRCost, int* dLOccludeMask, int* dROccludeMask, int cols, int rows, 
				int winRadius, float maxDisp, int iteration,
 				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,
				cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to,
				cudaTextureObject_t rGrad_to)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>cols-1 || y>rows-1 )
		return;	

	int idx = y*cols+x;

	if( dLOccludeMask[idx] == 0 )
		return;

	float xf = (float)x;
	float yf = (float)y;
	
	float nx, ny, nz, af, bf, cf, disp, tmp_disp;
	int i = 1;
	disp = 2*maxDisp;
	float u = (xf+0.5f)/(float)(cols);
	float v = (yf+0.5f)/(float)(rows);
	int tmp_idx;
	float min_cost = BAD_COST;
	float tmp_cost;

	float best_disp = 0.0f;

	if(iteration == 0)
	{
		// left disparity search horizontally
		best_disp = 0.0f;
		tmp_disp = 0.0f;
		for(int i=-cols/4; i<=cols/4; i++)
		{
			if(i+x<0 || i+x>=cols)
				continue;
			
			tmp_idx = idx + i;

			if(dLOccludeMask[tmp_idx] == 1)
				continue;

			tmp_disp = dLDisp[tmp_idx];

			if(tmp_disp == best_disp)
				continue;

			tmp_idx *= 3;

			tmp_cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
					u, v, x, y, tmp_disp, cols, rows, winRadius,
				dLPlanes[tmp_idx], dLPlanes[tmp_idx+1], dLPlanes[tmp_idx+2], 0);

	
			if(tmp_cost  < min_cost)
			{
				min_cost = tmp_cost;
				best_disp = tmp_disp;
			}
		}

		dLDisp[idx] = best_disp;
		dLCost[idx] = min_cost;

		// right disparity search horizontally
		min_cost = BAD_COST;
		best_disp = 0.0f;
		for(int i=-cols/4; i<=cols/4; i++)
		{
			if(i+x<0 || i+x>=cols)
				continue;
			
			tmp_idx = idx + i;

			if(dROccludeMask[tmp_idx] == 1)
				continue;

			tmp_disp = dRDisp[tmp_idx];

			if(tmp_disp == best_disp)
				continue;

			tmp_idx *= 3;

			
			tmp_cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to,
						u, v, x, y, tmp_disp, cols, rows, winRadius,
						dRPlanes[tmp_idx], dRPlanes[tmp_idx+1], dRPlanes[tmp_idx+2], 1);
	
	
			if(tmp_cost  < min_cost)
			{
				min_cost = tmp_cost;
				best_disp = tmp_disp;
			}
		}

		dRDisp[idx] = best_disp;

	}

	if(iteration == 1 )
	{
		// search right
		i = 1;
		while( x+i < cols )
		{
			if( dLOccludeMask[idx+i] == 0 )
			{
				nx = dLPlanes[(idx+i)*3];
				ny = dLPlanes[(idx+i)*3+1];
				nz = dLPlanes[(idx+i)*3+2];

				// af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz			
				af = nx/nz*(-1.0f);
				bf = ny/nz*(-1.0f);
				cf = (nx*xf + ny*yf + nz*dLDisp[idx+i])/nz;
      
				if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0 )
				{
					i++;
					continue;
				}

				tmp_disp = af*xf + bf*yf + cf;
				
				if(tmp_disp>=0.0f && tmp_disp <= maxDisp && tmp_disp == tmp_disp)
					disp = tmp_disp;
				else
					disp = dLDisp[idx+i];
				break;			
			
			}

			i++;
		}

		//search left for the nearest valid(none zero) disparity
		i = 1;
		while( x-i>=0 )
		{
			// valid disparity
			if( dLOccludeMask[idx-i] == 0)
			{
			
				nx = dLPlanes[(idx-i)*3];
				ny = dLPlanes[(idx-i)*3+1];
				nz = dLPlanes[(idx-i)*3+2];
		
				af = nx/nz*(-1.0f);
				bf = ny/nz*(-1.0f);
				cf = (nx*xf + ny*yf + nz*dLDisp[idx-i])/nz;
        	
				if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0 )
				{
					i++;
					continue;
				}
		
				tmp_disp = af*xf + bf*yf + cf;
				
				if(tmp_disp>=0.0f && tmp_disp<=maxDisp && tmp_disp == tmp_disp && tmp_disp<disp)
					disp = tmp_disp;
				else if(dLDisp[idx-i]<disp)
					disp = dLDisp[idx-i];

				break;
			}
			
			i++;
		}

		dLDisp[idx] = disp;
	}

	
	return;	
}


__global__ void weightedMedianFilter(float* dLDisp, int* dLOccludeMask, int cols, int rows, int winRadius, float maxDisp,
	 				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
					cudaTextureObject_t lGray_to, cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,
					cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to,
					cudaTextureObject_t rGrad_to)

{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int idx = y*cols + x;

	if( x>=cols || y >=rows || dLOccludeMask[idx] == 0 )
		return;

	float u = ((float)x+0.5f)/(float)(cols);
	float v = ((float)y+0.5f)/(float)(rows);

	const int winRa = 5;
	const int winSize = winRa*2+1;

	float weightedMask[winSize*winSize];
	float dispMask[winSize*winSize];


	float color_diff, weight;
	float weight_sum = 0.0f;


	// apply weights
	int tmp_idx = 0;
	int invalid_count = 0;
	for(int i=-winRa; i<=winRa; i++)
	{
		for(int j=-winRa; j<=winRa; j++)
		{
	
			color_diff = sqrt(fabsf(tex2D<float>(lR_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lR_to, u, v))
				     +fabsf(tex2D<float>(lG_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lG_to, u, v))
			    		+fabsf(tex2D<float>(lB_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lB_to, u, v)));

			weight = expf(-(color_diff)*0.1f);

			if(x+j>=0 && x+j<cols && y+i>=0 && y+i<rows)
			{
				weightedMask[tmp_idx] = weight;
				dispMask[tmp_idx] = dLDisp[idx+i*cols+j];
				weight_sum += weight;
			}
			else
			{
				weightedMask[tmp_idx] = 0.0f;
				dispMask[tmp_idx] = 0.0f;
				invalid_count++;
			}
			tmp_idx++; 
		}
	}


	// insertion sort
	float tmp;
	for(int i=1; i<winSize*winSize; i++)
	{
		for(int j=i; j>0; j--)
		{
			if(dispMask[j] < dispMask [j-1])
			{
				tmp = weightedMask[j];
				weightedMask[j] = weightedMask[j-1];
				weightedMask[j-1] = tmp;

				tmp = dispMask[j];
				dispMask[j] = dispMask[j-1];
				dispMask[j-1]= tmp;
			}
		}
	}

	// 1/2 weight
	weight = 0.0f;
	for(int i=0; i<winSize*winSize; i++)
	{
		weight += weightedMask[i]/weight_sum;
	
		if(weight >= 0.5f)
		{		
			dLDisp[idx] = dispMask[i];	
			return;	
		}

	}
	
	return;
}


void PatchMatchStereoGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp)
{
	int cols = leftImg.cols;
	int rows = leftImg.rows;


	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(leftImg, cvLeftBGR_v);
	cv::split(rightImg, cvRightBGR_v);

	// BGR 2 grayscale
	cv::Mat cvLeftGray;
	cv::Mat cvRightGray;

	cv::cvtColor(leftImg, cvLeftGray, CV_BGR2GRAY);
	cv::cvtColor(rightImg, cvRightGray, CV_BGR2GRAY);	

	// convert to float
	cv::Mat cvLeftB_f;
	cv::Mat cvLeftG_f;
	cv::Mat cvLeftR_f;
	cv::Mat cvRightB_f;
	cv::Mat cvRightG_f;
	cv::Mat cvRightR_f;
	cv::Mat cvLeftGray_f;
	cv::Mat cvRightGray_f;	

	cvLeftBGR_v[0].convertTo(cvLeftB_f, CV_32F);
	cvLeftBGR_v[1].convertTo(cvLeftG_f, CV_32F);
	cvLeftBGR_v[2].convertTo(cvLeftR_f, CV_32F);	
	cvRightBGR_v[0].convertTo(cvRightB_f, CV_32F);	
	cvRightBGR_v[1].convertTo(cvRightG_f, CV_32F);	
	cvRightBGR_v[2].convertTo(cvRightR_f, CV_32F);	
	cvLeftGray.convertTo(cvLeftGray_f, CV_32F);
	cvRightGray.convertTo(cvRightGray_f, CV_32F);
		
	float* leftRImg_f = cvLeftR_f.ptr<float>(0);
	float* leftGImg_f = cvLeftG_f.ptr<float>(0);
	float* leftBImg_f = cvLeftB_f.ptr<float>(0);
	float* leftGrayImg_f = cvLeftGray_f.ptr<float>(0);
	float* rightRImg_f = cvRightR_f.ptr<float>(0);
	float* rightGImg_f = cvRightG_f.ptr<float>(0);
	float* rightBImg_f = cvRightB_f.ptr<float>(0);
	float* rightGrayImg_f = cvRightGray_f.ptr<float>(0);

	unsigned int imgSize = (unsigned int)cols*rows;


/*	leftRImg_f = new float[imgSize];
	leftGImg_f = new float[imgSize];
	leftBImg_f = new float[imgSize];
	rightRImg_f = new float[imgSize];
	rightGImg_f = new float[imgSize];
	rightBImg_f = new float[imgSize];
	leftGrayImg_f = new float[imgSize];
	rightGrayImg_f = new float[imgSize];

	unsigned char* lImgPtr_8u = new unsigned char[imgSize];
	unsigned char* rImgPtr_8u = new unsigned char[imgSize];*/

	// allocate floating disparity map, plane normals and gradient image (global memory)
	float* dRDisp = NULL;
	float* dLDisp = NULL;
	float* dRPlanes = NULL;
	float* dLPlanes = NULL;
	float* dLGrad = NULL;
	float* dRGrad = NULL;
	float* dLCost = NULL;
	float* dRCost = NULL;



	cudaMalloc(&dRDisp, imgSize*sizeof(float));
	cudaMalloc(&dLDisp, imgSize*sizeof(float));
	cudaMalloc(&dRPlanes, 3*imgSize*sizeof(float));
	cudaMalloc(&dLPlanes, 3*imgSize*sizeof(float));

	cudaMalloc(&dRCost, imgSize*sizeof(float));
	cudaMalloc(&dLCost, imgSize*sizeof(float));

	cudaMalloc(&dRGrad, imgSize*sizeof(float));
	cudaMalloc(&dLGrad, imgSize*sizeof(float));



	cudaArray* lR_ca;
	cudaArray* lG_ca;
	cudaArray* lB_ca;
	cudaArray* lGray_ca;
	cudaArray* rR_ca;
	cudaArray* rG_ca;
	cudaArray* rB_ca;
	cudaArray* rGray_ca;
	cudaArray* lGrad_ca;
	cudaArray* rGrad_ca;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&lR_ca, &desc, cols, rows);
	cudaMallocArray(&lG_ca, &desc, cols, rows);
	cudaMallocArray(&lB_ca, &desc, cols, rows);
	cudaMallocArray(&lGray_ca, &desc, cols, rows);
	cudaMallocArray(&lGrad_ca, &desc, cols, rows);
	cudaMallocArray(&rR_ca, &desc, cols, rows);
	cudaMallocArray(&rG_ca, &desc, cols, rows);
	cudaMallocArray(&rB_ca, &desc, cols, rows);
	cudaMallocArray(&rGray_ca, &desc, cols, rows);
	cudaMallocArray(&rGrad_ca, &desc, cols, rows);

	cudaMemcpyToArray(lR_ca, 0, 0, leftRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lG_ca, 0, 0, leftGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lB_ca, 0, 0, leftBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lGray_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rR_ca, 0, 0, rightRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rG_ca, 0, 0, rightGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rB_ca, 0, 0, rightBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGray_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	
	// texture object test
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = lR_ca;


	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeMirror;
	texDesc.addressMode[1] = cudaAddressModeMirror;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t lR_to = 0;
	cudaCreateTextureObject(&lR_to, &resDesc, &texDesc, NULL);


	cudaTextureObject_t lG_to = 0;
	resDesc.res.array.array = lG_ca;
	cudaCreateTextureObject(&lG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lB_to = 0;
	resDesc.res.array.array = lB_ca;
	cudaCreateTextureObject(&lB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lGray_to = 0;
	resDesc.res.array.array = lGray_ca;
	cudaCreateTextureObject(&lGray_to, &resDesc, &texDesc, NULL);


	cudaTextureObject_t rR_to = 0;
	resDesc.res.array.array = rR_ca;
	cudaCreateTextureObject(&rR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rG_to = 0;
	resDesc.res.array.array = rG_ca;
	cudaCreateTextureObject(&rG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rB_to = 0;
	resDesc.res.array.array = rB_ca;
	cudaCreateTextureObject(&rB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGray_to = 0;
	resDesc.res.array.array = rGray_ca;
	cudaCreateTextureObject(&rGray_to, &resDesc, &texDesc, NULL);

	// launch kernels
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x); 

	// calculate gradient
        gradient<<<gridSize, blockSize>>>(dLGrad, dRGrad, lGray_to, rGray_to, cols, rows);
        cudaDeviceSynchronize();

        // copy gradient back
        cudaMemcpy(rightGrayImg_f, dRGrad, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(leftGrayImg_f, dLGrad, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);


	cudaMemcpyToArray(lGrad_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGrad_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaTextureObject_t lGrad_to = 0;
	resDesc.res.array.array = lGrad_ca;
	cudaCreateTextureObject(&lGrad_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGrad_to = 0;
	resDesc.res.array.array = rGrad_ca;
	cudaCreateTextureObject(&rGrad_to, &resDesc, &texDesc, NULL);

	StartTimer();
                              
	// allocate memory for states
        curandState_t* states;
        cudaMalloc(&states, imgSize*sizeof(curandState_t));
        // initialize random states
        init<<<gridSize, blockSize>>>(1234, states, cols, rows);
        cudaDeviceSynchronize();
	
	
        curandGenerator_t gen;
        // host CURAND
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        // set seed
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen, dLDisp, imgSize);
        cudaDeviceSynchronize();

        curandGenerator_t gen1;
        // host CURAND
        curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
        // set seed
        curandSetPseudoRandomGeneratorSeed(gen1, 4321ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen1, dRDisp, imgSize);
        cudaDeviceSynchronize();

        // random initial right and left plane
        init_plane_normals<<<gridSize, blockSize>>>(dRPlanes, states, cols, rows);
        cudaDeviceSynchronize();

        init_plane_normals<<<gridSize, blockSize>>>(dLPlanes, states, cols, rows);
        cudaDeviceSynchronize();

	std::cout<<"Random Init:"<<GetTimer()<<std::endl;
	
	StartTimer();

	for(int i=0; i<iteration; i++)
	{
		stereoMatching<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dLDisp, dLPlanes,
							dLCost, dRCost, cols, rows, winRadius,
							states, i, (float)Dmax, lR_to, lG_to, lB_to,
							lGray_to, lGrad_to, rR_to, rG_to, rB_to,
							rGray_to, rGrad_to);
		cudaDeviceSynchronize();
	}

	std::cout<<"Main loop:"<<GetTimer()<<std::endl;
	


	StartTimer();

	int* dLOccludeMask;
	int* dROccludeMask;

	cudaMalloc(&dLOccludeMask, imgSize*sizeof(int));
	cudaMalloc(&dROccludeMask, imgSize*sizeof(int));

#if POST_PROCESSING

/*	leftRightCheck<<<gridSize, blockSize>>>(dRDisp, dLDisp, dLOccludeMask, dROccludeMask, cols, rows, (float)Dmin, (float)Dmax, 0);

	cudaDeviceSynchronize();

	fillInOccluded<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLCost, dRCost, dLOccludeMask, 
								dROccludeMask, cols, rows, winRadius, (float)Dmax, 0,
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);
	cudaDeviceSynchronize();	
*/
	/*leftRightCheck<<<gridSize, blockSize>>>(dRDisp, dLDisp, dLOccludeMask, dROccludeMask, cols, rows, (float)Dmin, (float)Dmax, 1);

	cudaDeviceSynchronize();*/

/*	fillInOccluded<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLCost, dRCost, dLOccludeMask, 
								dROccludeMask, cols, rows, winRadius, (float)Dmax, 1,
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);
	cudaDeviceSynchronize();	

       	weightedMedianFilter<<<gridSize, blockSize>>>(dLDisp, dLOccludeMask, cols, rows, winRadius, (float)Dmax, 
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);

	cudaDeviceSynchronize();
*/
	std::cout<<"Post Process:"<<GetTimer()<<std::endl;	
#endif

	// result
	cv::Mat cvLeftDisp_f, cvRightDisp_f;
	cvLeftDisp_f.create(rows, cols, CV_32F);
	cvRightDisp_f.create(rows, cols, CV_32F);

        // copy disparity map from global memory on device to host
//        cudaMemcpy(rightGrayImg_f, dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
//        cudaMemcpy(leftGrayImg_f, dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	leftDisp = cvLeftDisp_f.clone();
	rightDisp = cvRightDisp_f.clone();

	if(showLeftDisp)
	{
		cv::Mat tmpDisp, tmpDisp1;
		cvLeftDisp_f.convertTo(tmpDisp, CV_8U, scale);
		cv::imshow("Left Disp", tmpDisp);
		cv::waitKey(0);
		cvRightDisp_f.convertTo(tmpDisp1, CV_8U, scale);
		cv::imshow("Right Disp", tmpDisp1);
		cv::waitKey(0);
	}

        //float to char
      //  imgCharToFloat(lImgPtr_8u, leftGrayImg_f, true, imgSize, scale);
      //  imgCharToFloat(rImgPtr_8u, rightGrayImg_f, true, imgSize, scale);


        // Free device memory
        cudaFree(dRDisp);
        cudaFree(dRPlanes);
        cudaFree(dLDisp);
        cudaFree(dLPlanes);
        cudaFree(states);
        cudaFree(dRGrad);
        cudaFree(dLGrad);
	cudaFree(dLCost);
	cudaFree(dRCost);
	cudaFree(lR_ca);
	cudaFree(lG_ca);
	cudaFree(lB_ca);
	cudaFree(lGray_ca);
	cudaFree(lGrad_ca);
	cudaFree(rR_ca);
	cudaFree(rG_ca);
	cudaFree(rB_ca);
	cudaFree(rGray_ca);
	cudaFree(rGrad_ca);
	cudaFree(dLOccludeMask);
	cudaFree(dROccludeMask);

        curandDestroyGenerator(gen);
        curandDestroyGenerator(gen1);
	cudaDestroyTextureObject(lR_to);
	cudaDestroyTextureObject(lG_to);
	cudaDestroyTextureObject(lB_to);
	cudaDestroyTextureObject(lGray_to);
	cudaDestroyTextureObject(lGrad_to);
	cudaDestroyTextureObject(rR_to);
	cudaDestroyTextureObject(rG_to);
	cudaDestroyTextureObject(rB_to);
	cudaDestroyTextureObject(rGray_to);
	cudaDestroyTextureObject(rGrad_to);

        cudaDeviceReset();

	// save disparity image
	 
//	savePNG(lImgPtr_8u, "l.png", cols, rows);

//	savePNG(rImgPtr_8u, "r.png", cols, rows);
	

}

__device__ float deviceSmoothStep(float a, float b, float x)
{
    float t = fminf((x - a)/(b - a), 1.0f);
    t = fmaxf(t, 0.0f);
    return t*t*(3.0f - (2.0f*t));
}

// evaluate window-based disimilarity unary cost
__device__ float evaluateCost_huber(cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGradX_to, cudaTextureObject_t lGradY_to, 
				cudaTextureObject_t lGradXY_to, cudaTextureObject_t lGradYX_to, 
				cudaTextureObject_t rR_to, cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, 
				cudaTextureObject_t rGradX_to, cudaTextureObject_t rGradY_to, 
				cudaTextureObject_t rGradXY_to, cudaTextureObject_t rGradYX_to,
				float u, float v, int x, int y, float disp, int cols, int rows,
                                int winRadius, float nx, float ny, float nz, int base) // base 0 left, 1 right
{
        float cost = 0.0f;
        float weight;
        float af, bf, cf;

        const float xf = (float)x;
        const float yf = (float)y;
	const float du = 1.0f/(float)cols;
	const float dv = 1.0f/(float)rows;
		
	float weight_c = 1.f/10.f*255.f;
	const float alpha_c = 0.1f;
	const float alpha_g = 1.0f - alpha_c;
	const float gammaMin = 5.0f;
	const float gammaMax = 28.0f;
	const float gammaRadius = 39.0f;

/*	nx = 0.f;
	ny = 0.f;
	nz = 1.f;
*/

        // af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz
        af = nx/nz*(-1.0f);
        bf = ny/nz*(-1.0f);
        cf = (nx*xf + ny*yf + nz*disp)/nz;

        if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0/* || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0*/ )
                return BAD_COST;

        float tmp_disp;
        float weight_sum = 0.0f;

        float r, g, b, color_L1;

	cudaTextureObject_t* baseR;
	cudaTextureObject_t* baseG;
	cudaTextureObject_t* baseB;
	cudaTextureObject_t* matchR;
	cudaTextureObject_t* matchG;
	cudaTextureObject_t* matchB;
	cudaTextureObject_t* baseGradX;
	cudaTextureObject_t* baseGradY;
	cudaTextureObject_t* baseGradXY;
	cudaTextureObject_t* baseGradYX;
	cudaTextureObject_t* matchGradX;
	cudaTextureObject_t* matchGradY;
	cudaTextureObject_t* matchGradXY;
	cudaTextureObject_t* matchGradYX;
	float sign = 1.0f;

	if(base == 0) // left base
	{
		sign = -1.0f;
		baseR = &lR_to;
		baseG = &lG_to;
		baseB = &lB_to;
		matchR = &rR_to;
		matchG = &rG_to;
		matchB = &rB_to;
		baseGradX = &lGradX_to;
		baseGradY = &lGradY_to;
		baseGradXY = &lGradXY_to;
		baseGradYX = &lGradYX_to;
		matchGradX = &rGradX_to;
		matchGradY = &rGradY_to;
		matchGradXY = &rGradXY_to;
		matchGradYX = &rGradYX_to;
	}
	else	// right base
	{
		sign = 1.0f;
		baseR = &rR_to;
		baseG = &rG_to;
		baseB = &rB_to;
		matchR = &lR_to;
		matchG = &lG_to;
		matchB = &lB_to;
		baseGradX = &rGradX_to;
		baseGradY = &rGradY_to;
		baseGradXY = &rGradXY_to;
		baseGradYX = &rGradYX_to;
		matchGradX = &lGradX_to;
		matchGradY = &lGradY_to;
		matchGradXY = &lGradXY_to;
		matchGradYX = &lGradYX_to;
	}

	for(int h=-winRadius; h<=winRadius; h++)
        {
                for(int w=-winRadius; w<=winRadius; w++)
                {
                        tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf)*sign;

                       // if( isinf(tmp_disp)!=0 || isnan(tmp_disp)!=0 )
                         //       return BAD_COST;

                        tmp_disp = tmp_disp*du;

                        float wn = (float)w*du;
                        float hn = (float)h*dv;

                        r = fabsf(tex2D<float>(*baseR, u, v)-tex2D<float>(*baseR, u+wn, v+hn));
                        g = fabsf(tex2D<float>(*baseG, u, v)-tex2D<float>(*baseG, u+wn, v+hn));
                        b = fabsf(tex2D<float>(*baseB, u, v)-tex2D<float>(*baseB, u+wn, v+hn));

			weight_c = gammaMin+gammaRadius*deviceSmoothStep(0.0f, gammaMax, sqrtf((float)(w*w+h*h)));

                        weight = expf(-(r+b+g)*weight_c);
		
/*                               r = fabsf( tex2D<float>(lR_to, u + tmp_disp, v) - tex2D<float>(lR_to, u + tmp_disp + wn, v + hn));
                        g = fabsf( tex2D<float>(lG_to, u + tmp_disp, v) - tex2D<float>(lG_to, u + tmp_disp + wn, v + hn));
                        b = fabsf( tex2D<float>(lB_to, u + tmp_disp, v) - tex2D<float>(lB_to, u + tmp_disp + wn, v + hn));

							weight *= expf(-(r+b+g)*0.1f);
*/
                       // weight = expf(-sqrtf(r*r+g*g+b*b)*0.1f);

                        weight_sum += weight;

                        r = fabsf( tex2D<float>(*baseR, u + wn, v + hn) - tex2D<float>(*matchR, u + tmp_disp + wn, v + hn));
                        g = fabsf( tex2D<float>(*baseG, u + wn, v + hn) - tex2D<float>(*matchG, u + tmp_disp + wn, v + hn));
                        b = fabsf( tex2D<float>(*baseB, u + wn, v + hn) - tex2D<float>(*matchB, u + tmp_disp + wn, v + hn));

                        color_L1 = (r+g+b);

                        cost += weight * (alpha_c*min(color_L1, 0.04f)
                                        + alpha_g*min( fabsf( tex2D<float>(*baseGradX, u + wn, v + hn)
                                        		    - tex2D<float>(*matchGradX, u + tmp_disp + wn, v + hn))
						      +fabsf( tex2D<float>(*baseGradY, u + wn, v + hn)
                                        		    - tex2D<float>(*matchGradY, u + tmp_disp + wn, v + hn))	
						      +fabsf( tex2D<float>(*baseGradXY, u + wn, v + hn)
                                        		    - tex2D<float>(*matchGradXY, u + tmp_disp + wn, v + hn))	
						      +fabsf( tex2D<float>(*baseGradYX, u + wn, v + hn)
                                        		    - tex2D<float>(*matchGradYX, u + tmp_disp + wn, v + hn)), 0.04f));
                }
        }

	
#if 0
        if(base == 1) // right
        {
                for(int h=-winRadius; h<=winRadius; h++)
                {
                        for(int w=-winRadius; w<=winRadius; w++)
                        {
                                tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf);

                               // if( isinf(tmp_disp)!=0 || isnan(tmp_disp)!=0 )
                                 //       return BAD_COST;

                                tmp_disp = tmp_disp/(float)(cols);

                                float wn = (float)w/(float)(cols);
                                float hn = (float)h/(float)(rows);

                                r = fabsf(tex2D<float>(rR_to, u, v)-tex2D<float>(rR_to, u+wn, v+hn));
                                g = fabsf(tex2D<float>(rG_to, u, v)-tex2D<float>(rG_to, u+wn, v+hn));
                                b = fabsf(tex2D<float>(rB_to, u, v)-tex2D<float>(rB_to, u+wn, v+hn));

				weight_c = gammaMin+gammaRadius*deviceSmoothStep(0.0f, gammaMax, sqrtf((float)(w*w+h*h)));

                                weight = expf(-(r+b+g)*weight_c);
			
 /*                               r = fabsf( tex2D<float>(lR_to, u + tmp_disp, v) - tex2D<float>(lR_to, u + tmp_disp + wn, v + hn));
                                g = fabsf( tex2D<float>(lG_to, u + tmp_disp, v) - tex2D<float>(lG_to, u + tmp_disp + wn, v + hn));
                                b = fabsf( tex2D<float>(lB_to, u + tmp_disp, v) - tex2D<float>(lB_to, u + tmp_disp + wn, v + hn));

								weight *= expf(-(r+b+g)*0.1f);
*/
                               // weight = expf(-sqrtf(r*r+g*g+b*b)*0.1f);

                                weight_sum += weight;

                                r = fabsf( tex2D<float>(rR_to, u + wn, v + hn) - tex2D<float>(lR_to, u + tmp_disp + wn, v + hn));
                                g = fabsf( tex2D<float>(rG_to, u + wn, v + hn) - tex2D<float>(lG_to, u + tmp_disp + wn, v + hn));
                                b = fabsf( tex2D<float>(rB_to, u + wn, v + hn) - tex2D<float>(lB_to, u + tmp_disp + wn, v + hn));

                                color_L1 = (r+g+b);
                               // color_L2 = sqrtf(r*r+g*g+b*b);

                                cost += weight * (alpha_c*min(color_L1, 0.04f)
                                                + alpha_g*min( fabsf( tex2D<float>(rGradX_to, u + wn, v + hn)
                                                		    - tex2D<float>(lGradX_to, u + tmp_disp + wn, v + hn))
							      +fabsf( tex2D<float>(rGradY_to, u + wn, v + hn)
                                                		    - tex2D<float>(lGradY_to, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(rGradXY_to, u + wn, v + hn)
                                                		    - tex2D<float>(lGradXY_to, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(rGradYX_to, u + wn, v + hn)
                                                		    - tex2D<float>(lGradYX_to, u + tmp_disp + wn, v + hn)), 0.03f));
                        }
                }
        }
        else	//left
        {
                for(int h=-winRadius; h<=winRadius; h++)
                {
                        for(int w=-winRadius; w<=winRadius; w++)
                        {
                                tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf);

                            //    if( isinf(tmp_disp)!=0 || isnan(tmp_disp)!=0 )
                              //          return BAD_COST;

                                tmp_disp = tmp_disp/(float)(cols);

                                float wn = (float)w/(float)(cols);
                                float hn = (float)h/(float)(rows);

                                r = fabsf(tex2D<float>(lR_to, u, v)-tex2D<float>(lR_to, u + wn, v + hn));
                                g = fabsf(tex2D<float>(lG_to, u, v)-tex2D<float>(lG_to, u + wn, v + hn));
                                b = fabsf(tex2D<float>(lB_to, u, v)-tex2D<float>(lB_to, u + wn, v + hn));

				weight_c = gammaMin+gammaRadius*deviceSmoothStep(0.0f, gammaMax, sqrtf((float)(w*w+h*h)));
                                weight = expf(-(r+b+g)*weight_c);
                               // weight = expf(-sqrtf(r*r+b*b+g*g)*0.1f);

  /*                              r = fabsf(tex2D<float>(rR_to, u - tmp_disp, v) - tex2D<float>(rR_to, u - tmp_disp + wn, v + hn));
                                g = fabsf(tex2D<float>(rG_to, u - tmp_disp, v) - tex2D<float>(rG_to, u - tmp_disp + wn, v + hn));
                                b = fabsf(tex2D<float>(rB_to, u - tmp_disp, v) - tex2D<float>(rB_to, u - tmp_disp + wn, v + hn));
                                weight *= expf(-(r+b+g)*0.1f);
    */                          weight_sum += weight;

                                r = fabsf(tex2D<float>(lR_to, u + wn, v + hn) - tex2D<float>(rR_to, u - tmp_disp + wn, v + hn));
                                g = fabsf(tex2D<float>(lG_to, u + wn, v + hn) - tex2D<float>(rG_to, u - tmp_disp + wn, v + hn));
                                b = fabsf(tex2D<float>(lB_to, u + wn, v + hn) - tex2D<float>(rB_to, u - tmp_disp + wn, v + hn));

  	                        color_L1 = (r+g+b);
                                //color_L2 = sqrtf(r*r+g*g+b*b);

                                cost += weight * (alpha_c*min(color_L1, 0.04f)
                                                + alpha_g*min(fabsf(tex2D<float>(lGradX_to, u + wn, v + hn)
                                                                  - tex2D<float>(rGradX_to, u - tmp_disp + wn, v + hn))
							     +fabsf(tex2D<float>(lGradY_to, u + wn, v + hn)
                                                                  - tex2D<float>(rGradY_to, u - tmp_disp + wn, v + hn))
							     +fabsf(tex2D<float>(lGradXY_to, u + wn, v + hn)
                                                                  - tex2D<float>(rGradXY_to, u - tmp_disp + wn, v + hn))
							     +fabsf(tex2D<float>(lGradYX_to, u + wn, v + hn)
                                                                  - tex2D<float>(rGradYX_to, u - tmp_disp + wn, v + hn)), 0.03f));
                        }
                }
        }
#endif
	
	return cost/weight_sum;
}

__global__ void stereoMatching_huber( float* dRDispV, float* dLDispV, float* dRPlanesV, float* dLPlanesV,
				float* dRDisp, float* dRPlanes, float* dLDisp, float* dLPlanes,
                                float* dLCost, float* dRCost, int cols, int rows, int winRadius,
                                curandState* states, float maxDisp, 
				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t lGradX_to, cudaTextureObject_t lGradY_to, 
				cudaTextureObject_t lGradXY_to, cudaTextureObject_t lGradYX_to, 
				cudaTextureObject_t rR_to, cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, 
				cudaTextureObject_t rGray_to, cudaTextureObject_t rGradX_to, cudaTextureObject_t rGradY_to, 
				cudaTextureObject_t rGradXY_to, cudaTextureObject_t rGradYX_to, 
				float theta_sigma_d, float theta_sigma_n)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        // does not need to process borders
        if(x>=cols || y>=rows) return;

        const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);

        const int idx = y*cols + x;

        // evaluate disparity of current pixel (based on right)
        float min_cost;
        float cost;
        float tmp_disp;
        float s;
        int tmp_idx;
        int new_x;
        int best_i;
        int best_j;
        const bool VIEW_PROPAGATION = true;
        const bool PLANE_REFINE = true;
	const float lambda = 50.0f;
  	//------------------------------------------------------------ base left 0
        {
                min_cost = BAD_COST;
                best_i = 0;
                best_j = 0;

                // spatial  propagation
                for(int i=-1; i<=1; i++)
                {
                        for(int j=-1; j<=1; j++)
                        {
                                if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows)
                                        continue;

                                tmp_idx = idx + i*cols + j;

                                tmp_disp = dLDisp[tmp_idx]*maxDisp;
								
				float nz = sqrtf(1.0f-powf(dLPlanes[tmp_idx*2],2.0f)-powf(dLPlanes[tmp_idx*2+1], 2.0f));

                                cost =  lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
								 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
									u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                			dLPlanes[tmp_idx*2], dLPlanes[tmp_idx*2+1], nz, 0);

				cost += 0.5f*( theta_sigma_d*powf(dLDisp[tmp_idx]-dLDispV[tmp_idx], 2.0f) 
						+ theta_sigma_n*(powf(dLPlanes[tmp_idx*2]-dLPlanesV[tmp_idx*2], 2.0f)
						               + powf(dLPlanes[tmp_idx*2+1]-dLPlanesV[tmp_idx*2+1], 2.0f)) );

                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = i;
                                        best_j = j;
                                }
                        }
                }


		__syncthreads();
 
                // update best plane
                tmp_idx = idx + best_i*cols + best_j;
                dLDisp[idx] = dLDisp[tmp_idx];
                dLPlanes[idx*2] = dLPlanes[tmp_idx*2];
                dLPlanes[idx*2 + 1] = dLPlanes[tmp_idx*2 + 1];
                dLCost[idx] = min_cost;

                // view propagation
                if(VIEW_PROPAGATION)
                {
                        new_x = x - (int)lroundf(dLDisp[idx]);

                        // check if in range
                        if(new_x>=0 && new_x<cols)
                        {
                                tmp_idx = idx + new_x - x;
                                tmp_disp = dRDisp[tmp_idx]*maxDisp;
								
				float nz = sqrtf(1.0f-powf(dRPlanes[tmp_idx*2],2.0f)-powf(dRPlanes[tmp_idx*2+1],2.0f));

                                cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
								 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
								u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                		dRPlanes[tmp_idx*2], dRPlanes[tmp_idx*2+1], nz, 0);

				cost += 0.5f*( theta_sigma_d*powf(dLDisp[tmp_idx]-dLDispV[tmp_idx], 2.0f) 
						+ theta_sigma_n*(powf(dLPlanes[tmp_idx*2]-dLPlanesV[tmp_idx*2], 2.0f)
							       + powf(dLPlanes[tmp_idx*2+1]-dLPlanesV[tmp_idx*2+1], 2.0f)) );
												
                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        dLCost[idx] = min_cost;
                                        dLDisp[idx] = dRDisp[tmp_idx];
                                        dLPlanes[2*idx] = dRPlanes[tmp_idx*2];
                                        dLPlanes[2*idx+1] = dRPlanes[tmp_idx*2+1];
                                }
                        }
                }

		__syncthreads();

                // left plane refinement
                // exponentially reduce disparity search range
                if(PLANE_REFINE)
                {
                        s = 1.0f;

                        for(float delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                        {
                                float cur_disp = dLDisp[idx]*maxDisp;

                                cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                                if(cur_disp<0.0f || cur_disp>maxDisp)
                                {
                                        s *= 0.5f;
                                        continue;
                                }

                                float nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dLPlanes[idx*2];
                                float ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dLPlanes[idx*2+1];
				float nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);

                                //normalize
                                float norm = sqrtf(nx*nx+ny*ny+nz*nz);

				nx /= norm;
				ny /= norm;
				nz /= norm;
				nz = fabs(nz);

				if( isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 )
				{
					s *= 0.5f;
					continue;
				}

				
				cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
								 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
								 u, v, x, y, cur_disp, cols, rows, winRadius, nx, ny, nz, 0);

				cost += 0.5f*( theta_sigma_d*powf(dLDisp[tmp_idx]-dLDispV[tmp_idx],2.0f) 
					    + theta_sigma_n*(powf(dLPlanes[tmp_idx*2]-dLPlanesV[tmp_idx*2], 2.0f)
							   + powf(dLPlanes[tmp_idx*2+1]-dLPlanesV[tmp_idx*2+1], 2.0f)) );


				if(cost < min_cost)
				{
					min_cost = cost;
					dLCost[idx] = min_cost;
					dLDisp[idx] = cur_disp/maxDisp;
					dLPlanes[idx*2] = nx;
					dLPlanes[idx*2 + 1] = ny;
				}

				s *= 0.5f;
                        }
                }
        } 

	__syncthreads();      

	//--------------------------------------------  base right 1
        {
                min_cost = BAD_COST;
                best_i = 0;
                best_j = 0;
                // spatial  propagation
                for(int i=-1; i<=1; i++)
                {
                        for(int j=-1; j<=1; j++)
                        {
                                if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows)
                                        continue;

                                tmp_idx = idx + i*cols + j;

                                tmp_disp = dRDisp[tmp_idx]*maxDisp;
								
				float nz = sqrtf(1.0f-powf(dRPlanes[tmp_idx*2], 2.0f)-powf(dRPlanes[tmp_idx*2+1], 2.0f));

                                cost =  lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to, 
								rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
								u, v, x, y, tmp_disp, cols, rows, winRadius,
								dRPlanes[tmp_idx*2], dRPlanes[tmp_idx*2+1], nz, 1);
													
				cost += 0.5f*( theta_sigma_d*powf(dRDisp[tmp_idx]-dRDispV[tmp_idx], 2.0f) 
					     + theta_sigma_n*( powf(dRPlanes[tmp_idx*2]-dRPlanesV[tmp_idx*2], 2.0f)
					                     + powf(dRPlanes[tmp_idx*2+1]-dRPlanesV[tmp_idx*2+1], 2.0f) ) );

                                // base 0 left, 1 right
                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        best_i = i;
                                        best_j = j;
                                }
                        }
                }

		__syncthreads();

                // update best plane
                tmp_idx = idx + best_i*cols + best_j;
                dRDisp[idx] = dRDisp[tmp_idx];
                dRPlanes[idx*2] = dRPlanes[tmp_idx*2];
                dRPlanes[idx*2 + 1] = dRPlanes[tmp_idx*2 + 1];               
                dRCost[idx] = min_cost;


                // view propagation
                if(VIEW_PROPAGATION)
                {
                        new_x = (int)lroundf(dRDisp[idx]) + x;

                        // check if in range
                        if(new_x>=0 && new_x<cols)
                        {
                                tmp_idx = idx + new_x - x;
                                tmp_disp = dLDisp[tmp_idx]*maxDisp;
								
				float nz = sqrtf(1.0f-powf(dLPlanes[tmp_idx*2],2.0f)-powf(dLPlanes[tmp_idx*2+1],2.0f));

                                cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
								 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
								u, v, x, y, tmp_disp, cols, rows, winRadius,
                                                		dLPlanes[tmp_idx*2], dLPlanes[tmp_idx*2+1], nz, 1);
								
				cost += 0.5*( theta_sigma_d*powf(dRDisp[tmp_idx]-dRDispV[tmp_idx],2.0f) 
					    + theta_sigma_n*(powf(dRPlanes[tmp_idx*2]-dRPlanesV[tmp_idx*2], 2.0f)
						           + powf(dRPlanes[tmp_idx*2+1]-dRPlanesV[tmp_idx*2+1], 2.0f)) );

                                if(cost < min_cost)
                                {
                                        min_cost = cost;
                                        dRCost[idx] = min_cost;
                                        dRDisp[idx] = dLDisp[tmp_idx];
                                        dRPlanes[2*idx] = dLPlanes[tmp_idx*2];
                                        dRPlanes[2*idx+1] = dLPlanes[tmp_idx*2+1];                                      
                                }
                        }
                }

		__syncthreads();

                // right plane refinement
                if(PLANE_REFINE)
                {
                        s = 1.0f;

                        for(float delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                        {
                                float cur_disp = dRDisp[idx]*maxDisp;

                                cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                                if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                                {
                                        s *= 0.5f;
                                        continue;
                                }

                                float nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dRPlanes[idx*2];
                                float ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + dRPlanes[idx*2+1];    
				float nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);
                                
				//normalize
                                float norm = sqrtf(nx*nx+ny*ny+nz*nz);

				nx /= norm;
				ny /= norm;
				nz /= norm;
				nz = fabs(nz);

				if(isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0)
				{
					s *= 0.5f;
					continue;
				}

				cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
								 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
								u, v, x, y, cur_disp, cols, rows, winRadius, nx, ny, nz, 1);

				cost += 0.5f*( theta_sigma_d*powf(dRDisp[tmp_idx]-dRDispV[tmp_idx], 2.0f) 
							+ theta_sigma_n*(powf(dRPlanes[tmp_idx*2]-dRPlanesV[tmp_idx*2], 2.0f)
							               + powf(dRPlanes[tmp_idx*2+1]-dRPlanesV[tmp_idx*2+1], 2.0f)) );

				if(cost < min_cost)
				{
					min_cost = cost;
					dRCost[idx] = min_cost;
					dRDisp[idx] = cur_disp/maxDisp;
					dRPlanes[idx*2] = nx;
					dRPlanes[idx*2 + 1] = ny;
				}
				
				s *= 0.5f;
                        }
                }
        }   
}


__global__ void UpdateDualVariablesKernel(int cols, int rows, float* dDispV, float* dPlanesV, float* dWeight, float* dDispPd,
					  float* dPlanesPn, float* ddDisp_dx, float* ddDisp_dy, 
					  float* ddnx_dx, float* ddnx_dy, float* ddny_dx, float* ddny_dy,
					  float theta_sigma_d, float theta_sigma_n)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;
		
	const int idx = y*cols+x;

	// forward difference derivative
	ddDisp_dx[idx] = x==cols-1 ? 0.0f : dDispV[idx+1] - dDispV[idx]; 
	ddDisp_dy[idx] = y==rows-1 ? 0.0f : dDispV[idx+cols] - dDispV[idx];

	ddnx_dx[idx] =  x==cols-1 ? 0.0f : dPlanesV[2*(idx+1)] - dPlanesV[2*idx];
 	ddny_dx[idx] =  x==cols-1 ? 0.0f : dPlanesV[2*(idx+1)+1] - dPlanesV[2*idx+1]; 


	ddnx_dy[idx] = y==rows-1 ? 0.0f : dPlanesV[2*(idx+cols)] - dPlanesV[2*idx];
	ddny_dy[idx] = y==rows-1 ? 0.0f : dPlanesV[2*(idx+cols)+1] - dPlanesV[2*idx+1];

	// weight
	const float gp = dWeight[idx];
	const float gp_inv = 1.f/gp;
	
	// from Pock's paper ALG3 Huber-ROF model
	// L = sqrt(8)
	// gamma = lambda, delta = alpha huber param, mu = 2*sqrt(gamma*delta)/L   
	// tau = mu/(2*gamma) primal, sigma = mu/(2*delta) dual
	const float eps = 0.001f;	//huber param		
	const float beta_d = 2.0f*sqrtf(theta_sigma_d*eps)/sqrtf(8.0f)/(2.0f*eps);	// dual sigma	
	const float beta_n = 2.0f*sqrtf(theta_sigma_n*eps)/sqrtf(8.0f)/(2.0f*eps);

	// update dual disparity x direction
	float tmp[2];
	tmp[0] = (dDispPd[2*idx]+beta_d*gp*ddDisp_dx[idx])/(1.0f+beta_d*eps*gp_inv);

	// dual disparity y
	tmp[1] = (dDispPd[2*idx+1]+beta_d*gp*ddDisp_dy[idx])/(1.0f+beta_d*eps*gp_inv);

	float norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);
		
	// project back to unit ball x
	dDispPd[2*idx] = tmp[0]/fmaxf(1.0f, norm);
	
	// project back y
	dDispPd[2*idx+1] = tmp[1]/fmaxf(1.0f, norm);

	// update dual unit normal
	// gradient of normal
	// x direction of normal x element
	tmp[0]	= (dPlanesPn[4*idx]+beta_n*gp*ddnx_dx[idx])/(1.0f+beta_n*eps*gp_inv);
	// y direction of normal x element
	tmp[1] = (dPlanesPn[4*idx+1]+beta_n*gp*ddnx_dy[idx])/(1.0f+beta_n*eps*gp_inv);
	
	// norm
	norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);
	
	// 0, 1 index: x dir of normal x element, y dir of normal x element
	dPlanesPn[4*idx] = tmp[0]/fmaxf(1.0f, norm);
	dPlanesPn[4*idx+1] = tmp[1]/fmaxf(1.0f, norm);
	
	// x direction of normal y element
	tmp[0] = (dPlanesPn[4*idx+2]+beta_n*gp*ddny_dx[idx])/(1.0f+beta_n*eps*gp_inv);
	// y direction of normal y element
	tmp[1] = (dPlanesPn[4*idx+3]+beta_n*gp*ddny_dy[idx])/(1.0f+beta_n*eps*gp_inv);
	
	// norm
	norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);

	// project back
	// 2, 3 index: x dir of normal y element, y dir of normal y element
	dPlanesPn[4*idx+2] = tmp[0]/fmaxf(1.0f, norm);
	dPlanesPn[4*idx+3] = tmp[1]/fmaxf(1.0f, norm);
}


__global__ void UpdatePrimalVariablesKernel(int cols, int rows, float* dDispPd, float* dPlanesPn, float* dWeight, float* dDispV, float* dDisp,
					    float* dPlanesV, float* dPlanes, float* ddivPd, float* ddivPnx, float* ddivPny,
				            float theta_sigma_d, float theta_sigma_n)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;
		
	const int idx = y*cols+x;

	// disparity
	// divergence x direction, backward difference
	ddivPd[idx] = x==0 ? 0.0f : dDispPd[2*idx] - dDispPd[2*(idx-1)];
	// y direction
	ddivPd[idx] += y==0 ? 0.0f : dDispPd[2*idx+1] - dDispPd[2*(idx-cols)+1];

	// nx
	// divergence x direction, backward difference
	ddivPnx[idx] = x==0 ? 0.0f : dPlanesPn[4*idx] - dPlanesPn[4*(idx-1)];
	// y direction
	ddivPnx[idx] += y==0 ? 0.0f : dPlanesPn[4*idx+1] - dPlanesPn[4*(idx-cols)+1];

	// ny
	// divergence x direction, backward difference
	ddivPny[idx] = x==0 ? 0.0f : dPlanesPn[4*idx+2] - dPlanesPn[4*(idx-1)+2];
	// y direction
	ddivPny[idx] += y==0 ? 0.0f : dPlanesPn[4*idx+3] - dPlanesPn[4*(idx-cols)+3];

	// weight
	const float gp = dWeight[idx];

	// from Pock's paper ALG3 Huber-ROF model
	// L = sqrt(8)
	// gamma = lambda, delta = alpha huber param, mu = 2*sqrt(gamma*delta)/L   
	// tau = mu/(2*gamma) primal, sigma = mu/(2*delta) dual
	const float eps = 0.001f;	//huber param		
	const float nu_d = 2.0f*sqrtf(theta_sigma_d*eps)/sqrtf(8.0f)/(2.0f*theta_sigma_d);	// primal tau
	const float nu_n = 2.0f*sqrtf(theta_sigma_n*eps)/sqrtf(8.0f)/(2.0f*theta_sigma_n);

	dDispV[idx] = (dDispV[idx]+nu_d*(theta_sigma_d*dDisp[idx]+gp*ddivPd[idx]))/(1.0f+nu_d*theta_sigma_d);

	dPlanesV[2*idx] = (dPlanesV[2*idx]+nu_d*(theta_sigma_d*dPlanes[2*idx]+gp*ddivPnx[idx]))/(1.0f+nu_d*theta_sigma_d);

	dPlanesV[2*idx+1] = (dPlanesV[2*idx+1]+nu_d*(theta_sigma_d*dPlanes[2*idx+1]+gp*ddivPny[idx]))/(1.0f+nu_d*theta_sigma_d);	
}

void huberROFSmooth(float* dDispV, float* dPlanesV, float* dDisp, float* dPlanes,
		    float* dDispPd, float* dPlanesPn, float* dWeight, float* ddDisp_dx,
		    float* ddDisp_dy, float* ddnx_dx, float* ddnx_dy, float* ddny_dx,
		    float* ddny_dy, float* ddivPd, float* ddivPnx, float* ddivPny,	
                    int cols, int rows, float theta_sigma_d, float theta_sigma_n)
{
	// kernels size
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x); 

	// update dual variables
	UpdateDualVariablesKernel<<<gridSize, blockSize>>>(cols, rows, dDispV, dPlanesV, dWeight, dDispPd,
					 		   dPlanesPn, ddDisp_dx, ddDisp_dy, 
					    		   ddnx_dx, ddnx_dy, ddny_dx, ddny_dy,
							   theta_sigma_d, theta_sigma_n);
	cudaDeviceSynchronize();


	// update primal variables
	UpdatePrimalVariablesKernel<<<gridSize, blockSize>>>(cols, rows, dDispPd, dPlanesPn, dWeight, dDispV, dDisp,
					    		     dPlanesV, dPlanes, ddivPd, ddivPnx, ddivPny,
							     theta_sigma_d, theta_sigma_n);
	cudaDeviceSynchronize();


/*	cv::Mat cvL, cvR;
	cvL.create(rows, cols, CV_32F);
	cvR.create(rows, cols, CV_32F);
	cudaMemcpy(cvR.ptr<float>(0), ddivPd, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
	cudaMemcpy(cvL.ptr<float>(0), ddivPnx, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);

	std::cout<<"	derivative"<<std::endl;

	cv::imshow("dx", cvR);
	cv::imshow("dy", cvL);
	cv::waitKey(0);*/
}



// initialize random uniformally distributed plane normals
__global__ void init_plane_normals_huber(float* dPlanes, curandState_t* states, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows)
                return;

        int idx = y*cols+x;

        float x1, x2;
	
        while(true)
        {
                x1 = curand_uniform(&states[idx])*2.0f - 1.0f;
                x2 = curand_uniform(&states[idx])*2.0f - 1.0f;

		// 0.5 radius
                if( x1*x1 + x2*x2 < .25f )
                        break;
        }

        int i = idx*2;
        dPlanes[i] = 2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2);
        dPlanes[i+1] = 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2); 
}

// initialize disp, plane, dual variable ...
__global__ void init_variables(float* dLDisp, float* dRDisp, float* dLPlanes, float* dRPlanes,
			       float* dLDispV, float* dRDispV, float* dLPlanesV, float* dRPlanesV,
			       float* dLDispPd, float* dRDispPd, float* dLPlanesPn, float* dRPlanesPn,
			       float* dLCost, float* dRCost, float* dLWeight, float* dRWeight,
			       cudaTextureObject_t lGradX_to, cudaTextureObject_t rGradX_to, 
			       cudaTextureObject_t lGradY_to, cudaTextureObject_t rGradY_to, 
     			       cudaTextureObject_t lGradXY_to, cudaTextureObject_t rGradXY_to, 
			       cudaTextureObject_t lGradYX_to, cudaTextureObject_t rGradYX_to, 
			       int cols, int rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows) return;	
		
	const int idx = y*cols+x;
	const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);
	
	dLDispV[idx] = dLDisp[idx];
	dRDispV[idx] = dRDisp[idx];
	/*dLDispPd[2*idx] = 0.f;
	dLDispPd[2*idx+1] = 0.f;
	dRDispPd[2*idx] = 0.f;
	dRDispPd[2*idx+1] = 0.f;*/
	//dLCost[idx] = BAD_COST;
	//dRCost[idx] = BAD_COST;
	
	// per-pixel weight
	dLWeight[idx] = expf( -3.0f* powf( tex2D<float>(lGradX_to, u, v)*tex2D<float>(lGradX_to, u, v)
			                  +tex2D<float>(lGradY_to, u, v)*tex2D<float>(lGradY_to, u, v)				
			      		  +tex2D<float>(lGradXY_to, u, v)*tex2D<float>(lGradXY_to, u, v)
			      		  +tex2D<float>(lGradYX_to, u, v)*tex2D<float>(lGradYX_to, u, v), 0.4f) );


	dRWeight[idx] = expf( -3.0f* powf( tex2D<float>(rGradX_to, u, v)*tex2D<float>(rGradX_to, u, v)
			      		  +tex2D<float>(rGradY_to, u, v)*tex2D<float>(rGradY_to, u, v) 
			      		  +tex2D<float>(rGradXY_to, u, v)*tex2D<float>(rGradXY_to, u, v)
			      		  +tex2D<float>(rGradYX_to, u, v)*tex2D<float>(rGradYX_to, u, v), 0.4f) );
	
	dLPlanesV[2*idx] = dLPlanes[2*idx];
	dLPlanesV[2*idx+1] = dLPlanes[2*idx+1];
	dRPlanesV[2*idx] = dRPlanes[2*idx];
	dRPlanesV[2*idx+1] = dRPlanes[2*idx+1];
	
	/*dLPlanesPn[4*idx] = 0.f;
	dLPlanesPn[4*idx+1] = 0.f;
	dLPlanesPn[4*idx+2] = 0.f;
	dLPlanesPn[4*idx+3] = 0.f;
	dRPlanesPn[4*idx] = 0.f;
	dRPlanesPn[4*idx+1] = 0.f;
	dRPlanesPn[4*idx+2] = 0.f;
	dRPlanesPn[4*idx+3] = 0.f;*/
}

float smoothstep(float a, float b, float x)
{
    float t = min((x - a)/(b - a), 1.);
	t = max(t, 0.);
    return t*t*(3.0 - (2.0*t));
}



__global__ void leftRightCheckHuber(float* dRDispPtr, float* dLDispPtr, float* dRPlanes, float* dLPlanes, 				    
				    int cols, int rows, float minDisp, float maxDisp, bool final)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows)
		return;	

	const int idx = y*cols+x;

	float tmp_disp = dLDispPtr[idx]*maxDisp;

	float nx0 = dLPlanes[2*idx];
	float ny0 = dLPlanes[2*idx+1];
	float nz0 = sqrtf(1.0f-nx0*nx0-ny0*ny0);

	int tmp_idx = x - (int)lroundf(tmp_disp);

	float nx1 = dRPlanes[2*(idx+tmp_idx-x)];
	float ny1 = dRPlanes[2*(idx+tmp_idx-x)+1];
	float nz1 = sqrtf(1.0f-nx1*nx1-ny1*ny1);
	
	// unit normal, dot = cos(angle)
	float dot = nx0*nx1+ny0*ny1+nz0*nz1;

	const float OCCLUSION = 2.0f;

	const float d_thresh = final ? 3.0f : 1.0f; //0.5f;

	const float theta_thresh = final ? cospif(30.0f/180.0f) : cospif(10.0f/*5.0f*//180.0f);

	if( (tmp_disp<0.f) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || 
	    fabsf(tmp_disp - dRDispPtr[idx + tmp_idx - x]*maxDisp) > d_thresh || dot < theta_thresh)
	{	
		dLDispPtr[idx] = final ? 0.0f : OCCLUSION;
	}


	tmp_disp = dRDispPtr[idx]*maxDisp;
	nx0 = dRPlanes[2*idx];
	ny0 = dRPlanes[2*idx+1];
	nz0 = sqrtf(1.0f-nx0*nx0-ny0*ny0);

	tmp_idx = x + (int)lroundf(tmp_disp);

	nx1 = dLPlanes[2*(idx+tmp_idx-x)];
	ny1 = dLPlanes[2*(idx+tmp_idx-x)+1];
	nz1 = sqrtf(1.0f-nx1*nx1-ny1*ny1);

	dot = nx0*nx1+ny0*ny1+nz0*nz1;

	if( (tmp_disp<0.f) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || 
	    fabsf(tmp_disp - dLDispPtr[idx + tmp_idx - x]*maxDisp) > d_thresh || dot < theta_thresh)
	{
		dRDispPtr[idx] = final ? 0.0f : OCCLUSION;
	}
}

__global__ void fillInOccludedHuber(float* dDisp, float* dPlanes, float* dDispV, float* dPlanesV, 
				    float* dDispPd, float* dPlanesPn, int cols, int rows, float maxDisp)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	const float OCCLUSION = 2.0f;

	if( dDisp[idx] != OCCLUSION )
		return;


/*	dDisp[idx] = 0.f;
	dDispV[idx] = 0.f;
	dPlanes[2*idx] = 0.f;
	dPlanes[2*idx+1] = 0.f;
	dPlanesV[2*idx] = 0.f;
	dPlanesV[2*idx+1] = 0.f;
	dDispPd[2*idx] = 0.f;
	dDispPd[2*idx+1] = 0.f;
	dPlanesPn[4*idx] = 0.f;
	dPlanesPn[4*idx+1] = 0.f;
	dPlanesPn[4*idx+2] = 0.f;
	dPlanesPn[4*idx+3] = 0.f;
	return;*/

	const float xf = (float)x;
	const float yf = (float)y;
	
	float nx, ny, nz, af, bf, cf, tmp_disp;
	//float u = (xf+0.5f)/(float)(cols);
	//float v = (yf+0.5f)/(float)(rows);

	// memory of min disp range (0, maxDisp)
	float disp = 2.0f*maxDisp;

	int best_i = 0;
	
	// search right
	int i = 1;
	while( x+i < cols )
	{
		if( dDisp[idx+i] != OCCLUSION )
		{
			nx = dPlanes[(idx+i)*2];
			ny = dPlanes[(idx+i)*2+1];
			nz = sqrtf(1.0f - nx*nx - ny*ny);

			// disp in memory ranges 0 to 1, scale back 
			float dispOrigionalScale = dDisp[idx+i]*maxDisp;

			// af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz			
			af = nx/nz*(-1.0f);
			bf = ny/nz*(-1.0f);
			cf = (nx*xf + ny*yf + nz*dispOrigionalScale)/nz;

			if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0 )
			{
				i++;
				continue;
			}
			// extrapolate
			tmp_disp = af*xf + bf*yf + cf;
			
			if(tmp_disp>=0.0f && tmp_disp <= maxDisp && tmp_disp == tmp_disp)
			{
				disp = tmp_disp;		
			}

			best_i = i;

			break;				
		}

		i++;
	}

	//search left for the nearest valid(none zero) disparity
	i = -1;
	while( x+i>=0 )
	{
		// valid disparity
		if( dDisp[idx+i] != OCCLUSION)
		{
			nx = dPlanes[(idx+i)*2];
			ny = dPlanes[(idx+i)*2+1];
			nz = sqrtf(1.f - nx*nx - ny*ny);

			// disp in memory ranges 0 to 1, scale back 
			float dispOrigionalScale = dDisp[idx+i]*maxDisp;
	
			af = nx/nz*(-1.0f);
			bf = ny/nz*(-1.0f);
			cf = (nx*xf + ny*yf + nz*dispOrigionalScale)/nz;
	
			if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 || isnan(af)!=0 || isnan(bf)!=0 || isnan(cf)!=0 )
			{
				i--;
				continue;
			}
	
			// extrapolate 
			tmp_disp = af*xf + bf*yf + cf;
			
			if(tmp_disp == tmp_disp && tmp_disp < disp)
			{
				if(tmp_disp>=0.0f && tmp_disp<=maxDisp)
				{
					disp = tmp_disp;			
				}

				best_i = i;
			}

			break;
		}
		
		i--;
	}

	if(best_i != 0)
	{
		dDisp[idx] = dDisp[idx+best_i];
		dPlanes[idx*2] = nx;
		dPlanes[idx*2+1] = ny;
		// auxiliary 
		dDispV[idx] = dDispV[idx+i];
		dPlanesV[idx*2] = dPlanesV[(idx+i)*2];
		dPlanesV[idx*2+1] = dPlanesV[(idx+i)*2+1];

		// primal
		dDispPd[2*idx] = dDispPd[2*(idx+i)];
		dDispPd[2*idx+1] = dDispPd[2*(idx+i)+1];
		dPlanesPn[4*idx] = dPlanesPn[4*(idx+i)];
		dPlanesPn[4*idx+1] = dPlanesPn[4*(idx+i)+1];
		dPlanesPn[4*idx+2] = dPlanesPn[4*(idx+i)+2];
		dPlanesPn[4*idx+3] = dPlanesPn[4*(idx+i)+3];
	}
	else	
	{
		dDisp[idx] = 0.0f;
		dPlanes[idx*2] = 0.0f;
		dPlanes[idx*2+1] = 0.0f;
		// auxiliary 
		dDispV[idx] = 0.0f;
		dPlanesV[idx*2] = 0.0f;
		dPlanesV[idx*2+1] = 0.0f;

		// primal
		dDispPd[2*idx] = 0.0f;
		dDispPd[2*idx+1] = 0.0f;
		dPlanesPn[4*idx] = 0.0f;
		dPlanesPn[4*idx+1] = 0.0f;
		dPlanesPn[4*idx+2] = 0.0f;
		dPlanesPn[4*idx+3] = 0.0f;
	}
}


void PatchMatchStereoHuberGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp)
{
	int cols = leftImg.cols;
	int rows = leftImg.rows;

	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(leftImg, cvLeftBGR_v);
	cv::split(rightImg, cvRightBGR_v);

	// BGR 2 grayscale
	cv::Mat cvLeftGray;
	cv::Mat cvRightGray;

	cv::cvtColor(leftImg, cvLeftGray, CV_BGR2GRAY);
	cv::cvtColor(rightImg, cvRightGray, CV_BGR2GRAY);	

	// convert to float
	cv::Mat cvLeftB_f;
	cv::Mat cvLeftG_f;
	cv::Mat cvLeftR_f;
	cv::Mat cvRightB_f;
	cv::Mat cvRightG_f;
	cv::Mat cvRightR_f;
	cv::Mat cvLeftGray_f;
	cv::Mat cvRightGray_f;	

	cvLeftBGR_v[0].convertTo(cvLeftB_f, CV_32F, 1./255.);
	cvLeftBGR_v[1].convertTo(cvLeftG_f, CV_32F, 1./255.);
	cvLeftBGR_v[2].convertTo(cvLeftR_f, CV_32F, 1./255.);	
	cvRightBGR_v[0].convertTo(cvRightB_f, CV_32F, 1./255.);	
	cvRightBGR_v[1].convertTo(cvRightG_f, CV_32F, 1./255.);	
	cvRightBGR_v[2].convertTo(cvRightR_f, CV_32F, 1./255.);	
	cvLeftGray.convertTo(cvLeftGray_f, CV_32F, 1./255.);
	cvRightGray.convertTo(cvRightGray_f, CV_32F, 1./255.);
		
	float* leftRImg_f = cvLeftR_f.ptr<float>(0);
	float* leftGImg_f = cvLeftG_f.ptr<float>(0);
	float* leftBImg_f = cvLeftB_f.ptr<float>(0);
	float* leftGrayImg_f = cvLeftGray_f.ptr<float>(0);
	float* rightRImg_f = cvRightR_f.ptr<float>(0);
	float* rightGImg_f = cvRightG_f.ptr<float>(0);
	float* rightBImg_f = cvRightB_f.ptr<float>(0);
	float* rightGrayImg_f = cvRightGray_f.ptr<float>(0);

	unsigned int imgSize = (unsigned int)cols*rows;

	// allocate floating disparity map, plane normals and gradient image (global memory)
	float* dRDisp = NULL;
	float* dLDisp = NULL;
	float* dLPlanes = NULL;
	float* dRPlanes = NULL;

	float* dLCost = NULL;
	float* dRCost = NULL;

	float* dLGradX = NULL;
	float* dRGradX = NULL;
	float* dLGradY= NULL;
	float* dRGradY= NULL;
	// 45 deg
	float* dLGradXY= NULL;
	float* dRGradXY= NULL;
	// 135 deg
	float* dLGradYX= NULL;
	float* dRGradYX= NULL;

	cudaMalloc(&dRDisp, imgSize*sizeof(float));
	cudaMalloc(&dLDisp, imgSize*sizeof(float));
	// changed 3 to 2, remove nz
	cudaMalloc(&dRPlanes, 2*imgSize*sizeof(float));
	cudaMalloc(&dLPlanes, 2*imgSize*sizeof(float));

	cudaMalloc(&dRCost, imgSize*sizeof(float));
	cudaMalloc(&dLCost, imgSize*sizeof(float));

	cudaMalloc(&dRGradX, imgSize*sizeof(float));
	cudaMalloc(&dLGradX, imgSize*sizeof(float));
	cudaMalloc(&dRGradY, imgSize*sizeof(float));
	cudaMalloc(&dLGradY, imgSize*sizeof(float));
	cudaMalloc(&dRGradXY, imgSize*sizeof(float));
	cudaMalloc(&dLGradXY, imgSize*sizeof(float));
	cudaMalloc(&dRGradYX, imgSize*sizeof(float));
	cudaMalloc(&dLGradYX, imgSize*sizeof(float));
	
	// huber smoothing
	float* dLPlanesV = NULL; 
	float* dRPlanesV = NULL;
	float* dLDispV = NULL;
	float* dRDispV = NULL;
	float* dLPlanesPn = NULL;
	float* dRPlanesPn = NULL;
	float* dLDispPd = NULL;
	float* dRDispPd = NULL;
	float* dLWeight = NULL;
	float* dRWeight =NULL;

	float* ddDisp_dx = NULL;
	float* ddDisp_dy = NULL;
	float* ddnx_dx = NULL;
	float* ddnx_dy = NULL;
	float* ddny_dx = NULL;
	float* ddny_dy = NULL;
	float* ddivPd = NULL;
	float* ddivPnx = NULL; 
	float* ddivPny = NULL;
	
	cudaMalloc(&dLPlanesV, 2*imgSize*sizeof(float));
	cudaMalloc(&dRPlanesV, 2*imgSize*sizeof(float));
	cudaMalloc(&dLPlanesPn, 4*imgSize*sizeof(float));
	cudaMalloc(&dRPlanesPn, 4*imgSize*sizeof(float));
	cudaMalloc(&dLDispV, imgSize*sizeof(float));
	cudaMalloc(&dRDispV, imgSize*sizeof(float));
	cudaMalloc(&dLDispPd, 2*imgSize*sizeof(float));
	cudaMalloc(&dRDispPd, 2*imgSize*sizeof(float));
	cudaMalloc(&dLWeight, imgSize*sizeof(float));
	cudaMalloc(&dRWeight, imgSize*sizeof(float));

	cudaMalloc(&ddDisp_dx, imgSize*sizeof(float));
	cudaMalloc(&ddDisp_dy, imgSize*sizeof(float));
	cudaMalloc(&ddnx_dx, imgSize*sizeof(float));	
	cudaMalloc(&ddnx_dy, imgSize*sizeof(float));
	cudaMalloc(&ddny_dx, imgSize*sizeof(float));
	cudaMalloc(&ddny_dy, imgSize*sizeof(float));
	cudaMalloc(&ddivPd, imgSize*sizeof(float));
	cudaMalloc(&ddivPnx, imgSize*sizeof(float));
	cudaMalloc(&ddivPny, imgSize*sizeof(float));
	
	// set dual variable to zero
	cudaMemset(dLPlanesPn, 0, 4*imgSize*sizeof(float));
	cudaMemset(dRPlanesPn, 0, 4*imgSize*sizeof(float));
	cudaMemset(dLDispPd, 0,  2*imgSize*sizeof(float));
	cudaMemset(dRDispPd, 0,  2*imgSize*sizeof(float));

	// set auxiliary variables zero
	cudaMemset(dLDispV, 0,  imgSize*sizeof(float));
	cudaMemset(dRDispV, 0,  imgSize*sizeof(float));
	cudaMemset(dLPlanesV, 0, 2*imgSize*sizeof(float));
	cudaMemset(dRPlanesV, 0, 2*imgSize*sizeof(float));

	cudaMemset(ddivPd, 0,  imgSize*sizeof(float));
	cudaMemset(ddivPnx, 0,  imgSize*sizeof(float));
	cudaMemset(ddivPny, 0,  imgSize*sizeof(float));

	cudaArray* lR_ca;
	cudaArray* lG_ca;
	cudaArray* lB_ca;
	cudaArray* lGray_ca;
	cudaArray* rR_ca;
	cudaArray* rG_ca;
	cudaArray* rB_ca;
	cudaArray* rGray_ca;
	cudaArray* lGradX_ca;
	cudaArray* rGradX_ca;
	cudaArray* lGradY_ca;
	cudaArray* rGradY_ca;
	cudaArray* lGradXY_ca;
	cudaArray* rGradXY_ca;
	cudaArray* lGradYX_ca;
	cudaArray* rGradYX_ca;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&lR_ca, &desc, cols, rows);
	cudaMallocArray(&lG_ca, &desc, cols, rows);
	cudaMallocArray(&lB_ca, &desc, cols, rows);
	cudaMallocArray(&lGray_ca, &desc, cols, rows);
	cudaMallocArray(&lGradX_ca, &desc, cols, rows);
	cudaMallocArray(&lGradY_ca, &desc, cols, rows);
	cudaMallocArray(&lGradXY_ca, &desc, cols, rows);
	cudaMallocArray(&lGradYX_ca, &desc, cols, rows);
	cudaMallocArray(&rR_ca, &desc, cols, rows);
	cudaMallocArray(&rG_ca, &desc, cols, rows);
	cudaMallocArray(&rB_ca, &desc, cols, rows);
	cudaMallocArray(&rGray_ca, &desc, cols, rows);
	cudaMallocArray(&rGradX_ca, &desc, cols, rows);
	cudaMallocArray(&rGradY_ca, &desc, cols, rows);
	cudaMallocArray(&rGradXY_ca, &desc, cols, rows);
	cudaMallocArray(&rGradYX_ca, &desc, cols, rows);

	cudaMemcpyToArray(lR_ca, 0, 0, leftRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lG_ca, 0, 0, leftGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lB_ca, 0, 0, leftBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lGray_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rR_ca, 0, 0, rightRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rG_ca, 0, 0, rightGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rB_ca, 0, 0, rightBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGray_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	
	// texture object
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = lR_ca;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeMirror;
	texDesc.addressMode[1] = cudaAddressModeMirror;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t lR_to = 0;
	cudaCreateTextureObject(&lR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lG_to = 0;
	resDesc.res.array.array = lG_ca;
	cudaCreateTextureObject(&lG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lB_to = 0;
	resDesc.res.array.array = lB_ca;
	cudaCreateTextureObject(&lB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lGray_to = 0;
	resDesc.res.array.array = lGray_ca;
	cudaCreateTextureObject(&lGray_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rR_to = 0;
	resDesc.res.array.array = rR_ca;
	cudaCreateTextureObject(&rR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rG_to = 0;
	resDesc.res.array.array = rG_ca;
	cudaCreateTextureObject(&rG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rB_to = 0;
	resDesc.res.array.array = rB_ca;
	cudaCreateTextureObject(&rB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGray_to = 0;
	resDesc.res.array.array = rGray_ca;
	cudaCreateTextureObject(&rGray_to, &resDesc, &texDesc, NULL);

	// kernels size
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x); 

	// image gradient
	imgGradient_huber<<<gridSize, blockSize>>>( cols, rows, lGray_to, rGray_to, 
				   dLGradX, dRGradX, dLGradY, dRGradY,
				   dLGradXY, dRGradXY, dLGradYX, dRGradYX);

	// copy gradient back sobel x
	cudaMemcpy(rightGrayImg_f, dRGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(leftGrayImg_f, dLGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad X", cvLeftGray_f);
	cv::imshow("Right grad X", cvRightGray_f);
	cv::waitKey(0);*/

	cudaMemcpyToArray(lGradX_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGradX_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaTextureObject_t lGradX_to = 0;
	resDesc.res.array.array = lGradX_ca;
	cudaCreateTextureObject(&lGradX_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradX_to = 0;
	resDesc.res.array.array = rGradX_ca;
	cudaCreateTextureObject(&rGradX_to, &resDesc, &texDesc, NULL);

	// sobel y
	cudaMemcpy(rightGrayImg_f, dRGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(leftGrayImg_f, dLGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad Y", cvLeftGray_f);
	cv::imshow("Right grad Y", cvRightGray_f);
	cv::waitKey(0);*/

	cudaMemcpyToArray(lGradY_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGradY_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaTextureObject_t lGradY_to = 0;
	resDesc.res.array.array = lGradY_ca;
	cudaCreateTextureObject(&lGradY_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradY_to = 0;
	resDesc.res.array.array = rGradY_ca;
	cudaCreateTextureObject(&rGradY_to, &resDesc, &texDesc, NULL);

	// central difference 45 deg
	cudaMemcpy(rightGrayImg_f, dRGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(leftGrayImg_f, dLGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad 45", cvLeftGray_f);
	cv::imshow("Right grad 45", cvRightGray_f);
	cv::waitKey(0);*/

	cudaMemcpyToArray(lGradXY_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGradXY_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaTextureObject_t lGradXY_to = 0;
	resDesc.res.array.array = lGradXY_ca;
	cudaCreateTextureObject(&lGradXY_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradXY_to = 0;
	resDesc.res.array.array = rGradXY_ca;
	cudaCreateTextureObject(&rGradXY_to, &resDesc, &texDesc, NULL);

	// central difference 135 deg
	cudaMemcpy(rightGrayImg_f, dRGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(leftGrayImg_f, dLGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad 135", cvLeftGray_f);
	cv::imshow("Right grad 135", cvRightGray_f);
	cv::waitKey(0);*/

	cudaMemcpyToArray(lGradYX_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGradYX_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaTextureObject_t lGradYX_to = 0;
	resDesc.res.array.array = lGradYX_ca;
	cudaCreateTextureObject(&lGradYX_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradYX_to = 0;
	resDesc.res.array.array = rGradYX_ca;
	cudaCreateTextureObject(&rGradYX_to, &resDesc, &texDesc, NULL);


	StartTimer();
                              
	// allocate memory for states
        curandState_t* states;
        cudaMalloc(&states, imgSize*sizeof(curandState_t));
        // initialize random states
        init<<<gridSize, blockSize>>>(1234, states, cols, rows);
        cudaDeviceSynchronize();	
	
        curandGenerator_t gen;
        // host CURAND
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        // set seed
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen, dLDisp, imgSize);
        cudaDeviceSynchronize();
		
        curandGenerator_t gen1;
        // host CURAND
        curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
        // set seed
        curandSetPseudoRandomGeneratorSeed(gen1, 4321ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen1, dRDisp, imgSize);
        cudaDeviceSynchronize();
		
        // random initial right and left plane
        init_plane_normals_huber<<<gridSize, blockSize>>>(dRPlanes, states, cols, rows);
        cudaDeviceSynchronize();

        init_plane_normals_huber<<<gridSize, blockSize>>>(dLPlanes, states, cols, rows);
        cudaDeviceSynchronize();

	std::cout<<"Init:"<<GetTimer()<<std::endl;
	
	StartTimer();

	const int huberIter = 5;
	int huberStartIter = 8;
	const float t_s_n_max = 50;
	const float t_s_n_offset = 5;
	const float t_s_d_max = t_s_n_max/Dmax;
	const float t_s_d_offset = t_s_n_offset/Dmax;

	// result
	cv::Mat cvLeftDisp_f, cvRightDisp_f;
	cvLeftDisp_f.create(rows, cols, CV_32F);
	cvRightDisp_f.create(rows, cols, CV_32F);


	for(int i=0; i<=iteration; i++)
	{
		float theta_sigma_d = 0.0f;
		float theta_sigma_n = 0.0f;
		
		if(i >= huberStartIter)
		{
			float tmp = smoothstep((float)huberStartIter, (float)iteration, (float)i);
			theta_sigma_d = tmp*t_s_d_max+t_s_d_offset;
			theta_sigma_n = tmp*t_s_n_max+t_s_n_offset;		
		}
		
		stereoMatching_huber<<<gridSize, blockSize>>>(dRDispV, dLDispV, dRPlanesV, dLPlanesV,
								dRDisp, dRPlanes, dLDisp, dLPlanes,
								dLCost, dRCost, cols, rows, winRadius,
								states, (float)Dmax, lR_to, lG_to, lB_to,
								lGray_to, lGradX_to, lGradY_to, lGradXY_to,lGradYX_to,
								rR_to, rG_to, rB_to,
								rGray_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
							        theta_sigma_d, theta_sigma_n);
	
		cudaDeviceSynchronize();

		std::cout<<i<<" theta_sigma_d:"<<theta_sigma_d<<" theta_sigma_n:"<<theta_sigma_n<<std::endl;


		cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cv::imshow("Left Disp", cvLeftDisp_f);
		cv::imshow("Right Disp", cvRightDisp_f);
		cv::waitKey(500);

#if 1
		// fix occlusion
/*		if(i >= huberStartIter)
		{
			leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, cols, rows, 0.f, (float)Dmax, false);

			cudaDeviceSynchronize();
		
			fillInOccludedHuber<<<gridSize, blockSize>>>(dLDisp, dLPlanes, dLDispV,  dLPlanesV, dLDispPd, dLPlanesPn,cols, rows, (float)Dmax);

			cudaDeviceSynchronize();

			fillInOccludedHuber<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dRDispV,  dRPlanesV, dRDispPd, dRPlanesPn, cols, rows, (float)Dmax);	

			cudaDeviceSynchronize();

			// copy disparity map from global memory on device to host
			cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cv::imshow("Left Disp", cvLeftDisp_f);
			cv::imshow("Right Disp", cvRightDisp_f);
			cv::waitKey(500);
		}*/

		if(i == huberStartIter)
		{
			// copy primal variables to auxiliary variables
			init_variables<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLDispV, dRDispV, dLPlanesV, dRPlanesV,
						dLDispPd, dRDispPd, dLPlanesPn, dRPlanesPn, dLCost, dRCost, dLWeight, dRWeight,
						lGradX_to, rGradX_to, lGradY_to, rGradY_to, lGradXY_to, rGradXY_to, lGradYX_to, rGradYX_to, cols, rows);
												
			cudaDeviceSynchronize();

			// copy disparity map from global memory on device to host
			/*cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRWeight, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLWeight, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cv::imshow("Left weight", cvLeftDisp_f);
			cv::imshow("Right weight", cvRightDisp_f);
			cv::waitKey(0);*/
			//cv::destroyWindow("Left Disp V init");
			//cv::destroyWindow("Right Disp V init");
		}

		if(i >= huberStartIter)
		{
			for(int j=0; j<huberIter; j++)
			{

				huberROFSmooth(dLDispV, dLPlanesV, dLDisp, dLPlanes,
					       dLDispPd, dLPlanesPn, dLWeight, ddDisp_dx,
					       ddDisp_dy, ddnx_dx, ddnx_dy, ddny_dx,
					       ddny_dy, ddivPd, ddivPnx, ddivPny,	
					       cols, rows, theta_sigma_d, theta_sigma_n);

				huberROFSmooth(dRDispV, dRPlanesV, dRDisp, dRPlanes,
					       dRDispPd, dRPlanesPn, dRWeight, ddDisp_dx,
					       ddDisp_dy, ddnx_dx, ddnx_dy, ddny_dx,
					       ddny_dy, ddivPd, ddivPnx, ddivPny,	
					       cols, rows, theta_sigma_d, theta_sigma_n);

				cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDispV, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDispV, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	
				std::cout<<"	Huber done-"<<j<<std::endl;

				cv::imshow("Right Disp V", cvRightDisp_f);
				cv::imshow("Left Disp V", cvLeftDisp_f);
				cv::waitKey(500);
			}
		}
#endif		
	}

#if 0
	leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, cols, rows, 0.f, (float)Dmax, false);

	cudaDeviceSynchronize();

	fillInOccludedHuber<<<gridSize, blockSize>>>(dLDisp, dLPlanes, dLDispV,  dLPlanesV, dLDispPd, dLPlanesPn, cols, rows, (float)Dmax);

	cudaDeviceSynchronize();

	fillInOccludedHuber<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dRDispV,  dRPlanesV, dRDispPd, dRPlanesPn, cols, rows, (float)Dmax);	

	cudaDeviceSynchronize();	
#endif

	leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, cols, rows, 0.f, (float)Dmax, true);

	cudaDeviceSynchronize();


	std::cout<<"Main loop:"<<GetTimer()<<std::endl;



        // copy disparity map from global memory on device to host
        cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	leftDisp = cvLeftDisp_f.clone();
	rightDisp = cvRightDisp_f.clone();

	leftDisp *= Dmax;
	rightDisp *= Dmax;

	if(showLeftDisp)
	{
		cv::imshow("Left Disp", cvLeftDisp_f);
		cv::imshow("Right Disp", cvRightDisp_f);
		std::cout<<"Press space"<<std::endl;
		cv::waitKey(0);
	}



        // Free device memory
	cudaFree(dRDisp);
	cudaFree(dRPlanes);
	cudaFree(dLDisp);
	cudaFree(dLPlanes);
	cudaFree(states);

	cudaFree(dLCost);
	cudaFree(dRCost);
	cudaFree(lR_ca);
	cudaFree(lG_ca);
	cudaFree(lB_ca);
	cudaFree(lGray_ca);
	cudaFree(lGradX_ca);
	cudaFree(lGradY_ca);
	cudaFree(lGradXY_ca);
	cudaFree(lGradYX_ca);
	cudaFree(rR_ca);
	cudaFree(rG_ca);
	cudaFree(rB_ca);
	cudaFree(rGray_ca);
	cudaFree(rGradX_ca);
	cudaFree(rGradY_ca);
	cudaFree(rGradXY_ca);
	cudaFree(rGradYX_ca);
	cudaFree(dLPlanesV); 
	cudaFree(dRPlanesV);
	cudaFree(dLDispV);
	cudaFree(dRDispV);
	cudaFree(dLPlanesPn);
	cudaFree(dRPlanesPn);
	cudaFree(dLDispPd);
	cudaFree(dRDispPd);
	cudaFree(dLWeight);
	cudaFree(dRWeight);
	cudaFree(ddDisp_dx);
	cudaFree(ddDisp_dy);
	cudaFree(ddnx_dx);
	cudaFree(ddnx_dy);
	cudaFree(ddny_dx);
	cudaFree(ddny_dy);
	cudaFree(ddivPd);
	cudaFree(ddivPnx);
	cudaFree(ddivPny);
	cudaFree(dRGradX);
	cudaFree(dLGradX);
	cudaFree(dLGradY);
	cudaFree(dRGradY);
	cudaFree(dLGradXY);
	cudaFree(dRGradXY);
	cudaFree(dLGradYX);
	cudaFree(dRGradYX);

	curandDestroyGenerator(gen);
	curandDestroyGenerator(gen1);
	cudaDestroyTextureObject(lR_to);
	cudaDestroyTextureObject(lG_to);
	cudaDestroyTextureObject(lB_to);
	cudaDestroyTextureObject(lGray_to);
	cudaDestroyTextureObject(lGradX_to);
	cudaDestroyTextureObject(rR_to);
	cudaDestroyTextureObject(rG_to);
	cudaDestroyTextureObject(rB_to);
	cudaDestroyTextureObject(rGray_to);
	cudaDestroyTextureObject(rGradX_to);

	cudaDeviceReset();
}


// load png image to a float image in RGB order 8 bit
void loadPNG(float* img_ptr, float* R, float* G, float* B, std::string file_name, int* cols, int* rows)
{
        std::vector<unsigned char> tmp_img;

        unsigned int width;
        unsigned int height;
        unsigned error = lodepng::decode(tmp_img, width, height, file_name);

	// how to save img 
        //error = lodepng::encode("new1.png", tmp_img, width, height);

        for(unsigned int y=0; y<height; y++)
        {
                for(unsigned int x=0; x<width; x++)
                {
                        unsigned int idx = x+y*width;
                        img_ptr[idx] = (float)(tmp_img[idx*4]+tmp_img[idx*4+1]+tmp_img[idx*4+2])/3.0f;
                //      img_ptr[idx] = (float)(tmp_img[idx*4+1]);
                        R[idx] = (float)tmp_img[idx*4];
                        G[idx] = (float)tmp_img[idx*4+1];
                        B[idx] = (float)tmp_img[idx*4+2];

                }
        }

//	*cols = (int)width;
//	*rows = (int)height;
}


// save png disparity image
void savePNG(unsigned char* disp, std::string fileName, int cols, int rows)
{
        std::vector<unsigned char> tmp_img;

	int imgSize = cols*rows;

        for(int i=0; i<imgSize; i++)
        {
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(255);
        }

        unsigned error = lodepng::encode(fileName, tmp_img, (unsigned int)cols, (unsigned int)rows);
}

// convert char image to float image and normalize to [0,1]
// if reverse is true, convert float to char
int imgCharToFloat(unsigned char* imgCharPtr, float* imgFloatPtr, bool reverse, unsigned int imgSize, float scale)
{
        if(!reverse)
        {
                for(int i=0; i<imgSize; i++)
                        imgFloatPtr[i] = (float)imgCharPtr[i];
        }
        else
        {
                for(int i=0; i<imgSize; i++)
                        imgCharPtr[i] = (unsigned char)(round(imgFloatPtr[i]*scale));
        }
        return 0;
}

void StartTimer()
{
        gettimeofday(&timerStart, NULL);
}

// time elapsed in ms
double GetTimer()
{
        struct timeval timerStop, timerElapsed;
        gettimeofday(&timerStop, NULL);
        timersub(&timerStop, &timerStart, &timerElapsed);
        return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

void timingStat(double* time, int nt, double* average, double* sd)
{

        *average = 0.0;
	*sd = 0.0;
	
	if(nt < 2)
	{
		*average = time[0];
		return;
	}

        for(int i=1; i<=nt; i++)
                *average += time[i];

        *average /= (double)nt;

      
        for(int i=1; i<=nt; i++)
                *sd += pow(time[i] - *average, 2);

        *sd = sqrt(*sd/(double)(nt-1));

        return;
}

void StartTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop)
{
	cudaEventCreate(start);
	cudaEventCreate(stop);

	cudaEventRecord(*start, 0);
}

float GetTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop)
{
	cudaEventRecord(*stop, 0);
	float time;
	cudaEventElapsedTime(&time, *start, *stop);
	cudaEventDestroy(*start);
	cudaEventDestroy(*stop);
	return time;
}

