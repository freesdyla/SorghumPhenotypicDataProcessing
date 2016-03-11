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


using namespace std;

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
        float min_cost, tmp_min_cost;
        float cost;
        float tmp_disp;
        float s;
        int tmp_idx;
        int new_x;
        int best_i, tmp_best_i;
        int best_j, tmp_best_j;
        bool VIEW_PROPAGATION = true;
        bool PLANE_REFINE = true;
        //--------------------------------------------
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


__global__ void gradient(float* lGradPtr, float*rGradPtr, cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows)
                return;

        int idx = y*cols+x;
        float u = ((float)x+0.5f)/(float)(cols);
        float v = ((float)y+0.5f)/(float)(rows);

	float tmp;

/*	tmp =  3.0f*(tex2D<float>(lGray_to, u + 1.0f/(float)(cols), v - 1.0f/(float)rows ) - tex2D<float>(lGray_to, u - 1/(float)(cols), v - 1.0f/(float)rows)) ;
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


	return;		
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
	float* dLPlanes = NULL;
	float* dRPlanes = NULL;
	float* dLGrad = NULL;
	float* dRGrad = NULL;
	float* dLCost = NULL;
	float* dRCost = NULL;

	cudaMalloc(&dRDisp, imgSize*sizeof(float));
	cudaMalloc(&dLDisp, imgSize*sizeof(float));
	cudaMalloc(&dRPlanes, 3*imgSize*sizeof(float));
	cudaMalloc(&dLPlanes, 3*imgSize*sizeof(float));
	cudaMalloc(&dRGrad, imgSize*sizeof(float));
	cudaMalloc(&dLGrad, imgSize*sizeof(float));
	cudaMalloc(&dRCost, imgSize*sizeof(float));
	cudaMalloc(&dLCost, imgSize*sizeof(float));

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

	cout<<"Random Init:"<<GetTimer()<<endl;
	
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

	cout<<"Main loop:"<<GetTimer()<<endl;
	


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
	leftRightCheck<<<gridSize, blockSize>>>(dRDisp, dLDisp, dLOccludeMask, dROccludeMask, cols, rows, (float)Dmin, (float)Dmax, 1);

	cudaDeviceSynchronize();

/*	fillInOccluded<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLCost, dRCost, dLOccludeMask, 
								dROccludeMask, cols, rows, winRadius, (float)Dmax, 1,
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);
	cudaDeviceSynchronize();	

       	weightedMedianFilter<<<gridSize, blockSize>>>(dLDisp, dLOccludeMask, cols, rows, winRadius, (float)Dmax, 
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);

	cudaDeviceSynchronize();
*/
	cout<<"Post Process:"<<GetTimer()<<endl;	
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

