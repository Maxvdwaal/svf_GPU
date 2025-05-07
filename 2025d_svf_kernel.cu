#include <math.h>
#define PI 3.141592654

__device__ float Deg2Rad(float degree)
{
    return degree * PI / 180.0f;
}

__device__ float AnnulusWeight(float alt, float alt_interval, float azi_interval)
{
    /*
    Calculate the weighing factor of each sampling point in the RayTracing process.
    Based on the formula of the area of sphere segment (Area = (azimuth1 - azimuth2) * (cos(altitude1) - cos(altitude2))) divided by the total area of the hemisphere.
    
    Parameters:
        - alt: altitude of the sampling point in radians
        - alt_interval: altitude interval in radians
        - azi_interval: azimuth interval in radians
        - area_total: total area of the hemisphere

    Returns:
        - weighted_area_ray: the weighing factor of the sampling point
    */
   float area_ray = azi_interval * (cosf(alt - alt_interval/2.0f) - cosf(alt + alt_interval/2.0f)); 
   float weighted_area_ray = area_ray * cosf(alt);

   return weighted_area_ray;
}

__global__ void svfcalculator(float *svf_out, float *svfE_out, float *svfS_out, float *svfW_out, float *svfN_out,
                              float *svfveg_out, float *svfEveg_out, float *svfSveg_out, float *svfWveg_out, float *svfNveg_out,
                              float *svfaveg_out, float *svfEaveg_out, float *svfSaveg_out, float *svfWaveg_out, float *svfNaveg_out,
                              float *dsm, float *cdsm, float *tdsm,
                              float scale, int width, int height,
                              float traceRadius, float azimuth_start, float azimuth_end, float azimuth_interval, float altitude_interval)
{
        /**
     * Calculate Sky View Factor (SVF) using ray tracing on a GPU.
     *
     * This kernel function calculates the SVF for each pixel in the input DSM (Digital Surface Model) and vegetation models.
     * The SVF is calculated for different azimuth and altitude angles, and the results are stored in the output arrays.
     *
     * Args:
     *     svf_out (float*): Output array for overall SVF.
     *     svfE_out (float*): Output array for eastward SVF.
     *     svfS_out (float*): Output array for southward SVF.
     *     svfW_out (float*): Output array for westward SVF.
     *     svfN_out (float*): Output array for northward SVF.
     *     svfveg_out (float*): Output array for overall vegetation SVF.
     *     svfEveg_out (float*): Output array for eastward vegetation SVF.
     *     svfSveg_out (float*): Output array for southward vegetation SVF.
     *     svfWveg_out (float*): Output array for westward vegetation SVF.
     *     svfNveg_out (float*): Output array for northward vegetation SVF.
     *     svfaveg_out (float*): Output array for overall adjusted vegetation SVF.
     *     svfEaveg_out (float*): Output array for eastward adjusted vegetation SVF.
     *     svfSaveg_out (float*): Output array for southward adjusted vegetation SVF.
     *     svfWaveg_out (float*): Output array for westward adjusted vegetation SVF.
     *     svfNaveg_out (float*): Output array for northward adjusted vegetation SVF.
     *     dsm (float*): Input DSM array.
     *     cdsm (float*): Input vegetation height model array.
     *     tdsm (float*): Input vegetation canopy bottom height model array.
     *     scale (float): Scale factor for the height.
     *     width (int): Width of the input arrays.
     *     height (int): Height of the input arrays.
     *     traceRadius (float): The radius for ray tracing.
     *     azimuth_start (float): Starting azimuth angle in degrees.
     *     azimuth_end (float): Ending azimuth angle in degrees.
     *     azimuth_interval (float): Azimuth interval in degrees.
     *     altitude_interval (float): Altitude interval in degrees.
     *
     * Returns:
     *     svf's, svfveg's and svfaveg's are stored in the output arrays.
     */
    //Calculate the index of the current thread and avoid out-of-bounds access
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= width || iy >= height)
        return;
    int index = ix + iy * width;

    //Read the DSM values at the current index
    float clr00 = dsm[index];

    // Initialize the SVF arrays
    float svf = 0.0f;
    float svfE = 0.0f;
    float svfS = 0.0f;
    float svfW = 0.0f;
    float svfN = 0.0f;

    float svfveg = 0.0f;
    float svfEveg = 0.0f;
    float svfSveg = 0.0f;
    float svfWveg = 0.0f;
    float svfNveg = 0.0f;

    float svfaveg = 0.0f;
    float svfEaveg = 0.0f;
    float svfSaveg = 0.0f;
    float svfWaveg = 0.0f;
    float svfNaveg = 0.0f;

    //Initialize the angle parameters
    float alt_start = Deg2Rad(0.0f);
    float alt_end = Deg2Rad(90.0f);
    float alt_interval = Deg2Rad(altitude_interval);

    float azi_start = Deg2Rad(0.0f);
    float azi_end = Deg2Rad(360.0f);
    float azi_interval = Deg2Rad(azimuth_interval);

    float total_ray_area = 0.0f;
    float total_ray_areaE = 0.0f;
    float total_ray_areaS = 0.0f;
    float total_ray_areaW = 0.0f;
    float total_ray_areaN = 0.0f;

    //Main calculation loops for the SVFs
    for (float alt = alt_start; alt <= alt_end; alt += alt_interval)
    {
        float weight   = AnnulusWeight(alt, alt_interval, azi_interval);

        for (float azimuth = azi_start; azimuth < azi_end; azimuth += azi_interval)
        {
            float radius = 1.0f;
            float cos_azi = cosf(azimuth);
            float sin_azi = sinf(azimuth);
            bool crossed_veg = false;
            bool crossed_dsm = false;

            total_ray_area += weight;

            if(azimuth >= 0.0f && azimuth <= PI){
                total_ray_areaS += weight;
            } if (azimuth >= (0.5f * PI) && azimuth <= (1.5f * PI)){
                total_ray_areaW += weight;
            } if (azimuth >= (1.0f * PI) && azimuth <= (2.0f * PI)){
                total_ray_areaN += weight;
            } if (azimuth >= (1.5f * PI) || azimuth <= (0.5f * PI)){
                total_ray_areaE += weight;
            }            

            // Trace the ray until traceRadius
            while (radius <= traceRadius)
            {
                float dx = cos_azi;
                float dy = sin_azi;
                float x_float = float(ix);
                float y_float = float(iy);
                int x = int(roundf(x_float + radius * cosf(alt) * dx));
                int y = int(roundf(y_float + radius * cosf(alt) * dy));               
                if (x < 0 || x >= width || y < 0 || y >= height)
                    break;
                int index2 = x + y * width;

                float dsm_height = dsm[index2];
                float veg_height = cdsm[index2];
                float canopy_bottom_height = tdsm[index2];
                float ray_height = clr00 + radius/scale * sinf(alt);

                //Avegetation SVF calculations + anisotropics
                if (ray_height < dsm_height && crossed_veg && !crossed_dsm){
                    svfaveg += weight;
                    svfveg -= weight;
                    if(azimuth >= 0.0f && azimuth <= PI){
                        svfSaveg += weight;
                        svfSveg -= weight;
                    }
                    if(azimuth >= (0.5f * PI) && azimuth <= (1.5f * PI)){
                        svfWaveg += weight;
                        svfWveg -= weight;
                    }
                    if(azimuth >= (1.0f * PI) && azimuth <= (2.0f * PI)){
                        svfNaveg += weight;
                        svfNveg -= weight;
                    }
                    if(azimuth >= (1.5f * PI) || azimuth <= (0.5f * PI)){
                        svfEaveg += weight ;
                        svfEveg -= weight;
                    }
                }

                //Regular SVF calculations + anisotropics
                if (ray_height < dsm_height && !crossed_dsm)
                {
                    svf += weight;                    
                    if(azimuth >= 0.0f && azimuth <= PI){
                        svfS += weight;
                    }
                   if(azimuth >= (0.5f * PI) && azimuth <= (1.5f * PI)){
                        svfW += weight;
                    }
                    if(azimuth >= (1.0f * PI) && azimuth <= (2.0f * PI)){
                        svfN += weight;
                    }
                    if(azimuth >= (1.5f * PI) || azimuth <= (0.5f * PI)){
                        svfE += weight;
                    }
                    crossed_dsm = true;
                }

                //Vegetation SVF calculations + anisotropics
                if (ray_height < veg_height && ray_height > canopy_bottom_height && !crossed_dsm && !crossed_veg){
                    svfveg += weight;
                    if(azimuth >= 0.0f && azimuth <= PI){
                        svfSveg += weight;
                    }
                   if(azimuth >= (0.5f * PI) && azimuth <= (1.5f * PI)){
                        svfWveg += weight;
                    }
                    if(azimuth >= (1.0f * PI) && azimuth <= (2.0f * PI)){
                        svfNveg += weight;
                    }
                    if(azimuth >= (1.5f * PI) || azimuth <= (0.5f * PI)){
                        svfEveg += weight;
                    }
                    crossed_veg = true;
                }

                if(crossed_dsm && crossed_veg){
                    break;
                }

                // Adjust step size based on altitude angle
                float step_size = fmaxf(1.0f, (radius * cosf(alt)) * 0.1f);
                radius += step_size;
            }
        }
    }

    // Write the result to the output
    svf_out[index]  = 1.0f - svf/total_ray_area;
    svfE_out[index] = 1.0f - svfE/total_ray_areaE;
    svfS_out[index] = 1.0f - svfS/total_ray_areaS;
    svfW_out[index] = 1.0f - svfW/total_ray_areaW;
    svfN_out[index] = 1.0f - svfN/total_ray_areaN;

    svfveg_out[index]  = 1.0f - svfveg/total_ray_area;
    svfEveg_out[index] = 1.0f - svfEveg/total_ray_areaE;
    svfSveg_out[index] = 1.0f - svfSveg/total_ray_areaS;
    svfWveg_out[index] = 1.0f - svfWveg/total_ray_areaW;
    svfNveg_out[index] = 1.0f - svfNveg/total_ray_areaN;

    svfaveg_out[index]  = 1.0f - svfaveg/total_ray_area;
    svfEaveg_out[index] = 1.0f - svfEaveg/total_ray_areaE;
    svfSaveg_out[index] = 1.0f - svfSaveg/total_ray_areaS;
    svfWaveg_out[index] = 1.0f - svfWaveg/total_ray_areaW;
    svfNaveg_out[index] = 1.0f - svfNaveg/total_ray_areaN;
}