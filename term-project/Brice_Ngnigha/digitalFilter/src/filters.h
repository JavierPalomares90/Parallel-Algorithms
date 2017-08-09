#ifndef _FILTERS_H
#define _FILTERS_H

#include "pixels.h"

namespace kernels
{
    void getProperties( int &devices, int &devId, struct cudaDeviceProp &props );
    pixel* init_mem( const pixel *h_in, int w, int h );
    void copy_mem( pixel *d_in, pixel *h_out, int w, int h );
    void end_mem( pixel *d_in );

    namespace filters
    {
        // Filters
        void convolution(  pixel *pixels, int w, int h, const std::vector<float> &args );
        void scale( pixel *pixels, int w, int h, const std::vector<float> &args );
        void modulate( pixel *pixels, int w, int h, const std::vector<float> &args );
    }

}
#endif
