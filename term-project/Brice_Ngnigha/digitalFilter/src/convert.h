#ifndef _CONVERT_H
#define _CONVERT_H

#include "pixels.h"

namespace convert
{
    pixel* uint8_arr_to_pixel_arr( uint8_t *image, int w, int h, int chs );
    uint8_t* pixel_arr_to_uint8_arr( pixel *pixels, int w, int h, int chs );
}
#endif
