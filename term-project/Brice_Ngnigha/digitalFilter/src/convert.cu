#include "convert.h"
#include <iostream>

using namespace std;
using namespace convert;

pixel* convert::uint8_arr_to_pixel_arr( uint8_t *image, int w, int h, int chs )
{
    bool ch4 = (chs == 4);
    uint8_t *imgPtr = image;

    int len = w * h;
    pixel *pixels = (pixel*) malloc( len * sizeof(pixel) );

    for( int i = 0; i < len; i++ )
    {
        if(imgPtr == NULL)
        {
            cout << "Wrong addressing? " << i <<  endl;
            return NULL;
        }

        pixels[i].r = imgPtr[0];
        pixels[i].g = imgPtr[1];
        pixels[i].b = imgPtr[2];

        if( ch4 )
            pixels[i].a = imgPtr[3];
        else
            pixels[i].a = 0;

        imgPtr += chs;
    }

    return pixels;
}

uint8_t* convert::pixel_arr_to_uint8_arr( pixel *pixels, int w, int h, int chs )
{
    int len = w *w * chs;
    bool ch4 = (chs == 4);

    uint8_t *image  = (uint8_t*) malloc( len*sizeof(uint8_t) );
    uint8_t *imgPtr = image;

    for( int i = 0; i < w*h; i++ )
    {
        if(imgPtr == NULL)
        {
            cout << "Wrong addressing? " << i <<  endl;
            return NULL;
        }

        imgPtr[0] = pixels[i].r;
        imgPtr[1] = pixels[i].g;
        imgPtr[2] = pixels[i].b;

        if( ch4 )
            imgPtr[3] = pixels[i].a;

        imgPtr += chs;
    }

    return image;
}
