#ifndef _MAIN_H
#define _MAIN_H

#include <string>
#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <iostream>


#include "pixels.h"
#include "filters.h"
#include "convert.h"


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

enum  Type
{
    SCALE,
    MODULATE,
    CONVOLUTION,

    IDENTITY,
    EDGE_1,
    EDGE_2,
    EDGE_3,
    SHARPEN,
    BOX_BLUR,
    GAUSSIAN_BLUR_3,
    GAUSSIAN_BLUR_5,
    UNSHARP,
    GRAYSCALE,
    EMBOSS
};

typedef struct
{
    Type type;
    std::vector<float> args;
}filter;

// Convolution Kernels
const float idKernel[]  = { 0, 0, 0,
                            0, 1, 0,
                            0, 0, 0
};

const float edgeKernel1[] = { 1, 0, -1,
                              0, 0,  0,
                              0, 1,  0
};

const float edgeKernel2[] = { 1, 1,  0,
                              1, -4, 1,
                              0, 1,  0
};

const float edgeKernel3[] = { -1, -1, -1,
                              -1,  8, -1,
                              -1, -1, -1
};

const float sharpKernel[] = {  0, -1,  0,
                              -1,  5, -1,
                               0, -1,  0
};


const float boxBlurKernel[]={ 1, 1, 1,
                              1, 1, 1,
                              1, 1, 1
};

const float gaussKernel3[] = { 1, 2, 1,
                               2, 4, 2,
                               1, 2, 1
};


const float gaussKernel5[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1
};

const float unsharpKernel[] = { 1,  4,    6,  4, 1,
                                4, 16,   24, 16, 4,
                                6, 24, -476, 24, 6,
                                4, 16,   24, 16, 4,
                                1,  4,    6,  4, 1
};

const float grayKernel[]   = { .299f, .587f, .114f };

const float embossKernel[] = { -1, -1, 0,
                               -1,  0, 1,
                                0,  1, 1
};





#endif
