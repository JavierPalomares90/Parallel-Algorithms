#ifndef _PIXEL_H
#define _PIXEL_H

#include <vector>
#include <stdint.h>

struct accumulator;
struct pixel;

typedef struct pixel{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
    bool valid;

    __device__ pixel& operator+( const pixel &p );
    __device__ pixel& operator=( const pixel &p );
    __device__ pixel& operator=( const accumulator &p );
    __device__ pixel& operator*( const double &m );

    __device__ void copy_from( const pixel &p );
    __device__ void attenuate( double coeff );
    __device__ void apply_weights( float wr, float wg, float wb );
    __device__ void grayScale( float wr, float wg, float wb );

}pixel;

typedef struct accumulator{
    float r;
    float g;
    float b;
    float a;
    bool valid;

    __device__ void accumulate( pixel p, float coeff );
    __device__ void apply_weights( float wr, float wg, float wb );
    __device__ void grayScale( float wr, float wg, float wb );

}accumulator;

#endif
