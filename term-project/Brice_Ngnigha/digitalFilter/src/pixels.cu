#include "pixels.h"

using namespace std;

__device__ pixel& pixel::operator+( const pixel &p1 )
{
    r += p1.r;
    g += p1.g;
    b += p1.b;
    //a += p1.a;
    return *this;
}

__device__ pixel& pixel::operator=( const pixel &p1 )
{
    r = p1.r;
    g = p1.g;
    b = p1.b;
    //a = p1.a;
    valid = p1.valid;
    return *this;
}

__device__ pixel& pixel::operator=( const accumulator &p1 )
{
    r = p1.r;
    g = p1.g;
    b = p1.b;
    //a = p1.a;
    valid = p1.valid;
    return *this;
}

__device__ pixel& pixel::operator*( const double &m )
{
    this->r *= m;
    this->g *= m;
    this->b *= m;
    //this->a *= m;
    return *this;
}

__device__ void pixel::copy_from( const pixel &p )
{
    this->r = p.r;
    this->g = p.g;
    this->b = p.b;
    //this->a = p.a;
    this->valid = p.valid;
}

__device__ void pixel::attenuate( double coeff )
{
    this->r *= coeff;
    this->g *= coeff;
    this->b *= coeff;
}

__device__ void pixel::apply_weights( float wr, float wg, float wb )
{
    this->r *= wr;
    this->g *= wg;
    this->b *= wb;
}

#include "stdio.h"
__device__ void pixel::grayScale( float wr, float wg, float wb )
{
    float I = r*wr + g*wg + b*wb;
    this->r = I;
    this->g = I;
    this->b = I;
}

__device__ void accumulator::accumulate( pixel p, float coeff )
{
    this->r += coeff * p.r;
    this->g += coeff * p.g;
    this->b += coeff * p.b;
}
