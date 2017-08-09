#include "main.h"


using namespace std;

void load_vector( vector<float> &vec, const float A[], int len )
{
    float sum = 0.0;
    for(int i = 0; i < len; i++)
    {
        vec.push_back(A[i]);
        sum += A[i];
    }

    if( sum > 1.001 || sum < -0.001f)
      {

        // Normalize
        for(int j = 0; j < len; j++ )
        {
            vec[j] /= sum;
        }
    }
}

void load_filters( vector<filter> &filters, int start, int argc, char **argv )
{
    string arg;
    int i = start;

    while( i < argc )
    {
        filter f;
        arg = string(argv[i]);
        if( arg == "scale" )
        {
            i++;
            f.type = SCALE;

            int n = atoi(argv[i]);
	    i++;
            for(int j = 0; j < 3; j++ )
            {
                f.args.push_back( atof(argv[i]) );
		i++;
            }
        }
        else if( arg == "modulate" )
        {
            i++;
            f.type = MODULATE;

            int n = atoi(argv[i++]);
            for(int j = 0; j < 3; j++ )
            {
                f.args.push_back( atof(argv[i++]) );
            }
        }
        else if( arg == "convolution" )
        {
            i++;
            f.type = CONVOLUTION;

            int n = atoi(argv[i]);
	    i++;
            float sum = 0;
            for(int j = 0; j < n; j++ )
            {
                f.args.push_back( atof(argv[i]) );
                sum += f.args[j];
                i++;
            }

            // Normalize
            if( sum > 1.001 || sum < -.9999 )
            {
                for(int j = 0; j < n; j++ )
                {
                    f.args[j] /= sum;
                }
            }
        }
        else if( arg == "identity" )
        {
            i++;
            f.type = IDENTITY;
            load_vector(f.args, idKernel, 9);
        }
        else if( arg == "edge" )
        {
            i++;
            f.type = EDGE_1;
            load_vector(f.args, edgeKernel1, 9);
        }
        else if( arg == "edge2" )
        {
            i++;
            f.type = EDGE_2;
            load_vector(f.args, edgeKernel2, 9);
        }
        else if( arg == "edge3" )
        {
            i++;
            f.type = EDGE_3;
            load_vector(f.args, edgeKernel3, 9);
        }
        else if( arg == "sharpen" )
        {
            i++;
            f.type = SHARPEN;
            load_vector(f.args, sharpKernel, 9);
        }
        else if( arg == "boxblur" )
        {
            i++;
            f.type = BOX_BLUR;
            load_vector(f.args, boxBlurKernel, 9);
        }
        else if( arg == "gaussian" )
        {
            i++;
            f.type = GAUSSIAN_BLUR_3;
            load_vector(f.args, gaussKernel3, 9);
        }
        else if( arg == "gaussian5" )
        {
            i++;
            f.type = GAUSSIAN_BLUR_5;
            load_vector(f.args, gaussKernel5, 25);
        }
        else if( arg == "unsharpen" )
        {
            i++;
            f.type = UNSHARP;
            load_vector(f.args, unsharpKernel, 25);
        }
        else if( arg == "grayscale" )
        {
            i++;
            f.type = GRAYSCALE;
            load_vector(f.args, grayKernel, 3);
        }
        else if( arg == "emboss" )
        {
            i++;
            f.type = EMBOSS;
            load_vector(f.args, embossKernel, 9);
        }
        else
        {
            cout << "Unsupported operation " << endl;
            exit(-1);
        }

        filters.push_back( f );
    }
}

void process_image( const vector<filter> &filters, pixel *devImage, \
		    const string &filename, int w, int h, int chs )
{
    string out;

    for( int i = 0; i < filters.size(); i++ )
    {
        char opId[3];
        sprintf(opId, "%d", i+1);
        Type type = filters[i].type;

        if( type == SCALE )
        {
            cout << "Scale" << endl;
            kernels::filters::scale( devImage, w, h, filters[i].args );
            out += string(opId) + "_scale_";
        }
        else if( type == CONVOLUTION )
        {
            cout << "Convolution" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_convolution_";
        }
        else if( type == IDENTITY )
        {
            cout << "Identity" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out +=string(opId) + "_indentity_";
        }
        else if( type == EDGE_1 )
        {
            cout << "Edge 1" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_edge1_";
        }
        else if( type == EDGE_2 )
        {
            cout << "Edge 2" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_edge2_";
        }
        else if( type == EDGE_3 )
        {
            cout << "Edge 3" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_edge3_";
        }
        else if( type == GAUSSIAN_BLUR_3 )
        {
            cout << "Gaussian 3" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out +=string(opId) + "_gaussian_";
        }
        else if( type == GAUSSIAN_BLUR_5 )
        {
            cout << "Gaussian 5" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_gaussian5_";
        }
        else if( type == EMBOSS )
        {
            cout << "Emboss" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_emboss_";
        }
        else if( type == SHARPEN )
        {
            cout << "Sharpen" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_sharpen_";
        }
        else if( type == UNSHARP )
        {
            cout << "Unsharpen" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out += string(opId) + "_unsharpen_";
        }
        else if( type == BOX_BLUR )
        {
            cout << "Box blur" << endl;
            kernels::filters::convolution( devImage, w, h, filters[i].args );
            out +=string(opId) + "_boxblur_";
        }
        else if( type == GRAYSCALE )
        {
            cout << "Grayscale" << endl;
            kernels::filters::scale( devImage, w, h, filters[i].args );
            out += string(opId) + "_grayscale_";
        }
        else if( type == MODULATE )
        {
            cout << "Modulate" << endl;
            kernels::filters::modulate( devImage, w, h, filters[i].args );
            out += string(opId) + "_modulate_";
        }
        else
        {
            cout << "No filter type selected " << endl;
	    exit(0);
        }

	// copy result form device image
	pixel *hostImage = (pixel*) malloc( w*h*sizeof(pixel) );
	kernels::copy_mem( devImage, hostImage, w, h );

	// write result image to file
	string out2 = "output/step_" + out + filename;
	cout << "Saving to: " << out << "\n___________________________________________________________ " << endl;
	uint8_t *rgb_image = convert::pixel_arr_to_uint8_arr( hostImage, w, h, chs );
	stbi_write_png( out2.c_str(), w, h, chs, rgb_image, w * chs );
	free(rgb_image);

    }

    // // copy result form device image
    // pixel *hostImage = (pixel*) malloc( w*h*sizeof(pixel) );
    // kernels::copy_mem( devImage, hostImage, w, h );

    // // write result image to file
    // out = "output/" + out + filename;
    // cout << "Saving to: " << out << endl;
    // uint8_t *rgb_image = convert::pixel_arr_to_uint8_arr( hostImage, w, h, chs );
    // stbi_write_png( out.c_str(), w, h, chs, rgb_image, w * chs );
    // free(rgb_image);
}


int main(int argc, char ** argv)
{
    if( argc < 2 )
        return 1;

    int width, height, channels;
    char * pic = argv[1];
    string out("picout.png");
    vector<filter> filters;

    // Loading data
    load_filters(filters, 2, argc, argv);
    uint8_t* rgb_image = stbi_load( pic, &width, &height, &channels, STBI_rgb_alpha );

    if(rgb_image == NULL)
    {
        cout << "NULL IMAGE" << endl;
        return -1;
    }

    // convert uint8_t image to pixels array
    pixel *pixels = convert::uint8_arr_to_pixel_arr( rgb_image, width, height, channels );

    // Allocate memory and copy pixels to device memory
    pixel *devPtr = kernels::init_mem( pixels, width, height);

    // free device images
    free(pixels);
    stbi_image_free( rgb_image );

    // do image processing
    process_image( filters, devPtr, pic, width, height, channels );


    kernels::end_mem(devPtr);

    cout << "image properties "<< width << " " << height << " " << channels << endl;
    return 0;
}
