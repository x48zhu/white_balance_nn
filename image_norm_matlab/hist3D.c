#include <math.h>
#include "mex.h"

#define BOX_NUMBER_IN_BIT  7
#define BOX_NUMBER  	   (1<<BOX_NUMBER_IN_BIT)

#define MAXCOLORS          (BOX_NUMBER*BOX_NUMBER*BOX_NUMBER)
#define SHIFT_DIV          (8-BOX_NUMBER_IN_BIT)
#define SHIFT_IND_R1       (BOX_NUMBER_IN_BIT*2)
#define SHIFT_IND_G1       (BOX_NUMBER_IN_BIT)


void check_inputs(int nrhs, const mxArray *prhs[]) {
    const int *size;

    if (nrhs != 1)
    {
        mexErrMsgIdAndTxt("oneInputsRequired",
                          "%s","Only 1 input arguments required.");
    }
    
    if (!mxIsDouble(prhs[0])) {
        mexErrMsgIdAndTxt("firstInputMustBeDouble",
                          "%s","First input must be a uint8 array.");
    }
    
    if (mxGetNumberOfDimensions(prhs[0]) != 3) {
        mexErrMsgIdAndTxt("firstInputMustBe3DDouble",
                          "%s","First input must be a 3-D uint8 array.");
    }
    
    
    size = mxGetDimensions(prhs[0]);
    if (size[2] != 3) {
        mexErrMsgIdAndTxt("firstInputMustBeMbyNby3",
                          "%s","First input must be M-by-N-by-3.");
    }
    
    return;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    double r,g,b;
    int r8,g8,b8,inr,ing,inb,ind;
    int i;
    const int *size;
    int num_pixels;
    double *image_r, *image_g, *image_b, *map_r, *map_g, *map_b, *count;
    double sum_r[BOX_NUMBER*BOX_NUMBER*BOX_NUMBER];
    double sum_g[BOX_NUMBER*BOX_NUMBER*BOX_NUMBER];
    double sum_b[BOX_NUMBER*BOX_NUMBER*BOX_NUMBER];
    
    check_inputs(nrhs, prhs);

    size = mxGetDimensions(prhs[0]);
    num_pixels = size[0] * size[1];
    
    image_r = (double *) mxGetData(prhs[0]);
    image_g = image_r + num_pixels;
    image_b = image_r + num_pixels*2;    
    
    plhs[0] = mxCreateNumericMatrix(BOX_NUMBER*BOX_NUMBER*BOX_NUMBER, 1, mxDOUBLE_CLASS, mxREAL);
    count = (double *) mxGetData(plhs[0]);
    plhs[1] = mxCreateNumericMatrix(BOX_NUMBER*BOX_NUMBER*BOX_NUMBER, 3, mxDOUBLE_CLASS, mxREAL);
    map_r = (double *) mxGetData(plhs[1]);
    map_g = map_r + BOX_NUMBER*BOX_NUMBER*BOX_NUMBER;
    map_b = map_r + BOX_NUMBER*BOX_NUMBER*BOX_NUMBER*2;
    
    
    
    memset(count,0,BOX_NUMBER*BOX_NUMBER*BOX_NUMBER*sizeof(double));
    memset(sum_r,0,BOX_NUMBER*BOX_NUMBER*BOX_NUMBER*sizeof(double));
    memset(sum_g,0,BOX_NUMBER*BOX_NUMBER*BOX_NUMBER*sizeof(double));
    memset(sum_b,0,BOX_NUMBER*BOX_NUMBER*BOX_NUMBER*sizeof(double));
    
//     mexPrintf("Init\n");
    
    for(i=0; i<num_pixels; ++i) 
    {
        r = image_r[i]; 
        g = image_g[i]; 
        b = image_b[i];
        r8 = (int)(r*255);
        g8 = (int)(g*255);
        b8 = (int)(b*255);
        inr=(r8>>SHIFT_DIV); 
        ing=(g8>>SHIFT_DIV); 
        inb=(b8>>SHIFT_DIV); 
        ind=(inr<<SHIFT_IND_R1)+(ing<<SHIFT_IND_G1)+inb;
        count[ind] += 1;
        sum_r[ind] += r;
        sum_g[ind] += g;
        sum_b[ind] += b;
    }
    
//     mexPrintf("%d\n", num_pixels);

    for(inr=0; inr<BOX_NUMBER; ++inr)
    {
        for(ing=0; ing<BOX_NUMBER; ++ing)
        {
            for(inb=0; inb<BOX_NUMBER; ++inb)
            {
                ind=(inr<<SHIFT_IND_R1)+(ing<<SHIFT_IND_G1)+inb;
                if (count[ind]>0)
                {
                    map_r[ind] = sum_r[ind]/count[ind];
                    map_g[ind] = sum_g[ind]/count[ind];
                    map_b[ind] = sum_b[ind]/count[ind];
                }
            }
        }
    }
//     mexPrintf("Done!\n");
}