#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

//#pragma comment(lib,"cutil64D.lib")
//#pragma comment(lib,"cutil64.lib")

# define N 16

typedef struct
{
  float   real;
  float   img;
}complex;

 //申请纹理
texture<float,2,cudaReadModeElementType>texRef1;
texture<float,2,cudaReadModeElementType>texRef2;
texture<unsigned,2,cudaReadModeElementType>texRef3;

 //计算旋转因子
__global__ void initW(complex* W,int size_x)
{   
		float PI=atan((float)1)*4;
		int i = blockIdx.x*blockDim.x+threadIdx.x; 
		if(i<size_x/2)
		{
			W[i].real=cos(2*PI/size_x*i);   
			W[i].img=-1.0*sin(2*PI/size_x*i);
		}  
} 
 

//计算旋转因子的数组
__global__ void initW_array(complex* W,float* W_array_real,float* W_array_img,int size_x)                     
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	long long j = blockIdx.y;  //级数
	int l;  
	l = exp2f(j);
	//l=1<<j;
	if(i<size_x/2&&j<log((float)size_x)/log((float)2))
	{W_array_real[j*size_x/2+i] = W[size_x*(i%l)/2/l].real;
	 W_array_img[j*size_x/2+i] = W[size_x*(i%l)/2/l].img;}
//	__syncthreads();
}

 
 //复数乘
__device__ complex ComplexMul(complex X_in,complex W_in)                    
{
	complex X_out;
	X_out.real = X_in.real*W_in.real-X_in.img*W_in.img;
	X_out.img = X_in.real*W_in.img+X_in.img*W_in.real;
	return X_out;
}


//复数加
__device__ complex ComplexAdd(complex X1,complex X2)                           
{
	complex X_out;
	X_out.real = X1.real+X2.real;
	X_out.img = X1.img+X2.img;
	return X_out;
}


//复数减
__device__ complex ComplexSub(complex X1,complex X2)                      
{
	complex X_out;
	X_out.real = X1.real-X2.real;
	X_out.img = X1.img-X2.img;
	return X_out;
}


__global__ void FFT_T(complex* DataIn,int size_x,int Ns,int stage) 
{ 
	//线程在block 中的位置
	int k = blockIdx.x*blockDim.x+threadIdx.x;                
    int width = size_x/(2*N);
	int p,q,t;
	complex Wn,Xp,XqWn;

	    //按级进行并行的蝶形运算
		if( k<size_x/2)                                             
		{
				p = k / Ns * Ns * 2 + k % Ns;
				q = p + Ns;

				t = (k/width)+stage;

				Wn.real = tex2D( texRef1,k%width,t );
				Wn.img = tex2D( texRef2,k%width,t );
				
				XqWn = ComplexMul( DataIn[q],Wn);
				Xp = DataIn[p];
				DataIn[p] = ComplexAdd( Xp,XqWn);
				DataIn[q] = ComplexSub( Xp,XqWn) ;
		} //end if
} //end kernel


__global__ void FFT_T1(complex* DataIn,int size_x) 
{ 
	//线程在block 中的位置
	int i = threadIdx.x;
	__shared__ complex sdata[1024];
	int j = blockIdx.x*blockDim.x;
	int k ;
	k = j + i;

    int width = size_x/(2*N);
	int p,q,t;
	int stage = 0;
	complex Wn,Xp,XqWn;
		
	    //按级进行并行的蝶形运算
		if( k<size_x/2)                                             
		{
			sdata[i] = DataIn[i+j*2];
			sdata[i+512] = DataIn[i+j*2+512];
			__syncthreads();  
			
			for(int Ns = 1;Ns < 1024;Ns = Ns * 2)
			{
				p = i / Ns * Ns * 2 + i % Ns;
				q = p + Ns;

				t = (k/width)+stage;

				Wn.real = tex2D( texRef1,k%width,t );
				Wn.img = tex2D( texRef2,k%width,t );

				stage = stage + N;
				
				XqWn = ComplexMul( sdata[q],Wn);
			
				Xp = sdata[p];
				sdata[p] = ComplexAdd( Xp,XqWn);
				sdata[q] = ComplexSub( Xp,XqWn) ;
				__syncthreads();  
			}
			DataIn[p+j*2] = sdata[p];
			DataIn[q+j*2] = sdata[q];
		} //end if
} //end kernel

//倒位序
__global__ void change(unsigned *trans,int size_x)                                                                   
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int j=0,k=0;
	unsigned int   t;

	if(i<size_x)
	{
		k=i;
		j=0;   
		t=(log((float)size_x)/log((float)2))+0.5;   //级数

		while(   (t--)>0   )   
		{   
			j=j<<1;  //最后一位 左移一位
			j|=(k   &   1);  //每次取二进制数的最后一位  其余取零
			k=k>>1;  //倒数第二位 右移一位
		} 
		trans[i] = j;
	}//end if

}//end kernal


__global__ void change1(complex* d_idata,complex* d_idata1,int size_x)     
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned j;

	if(i<size_x)
	{
		j = tex2D( texRef3,i%65536,i/65536 );
		d_idata[j]=d_idata1[i];
	}
	
}


int main()
{
	int size_x = 16;
	int size_x1;

	complex* h_idata;
	complex* h_odata;

	complex* d_idata;
	complex* d_idata1;
	unsigned* trans;
	complex* W;
	float *W_array_real;
	float *W_array_img;

	int i=0;
	int length,t;

	t=(log((double)size_x)/log((double)2))+0.5;   

	float gpu_time = 0;

	length = size_x/1025+1;

	//cuda数组高度   精度问题
	int height = (log((double)size_x)/log((double)2))+0.5;                               

	//cuda数组宽度
	int width = size_x/2;                                                        
	int size = width * height * sizeof(float);
                                     
	cudaMallocHost((void**)&h_odata,size_x*sizeof(complex));
	cudaMallocHost((void**)&h_idata,size_x*sizeof(complex));

	cudaMalloc((void**)&W,size_x/2*sizeof(complex));
	cudaMalloc((void**)&d_idata,size_x*sizeof(complex));
	cudaMalloc((void**)&d_idata1,size_x*sizeof(complex));

	cudaMalloc((void**)&trans,size_x*sizeof(unsigned));

	cudaMalloc((void**)&W_array_real,size);
	cudaMalloc((void**)&W_array_img,size);

	//CUDA数组类型
	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);   
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);

	//申请cuda数组
	cudaArray* d_Wdata_real;                                                                         
	cudaArray* d_Wdata_img;
	cudaArray* trans1;

	cudaMallocArray(&d_Wdata_real,&channelDesc1,width/N,height*N);
	cudaMallocArray(&d_Wdata_img,&channelDesc2,width/N,height*N);  
	cudaMallocArray(&trans1,&channelDesc3,65536,size_x/65537+1);  

	dim3 block(length,1,1);
	dim3 thread(512,1,1);
	initW<<<block,thread>>>(W,size_x);

	dim3 block1(length,height,1);
	dim3 thread1(512,1,1);
	initW_array<<<block1,thread1>>>(W,W_array_real,W_array_img,size_x);

	dim3 blocks3(length*32,1,1);
	dim3 threads3(32,1,1);
	change<<<blocks3,threads3>>>(trans,size_x);

	cudaMemcpyToArray(d_Wdata_real,0,0,W_array_real,size,cudaMemcpyDeviceToDevice);                 
	cudaMemcpyToArray(d_Wdata_img,0,0,W_array_img,size,cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(trans1,0,0,trans,size_x*sizeof(unsigned),cudaMemcpyDeviceToDevice);
	
	//cuda数组和纹理绑定
	cudaBindTextureToArray(texRef1,d_Wdata_real,channelDesc1);                       
	cudaBindTextureToArray(texRef2,d_Wdata_img,channelDesc2);
	cudaBindTextureToArray(texRef3,trans1,channelDesc3);

	 //需要FFT的数
	for(i=0;i<size_x;i++)                                                        
	{
		h_odata[i].real=i+1.0f;
		h_odata[i].img=0.0f;
	}

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMemcpy(d_idata1,h_odata,size_x*sizeof(complex),cudaMemcpyHostToDevice);

	dim3 blocks1(length*32,1,1);
	dim3 threads1(32,1,1);
	change1<<<blocks1,threads1>>>(d_idata,d_idata1,size_x);
cudaEventRecord(start,0);
	//size_x1 = 1024;
	dim3 blocks4(length,1,1);
	dim3 threads4(512,1,1);
	FFT_T1<<<blocks4,threads4>>>(d_idata,size_x);


	dim3 blocks2(length,1,1);
	dim3 threads2(512,1,1);
	//Ns是级数
	for ( int Ns = 1024,stage = N*10; Ns<size_x; Ns = Ns * 2,stage+=N)                           
	{
		FFT_T<<<blocks2,threads2>>>(d_idata,size_x,Ns,stage);
		//cudaThreadSynchronize();
	}
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);	
	cudaMemcpy(h_idata,d_idata,size_x*sizeof(complex),cudaMemcpyDeviceToHost);


	cudaEventElapsedTime(&gpu_time,start,stop);

/*
	FILE *fp = fopen("D:/cuda3.txt","w");
	if(fp==0)
	exit(0);

	 for(i = 0;i<size_x;i++)
	 {
		fprintf(fp,"%f+%f*i\n",h_idata[i].real,h_idata[i].img);
  	 }
	fclose(fp);
*/


	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	 //释放内存 显存cuda数组
	cudaUnbindTexture(texRef1);                                               
	cudaUnbindTexture(texRef2);
	cudaUnbindTexture(texRef3);

	cudaFreeHost(h_odata);
	cudaFreeHost(h_idata);

	cudaFree(W);
	cudaFree(d_idata);
	cudaFree(d_idata1);
	cudaFree(trans);

	cudaFree(W_array_real);
	cudaFree(W_array_img);

	cudaFreeArray(d_Wdata_real);
	cudaFreeArray(d_Wdata_img);
	cudaFreeArray(trans1);

	//释放线程
	cudaThreadExit();                                                           

	printf("%f\n",gpu_time);
	printf("OK\n");
	getchar();

	return 0;
}

