
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

typedef struct
{
  float   real;
  float   img;
}complex;

texture<float,2,cudaReadModeElementType>texRef1;                               //��������
texture<float,2,cudaReadModeElementType>texRef2;

	


__global__ void initW(complex* W,int size_x)                                                                     //������ת����
{   
		float PI=atan((float)1)*4;
		int i = blockIdx.x*blockDim.x+threadIdx.x; 
		if(i<size_x/2)
		{
			W[i].real=cos(2*PI/size_x*i);   
			W[i].img=-1.0*sin(2*PI/size_x*i);
		}  
//		__syncthreads();
} 


__global__ void initW_array(complex* W,float* W_array_real,float* W_array_img,int size_x)                      //������ת���ӵ�����
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	long long j = blockIdx.y;                                                                              //����
	int l;  
	l = exp2f(j);
	//l=1<<j;
	if(i<size_x/2&&j<log((float)size_x)/log((float)2))
	{W_array_real[j*size_x/2+i] = W[size_x*(i%l)/2/l].real;
	 W_array_img[j*size_x/2+i] = W[size_x*(i%l)/2/l].img;}
//	__syncthreads();
}

 

__device__ complex ComplexMul(complex X_in,complex W_in)                     //������
{
	complex X_out;
	X_out.real = X_in.real*W_in.real-X_in.img*W_in.img;
	X_out.img = X_in.real*W_in.img+X_in.img*W_in.real;
	return X_out;
}

__device__ complex ComplexAdd(complex X1,complex X2)                           //������
{
	complex X_out;
	X_out.real = X1.real+X2.real;
	X_out.img = X1.img+X2.img;
	return X_out;
}


__device__ complex ComplexSub(complex X1,complex X2)                      //������
{
	complex X_out;
	X_out.real = X1.real-X2.real;
	X_out.img = X1.img-X2.img;
	return X_out;
}

__global__ void FFT_T(complex* DataIn,int size_x,complex* W,int Ns,int stage) 
{ 
	int k = blockIdx.x*blockDim.x+threadIdx.x;                 //�߳���block �е�λ��
   
			int p,q;
//			long long t1,t;
			complex Wn,Xp,XqWn;

		if( k<size_x/2)                                              //�������в��еĵ�������
		{
				p = k / Ns * Ns * 2 + k % Ns;
				q = p + Ns;
				Wn.real = tex2D( texRef1,k,stage );
				Wn.img = tex2D( texRef2,k,stage );
				XqWn = ComplexMul( DataIn[q],Wn);
				Xp = DataIn[p];
				DataIn[p] = ComplexAdd( Xp,XqWn);
				DataIn[q] = ComplexSub( Xp,XqWn) ;

		} //end if
} //end kernel



/*void   change(complex* h_odata)                                                                    //��λ��
  {   
  complex   temp;   
  unsigned   int   i=0,j=0,k=0;   
  unsigned int t1;
  for(i=0;i<size_x;i++)   
  {   
	  k=i;
	  j=0;   
	  t1=(log((double)size_x)/log((double)2))+0.5;                                                            //����  �������루+0.5��
//	  t1=t;
	  while(   (t1--)>0   )   
	  {   
		  j=j<<1;                                                                            //����һλ
		  j|=(k   &   1);                                                                   //ÿ��ȡ�������������һλ  ����ȡ��
		  k=k>>1;                                                                           //����һλ
	 }   
	  if(j>i)   
	  {   
		  temp=h_odata[i];   
		  h_odata[i]=h_odata[j];   
		  h_odata[j]=temp;   
	  }   
  }   
  }   

*/




__global__ void change(complex* d_idata,int size_x)                                                                    //��λ��
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
//	int n = blockIdx.y*blockDim.y+threadIdx.y;
//	int i = n * gridDim.x *blockDim.x+ m;
	complex   temp;
	unsigned int j=0,k=0;
	unsigned int   t;
if(i<size_x)
{
	  k=i;
	  j=0;   
	  t=(log((float)size_x)/log((float)2))+0.5;                                                            //����
	  while(   (t--)>0   )   
	  {   
		  j=j<<1;                                                                            //���һλ ����һλ
		  j|=(k   &   1);                                                                   //ÿ��ȡ�������������һλ  ����ȡ��
		  k=k>>1;                                                                           //�����ڶ�λ ����һλ
	  }   
	  	  if(j>i)   
	  {   
		  temp=d_idata[i];   
		  d_idata[i]=d_idata[j];   
		  d_idata[j]=temp;   
	  }   

}//end if

}//end kernal











int main()
{
	int size_x = 256*512;
//	printf("please input the size of the data:");
//	scanf("%d",&size_x);




/*
	t = log((double)size_x)/log((double)2);                        //2��������
	 
	if(t-(int)t!=0)
	{
		t = int(t) + 1;
		size_x =(int)pow(2,t);
	}
*/

	complex* d_idata;
	complex* W;

	complex* h_odata;
	complex* h_idata;

	float *W_array_real;
	float *W_array_img;

	int i=0,j=0;
	int length;
	length = size_x/1025+1;
		 float gpu_time[100]={0};
/*
	   FILE *fp1 = fopen("D:/gputime1.txt","w");
	   if(fp1==0)
	 	  exit(0);
*/
//for(size_x = 2,j = 0;size_x <= 131072;size_x = size_x * 2,j++)
//{	

	int height = (log((double)size_x)/log((double)2))+0.5;                               //cuda����߶�   ��������
//	int height;
//	height = t;
	int width = size_x/2;                                                        //cuda������
	int size = width * height * sizeof(float);

	//complex* h_idata = (complex*)malloc(size_x*sizeof(complex));                //��device�ﾭ��FFT֮�������
	//complex* h_odata = (complex*)malloc(size_x*sizeof(complex));                                         //��������ҪFFT����
	cudaMallocHost((void**)&h_odata,size_x*sizeof(complex));
	cudaMallocHost((void**)&h_idata,size_x*sizeof(complex));


	cudaMalloc((void**)&W,size_x/2*sizeof(complex));
	cudaMalloc((void**)&d_idata,size_x*sizeof(complex));

	cudaMalloc((void**)&W_array_real,size);
	cudaMalloc((void**)&W_array_img,size);

	cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);   //CUDA��������
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
	cudaArray* d_Wdata_real;                                                                          //����cuda����
	cudaArray* d_Wdata_img;
	cudaMallocArray(&d_Wdata_real,&channelDesc1,width,height);
	cudaMallocArray(&d_Wdata_img,&channelDesc2,width,height);

	

	dim3 block(length,1,1);
	dim3 thread(512,1,1);
	initW<<<block,thread>>>(W,size_x);
	cudaThreadSynchronize();

	dim3 block1(length,height,1);
	dim3 thread1(512,1,1);
	initW_array<<<block1,thread1>>>(W,W_array_real,W_array_img,size_x);
	cudaThreadSynchronize();

	cudaMemcpyToArray(d_Wdata_real,0,0,W_array_real,size,cudaMemcpyDeviceToDevice);                 
	cudaMemcpyToArray(d_Wdata_img,0,0,W_array_img,size,cudaMemcpyDeviceToDevice);
/*
	texRef1.addressMode[0]=cudaAddressModeWrap;
	texRef1.addressMode[1]=cudaAddressModeWrap;
	texRef1.filterMode=cudaFilterModeLinear;
	texRef1.normalized=true;


	texRef2.addressMode[0]=cudaAddressModeWrap;
	texRef2.addressMode[1]=cudaAddressModeWrap;
	texRef2.filterMode=cudaFilterModeLinear;
	texRef2.normalized=true;
*/

	cudaBindTextureToArray(texRef1,d_Wdata_real,channelDesc1);                       //cuda����������
	cudaBindTextureToArray(texRef2,d_Wdata_img,channelDesc2);

	
	for(i=0;i<size_x;i++)                                                         //��ҪFFT����
	{
		h_odata[i].real=i+1.0f;
		h_odata[i].img=0.0f;
	}

	clock_t start = clock();                                          //����gpu����ʱ��   ��ʼ

for(int m = 0;m < 100;m++){

	cudaMemcpy(d_idata,h_odata,size_x*sizeof(complex),cudaMemcpyHostToDevice);
	
	dim3 blocks3(length*2,1,1);
	dim3 threads3(512,1,1);
	change<<<blocks3,threads3>>>(d_idata,size_x);                  //��λ��
	cudaThreadSynchronize();


	dim3 blocks2(length,1,1);
	dim3 threads2(512,1,1);
	for ( int Ns = 1,stage = 0; Ns<size_x; Ns = Ns * 2,stage++)                           //Ns�Ǽ���
	{
		FFT_T<<<blocks2,threads2>>>(d_idata,size_x,W,Ns,stage);
		cudaThreadSynchronize();
	}
	cudaMemcpy(h_idata,d_idata,size_x*sizeof(complex),cudaMemcpyDeviceToHost);
}
	gpu_time[j] = clock() - start;                                          //����gpu����ʱ��  ����

/*	FILE *fp = fopen("D:/cuda3.txt","w");

		if(fp==0)
		  exit(0);
	  for(i = 0;i<size_x;i++)
	  {
		fprintf(fp,"%f+%f*i\n",h_idata[i].real,h_idata[i].img);
  	  }
	fclose(fp);
*/
//	fprintf(fp1," %f\n",gpu_time[j]/1000.0f);

	cudaUnbindTexture(texRef1);                                               //�Ӵ�cuda���������İ�
	cudaUnbindTexture(texRef2);

	//free(h_idata);                                                               //�ͷ��ڴ� �Դ�cuda����
	//free(h_odata);
	cudaFreeHost(h_odata);
	cudaFreeHost(h_idata);

	cudaFree(W);
	cudaFree(W_array_real);
	cudaFree(W_array_img);
	cudaFreeArray(d_Wdata_real);
	cudaFreeArray(d_Wdata_img);
	cudaFree(d_idata);

	cudaThreadExit();                                                           //�ͷ��߳�
//	cudaThreadExit();
//	cudaThreadExit();
//}
//	fclose(fp1);
	printf("%f\n",gpu_time[0]);
	printf("OK\n");
	getchar();
	return 0;
}

