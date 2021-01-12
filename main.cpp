#include <cuda_runtime.h>
#include <stdio.h>
#include "cusparse.h"
#include "cublas_v2.h"
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#define CLEANUP(s){\
	printf("%s\n", s);\
	if(h_valA) free(h_valA);\
	if(h_csrRowPtrA) free(h_csrRowPtrA);\
	if(h_csrColIndA) free(h_csrColIndA);\
	if(valA) cudaFree(valA);\
	if(csrRowPtrA) cudaFree(csrRowPtrA);\
	if(csrColIndA) cudaFree(csrColIndA);\
	if(h_b) free(h_b);\
	if(b) cudaFree(b);\
	if(r) cudaFree(r);\
	if(p) cudaFree(p);\
	if(Ap) cudaFree(Ap);\
	if(h_x) free(h_x);\
	if(x) cudaFree(x);\
	if(handle1) cublasDestroy(handle1);\
	if(descrA) cusparseDestroyMatDescr(descrA);\
	if(handle2) cusparseDestroy(handle2);\
	cudaDeviceReset();\
}

//用于统计程序时间
double cpuSecond(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(){
	//initialization
	cudaError_t cudaStat1, cudaStat2, cudaStat3;	

	double* h_valA = 0;
	int* h_csrRowPtrA = 0;
	int* h_csrColIndA = 0;
	double* valA = 0;
	int* csrRowPtrA = 0;
	int* csrColIndA = 0;
	double* h_b = 0;
	double* b = 0;
	double* r = 0;
	double* p = 0;
	double* Ap = 0;
	double* h_x = 0;
	double* x = 0;

	cublasStatus_t status1;
	cublasHandle_t handle1 = 0;
	cusparseStatus_t status2;
	cusparseHandle_t handle2 = 0;
	cusparseMatDescr_t descrA = 0;

	int n, nnz;
	double dzero = 0.0;
	double done = 1.0;
	/*
	use CG to solve following linear system:
		Ax = b;
		A = |3 0 2|
			|0 2 0|
			|2 0 1|
		b = |3.5|
			|1.5|
			|2.0|
	correct result x:
		x = |0.50|
			|0.75|
			|1.00|
	   */
	n = 3;
	nnz = 5;
	//which device(gpu) to use
	cudaStat1 = cudaSetDevice(0);
	if(cudaStat1 != cudaSuccess){
		CLEANUP("Device Set failed");
		return EXIT_FAILURE;
	}

	//matrix A(csr format) in host
	h_valA = (double*)malloc(nnz*sizeof(double));
	h_csrRowPtrA = (int*)malloc((n+1)*sizeof(int));
	h_csrColIndA = (int*)malloc(nnz*sizeof(int));
	if(!h_valA||!h_csrRowPtrA||!h_csrColIndA){
		CLEANUP("Host malloc failed(A)");
		return EXIT_FAILURE;
	}

	h_valA[0] = 3.0;
	h_valA[1] = 2.0;
	h_valA[2] = 2.0;
	h_valA[3] = 2.0;
	h_valA[4] = 1.0;
	
	h_csrRowPtrA[0] = 0;
	h_csrRowPtrA[1] = 2;
	h_csrRowPtrA[2] = 3;
	h_csrRowPtrA[3] = 5;
	
	h_csrColIndA[0] = 0;
	h_csrColIndA[1] = 2;
	h_csrColIndA[2] = 1;
	h_csrColIndA[3] = 0;
	h_csrColIndA[4] = 2;

	//matrix A(csr format) in device
	cudaStat1 = cudaMalloc((void**)&valA, nnz*sizeof(double));
	cudaStat2 = cudaMalloc((void**)&csrRowPtrA, (n+1)*sizeof(int));
	cudaStat3 = cudaMalloc((void**)&csrColIndA, nnz*sizeof(int));
	if(cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess){
		CLEANUP("device malloc failed(A)");
		return EXIT_FAILURE;
	}
	cudaStat1 = cudaMemcpy(valA, h_valA, (size_t)(nnz*sizeof(double)), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (size_t)((n+1)*sizeof(int)), cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(csrColIndA, h_csrColIndA, (size_t)(nnz*sizeof(int)), cudaMemcpyHostToDevice);
	if(cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess){
		CLEANUP("memcpy from host to device failed(A)");
		return EXIT_FAILURE;
	}

	//vector b in host
	h_b = (double*)malloc(n*sizeof(double));
	h_b[0] = 3.5;
	h_b[1] = 1.5;
	h_b[2] = 2.0;

	//vector b in device
	cudaStat1 = cudaMalloc((void**)&b, n*sizeof(double));
	if(cudaStat1 != cudaSuccess){
		CLEANUP("device malloc failed(b)");
		return EXIT_FAILURE;
	}
	cudaStat1 = cudaMemcpy(b, h_b, (size_t)(n*sizeof(double)), cudaMemcpyHostToDevice);
	if(cudaStat1 != cudaSuccess){
		CLEANUP("memcpy from host to device failed(b)");
		return EXIT_FAILURE;
	}

	//vector r in device
	cudaStat1 = cudaMalloc((void**)&r, n*sizeof(double));
	if(cudaStat1 != cudaSuccess){
		CLEANUP("device malloc failed(r)");
		return EXIT_FAILURE;
	}

	//vector p in device
	cudaStat1 = cudaMalloc((void**)&p, n*sizeof(double));
	if(cudaStat1 != cudaSuccess){
		CLEANUP("device malloc failed(p)");
		return EXIT_FAILURE;
	}

	//vector Ap in device
	cudaStat1 = cudaMalloc((void**)&Ap, n*sizeof(double));
	if(cudaStat1 != cudaSuccess){
		CLEANUP("device malloc failed(Ap)");
		return EXIT_FAILURE;
	}

	//vector x in host
	h_x = (double*)malloc(n*sizeof(double));
	for(int i=0; i<n; ++i)
		h_x[i] = 0.0;

	//vector x in device
	cudaStat1 = cudaMalloc((void**)&x, n*sizeof(double));
	if(cudaStat1 != cudaSuccess){
		CLEANUP("device malloc faile(x)");
		return EXIT_FAILURE;
	}
	cudaStat1 = cudaMemcpy(x, h_x, (size_t)(n*sizeof(double)), cudaMemcpyHostToDevice);
	if(cudaStat1 != cudaSuccess){
		CLEANUP("memcpy from host to device failed(x)");
		return EXIT_FAILURE;
	}

	//initialize cublas library
	status1 = cublasCreate(&handle1);
	if(status1 != CUBLAS_STATUS_SUCCESS){
		CLEANUP("CUBLAS initialize failed");
		return EXIT_FAILURE;
	}
	//initialize cusparse library
	status2 = cusparseCreate(&handle2);
	if(status2 != CUSPARSE_STATUS_SUCCESS){
		CLEANUP("CUSPARSE Library initialize failed");
		return EXIT_FAILURE;
	}
	//initialize matrix descriptor
	status2 = cusparseCreateMatDescr(&descrA);
	if(status2 != CUSPARSE_STATUS_SUCCESS){
		CLEANUP("Matrix descriptor initialize failed");
		return EXIT_FAILURE;
	}
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);


	//CG
	//1: 
	//max iteration number = 2000
	//relative residual = 1e-7
	//compute initial residual r = b - Ax_0
	//x_0 = |0|
	//		|0|
	//		|0|
	//initial p = r
	int maxit = 2000;
	double tol = 1e-7;
	double alpha, beta, rhop, rho;
	//r = b
	status1 = cublasDcopy(handle1, n, b, 1, r, 1);
	if(status1 != CUBLAS_STATUS_SUCCESS){
		CLEANUP("Vector copy failed(r)");
		return EXIT_FAILURE;
	}
	//p = b
	status1 = cublasDcopy(handle1, n, b, 1, p, 1);
	if(status1 != CUBLAS_STATUS_SUCCESS){
		CLEANUP("Vector copy failed(p)");
		return EXIT_FAILURE;
	}
	status1 = cublasDnrm2(handle1, n, r, 1, &rho);
	if(status1 != CUBLAS_STATUS_SUCCESS){
		CLEANUP("compute norm2 failed(r)");
		return EXIT_FAILURE;
	}
	rho = rho*rho;

	//2: repeat until convergence
	for(int i=0; i<maxit; ++i){
		status2 = cusparseDcsrmv(handle2, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &done, 
									descrA, valA, csrRowPtrA, csrColIndA, p, &dzero, Ap);
		if(status2 != CUSPARSE_STATUS_SUCCESS){
			CLEANUP("compute mv failed(Ap)");
			return EXIT_FAILURE;
		}
		double tmp;
		status1 = cublasDdot(handle1, n, p, 1, Ap, 1, &tmp);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute dot failed(tmp)");
			return EXIT_FAILURE;
		}
		//alpha = (r^T*r)/(p^T*A*p)
		alpha = rho/tmp;
		//x = x + alpha*p
		status1 = cublasDaxpy(handle1, n, &alpha, p, 1, x, 1);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute axpy failed(x)");
			return EXIT_FAILURE;
		}
		double tmp2 = -alpha;
		//r = r - alpha*A*p
		status1 = cublasDaxpy(handle1, n, &tmp2, Ap, 1, r, 1);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute axpy failed(r)");
			return EXIT_FAILURE;
		}

		rhop = rho;
		status1 = cublasDnrm2(handle1, n, r, 1, &rho);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute norm2 failed(r)");
			return EXIT_FAILURE;
		}
		if(rho < tol){
			break;
		}
		rho = rho*rho;
		
		//beta = rho/rhop
		beta = rho/rhop;
		
		//p = r + beta*p
		status1 = cublasDscal(handle1, n, &beta, p, 1);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute scal failed(p)");
			return EXIT_FAILURE;
		}
		status1 = cublasDaxpy(handle1, n, &done, r, 1, p, 1);
		if(status1 != CUBLAS_STATUS_SUCCESS){
			CLEANUP("compute axpy failed(p)");
			return EXIT_FAILURE;
		}
	}
	
	//copy x to h_x, and output h_x
	cudaStat1 = cudaMemcpy(h_x, x, (size_t)(n*sizeof(double)), cudaMemcpyDeviceToHost);
	if(cudaStat1 != cudaSuccess){
		CLEANUP("memcpy from device to host failed(x)");
		return EXIT_FAILURE;
	}
	for(int i=0; i<n; ++i)
		printf("%f\n", h_x[i]);

	CLEANUP("Success!");
	return 0;
}
