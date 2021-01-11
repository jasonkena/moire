#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cusolverSp.h>
//https://forums.developer.nvidia.com/t/cusolver-sparse-cusolverspdcsrlsvqr-error/38214
//https://stackoverflow.com/questions/31840341/solving-general-sparse-linear-systems-in-cuda

#define sparseErrchk(ans) { sparseAssert((ans), __FILE__, __LINE__); }
void sparseAssert(cusparseStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cusparseGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}


const char* cusolverGetErrorString(cusolverStatus_t error);
#define solverErrchk(ans) { solverAssert((ans), __FILE__, __LINE__); }
void solverAssert(cusolverStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cusolverGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

const char* cusolverGetErrorString(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}


int solve_cuda(int nnz, int m, double tol, double *dcooVal, int *dcooColInd, int *dcooRowInd, int *dcsrRowPtr, double *db, double *dx) {
  // --- create library handles:
  cusolverSpHandle_t cusolver_handle;
  solverErrchk(cusolverSpCreate(&cusolver_handle));

  cusparseHandle_t cusparse_handle;
  sparseErrchk(cusparseCreate(&cusparse_handle));

  // --- prepare solving and copy to GPU:
  int reorder = 0;
  int singularity = 0;

  // create matrix descriptor
  cusparseMatDescr_t descrA;
  sparseErrchk(cusparseCreateMatDescr(&descrA));
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  cudaDeviceSynchronize();
  // convert COO to CSR
  sparseErrchk(cusparseXcoo2csr(cusparse_handle,
                 dcooRowInd,
                 nnz,
                 m,
                 dcsrRowPtr,
                 CUSPARSE_INDEX_BASE_ZERO));

  cudaDeviceSynchronize();

  // solve the system
  solverErrchk(cusolverSpDcsrlsvqr(cusolver_handle, m, nnz, descrA, dcooVal,
                                dcsrRowPtr, dcooColInd, db, tol, reorder, dx,
                                &singularity));

  cudaDeviceSynchronize();

  sparseErrchk(cusparseDestroy(cusparse_handle));
  solverErrchk(cusolverSpDestroy(cusolver_handle));

  return singularity;
}


// write info about memory requirements of the qr-decomposition to stdout
void get_memInfo(int nnz, int m, double tol, double *csrVal, int *csrColInd,
                 int *csrRowPtr, double *b, double *x) {

    int* dCol, *dRow;
    double* dVal;
    cudaError_t error;

    //allocate device memory, copy H2D
    cudaMalloc((void**)&dCol, sizeof(int)*nnz);
    cudaMalloc((void**)&dRow, sizeof(int)*(m+1));
    cudaMalloc((void**)&dVal, sizeof(double)*nnz);
    cudaMemcpy(dCol, csrColInd, sizeof(int)*nnz,    cudaMemcpyHostToDevice);
    cudaMemcpy(dRow, csrRowPtr, sizeof(int)*(m+1),  cudaMemcpyHostToDevice);
    cudaMemcpy(dVal, csrVal, sizeof(double)*nnz,  cudaMemcpyHostToDevice);

    error = cudaGetLastError();
    std::cout << "Error status after cudaMemcpy in getmemInfo: " << error << std::endl;

    //create and initialize library handles
    cusolverSpHandle_t cusolver_handle;
    cusparseHandle_t cusparse_handle;
    cusolverStatus_t cusolver_status;
    cusparseStatus_t cusparse_status;
    cusparse_status = cusparseCreate(&cusparse_handle);
    std::cout << "status cusparseCreate: " << cusparse_status << std::endl;
    cusolver_status = cusolverSpCreate(&cusolver_handle);
    std::cout << "status cusolverSpCreate: " << cusolver_status << std::endl;

    //create CsrqrInfo
    csrqrInfo_t info;
    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    std::cout << "status cusolverSpCrateCsrqrInfo: " << cusolver_status << std::endl;

    //create mat descriptor
    cusparseMatDescr_t descrA;
    cusparse_status = cusparseCreateMatDescr(&descrA);
    cusparse_status = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;
    cudaDeviceSynchronize();
    //call SpDcsrqrAnalysisBatched.
    cusolver_status = cusolverSpXcsrqrAnalysisBatched(cusolver_handle,
                                                      m,
                                                      m,
                                                      nnz,
                                                      descrA,
                                                      dRow,
                                                      dCol,
                                                      info);
    std::cout << "status cusolverSpDcsrqrAnalysisBatched: " << cusolver_status << std::endl;

    //get the buffer size via BufferInfoBatched
    int batchsize = 1;
    size_t internalDataInBytes = 99;
    size_t workspaceInBytes = 99;
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(cusolver_handle,
                                                        m,
                                                        m,
                                                        nnz,
                                                        descrA,
                                                        dVal,
                                                        dRow,
                                                        dCol,
                                                        batchsize,
                                                        info,
                                                        &internalDataInBytes,
                                                        &workspaceInBytes);

    std::cout << "status cusolverSpDcsrqrBufferInfoBatched: " << cusolver_status << std::endl;
    std::cout << "internalbuffer(Bytes): " << internalDataInBytes << std::endl;
    std::cout << "workspace(Bytes): " << workspaceInBytes << std::endl;

    //destroy stuff
    cusolver_status = cusolverSpDestroyCsrqrInfo(info);
    std::cout << "status cusolverSpDestroyCsrqrInfo: " << cusolver_status << std::endl;

    cusparse_status = cusparseDestroy(cusparse_handle);
    std::cout << "status cusparseDestroy: " << cusparse_status << std::endl;
    cusolver_status = cusolverSpDestroy(cusolver_handle);
    std::cout << "status cusolverSpDestroy: " << cusolver_status << std::endl;
    cudaFree(dCol);
    cudaFree(dRow);
    cudaFree(dVal);
}
