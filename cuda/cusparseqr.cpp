//#include <THC/THC.h>
#include <torch/extension.h>

// extern THCState *state;

// CUDA forward declarations
int solve_cuda(int nnz, int m, double tol, double *dcooVal, int *dcooColInd,
               int *dcooRowInd, int *dcsrRowPtr, double *db, double *dx);

// C++ forward declarations

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

int solve(int nnz, int m, double tol, torch::Tensor dcooVal_tensor,
          torch::Tensor dcooColInd_tensor, torch::Tensor dcooRowInd_tensor,
          torch::Tensor dcsrRowPtr_tensor, torch::Tensor db_tensor,
          torch::Tensor dx_tensor) {

  double *dcooVal = dcooVal_tensor.data_ptr<double>();
  int *dcooColInd = dcooColInd_tensor.data_ptr<int>();
  int *dcooRowInd = dcooRowInd_tensor.data_ptr<int>();
  int *dcsrRowPtr = dcsrRowPtr_tensor.data_ptr<int>();
  double *db = db_tensor.data_ptr<double>();
  double *dx = dx_tensor.data_ptr<double>();

  int singularity = solve_cuda(nnz, m, tol, dcooVal, dcooColInd, dcooRowInd,
                               dcsrRowPtr, db, dx);
  return singularity;
  // CHECK_INPUT(input);
  // TORCH_CHECK(num_bits > 0, "sanity check");
  // return axes_to_transpose_cuda(input, num_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve", &solve, "QR Decomposition");
}
