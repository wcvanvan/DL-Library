#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle
{
  namespace cuda
  {

#define BASE_THREAD_NUM 256

#define TILE 4
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct CudaArray
    {
      CudaArray(const size_t size)
      {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
      }
      ~CudaArray() { cudaFree(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }

      scalar_t *ptr;
      size_t size;
    };

    struct CudaDims
    {
      dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size)
    {
      /**
       * Utility function to get cuda dimensions for 1D call
       */
      CudaDims dim;
      size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim.block = dim3(BASE_THREAD_NUM, 1, 1);
      dim.grid = dim3(num_blocks, 1, 1);
      return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec
    {
      uint32_t size;
      int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x)
    {
      CudaVec shape;
      if (x.size() > MAX_VEC_SIZE)
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
      shape.size = x.size();
      for (size_t i = 0; i < x.size(); i++)
      {
        shape.data[i] = x[i];
      }
      return shape;
    }

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val)
    {
      CudaDims dim = CudaOneDim(out->size);
      FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN YOUR SOLUTION
      if (gid < size)
      {
        size_t gid_a = 0;
        size_t tmp = gid;
        for (int i = shape.size - 1; i >= 0; i--)
        {
          size_t idx = tmp % shape.data[i];
          tmp /= shape.data[i];
          gid_a += idx * strides.data[i];
        }
        out[gid] = a[gid_a + offset];
      }
      /// END YOUR SOLUTION
    }

    void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
       * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
       * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
       * the functions after this, however, you'll need to define these kernels as you see fit to
       * execute the underlying function.
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */

      // Nothing needs to be added here
      CudaDims dim = CudaOneDim(out->size);
      CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
    }

    __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                       CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        size_t total = offset, num = gid;
        for (int idx = shape.size - 1; idx >= 0; idx--)
        {
          total += (num % shape.data[idx]) * strides.data[idx];
          num /= shape.data[idx];
        }
        out[total] = a[gid];
      }
    }

    void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */

      CudaDims dim = CudaOneDim(a.size);
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
    }

    __global__ void ScalarSetitemKernel(const size_t size, const scalar_t val, scalar_t *out, CudaVec shape,
                                        CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        size_t total = offset, num = gid;
        for (int idx = shape.size - 1; idx >= 0; idx--)
        {
          total += (num % shape.data[idx]) * strides.data[idx];
          num /= shape.data[idx];
        }
        out[total] = val;
      }
    }

    void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but covenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */

      CudaDims dim = CudaOneDim(size);
      ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                                   VecToCuda(strides), offset);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + b[gid];
    }

    void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      /**
       * Add together two CUDA array
       */
      CudaDims dim = CudaOneDim(out->size);
      EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + val;
    }

    void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      /**
       * Add together a CUDA array and a scalar value.
       */
      CudaDims dim = CudaOneDim(out->size);
      ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = a[gid] * b[gid];
      }
    }

    void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = a[gid] * val;
      }
    }

    void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = a[gid] / b[gid];
      }
    }

    void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = a[gid] / val;
      }
    }

    void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void ScalarPowerKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = std::pow(a[gid], val);
      }
    }

    void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] > b[gid]) ? a[gid] : b[gid];
      }
    }

    void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] > val) ? a[gid] : val;
      }
    }

    void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] == b[gid]);
      }
    }

    void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] == val);
      }
    }

    void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] >= b[gid]);
      }
    }

    void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = (a[gid] >= val);
      }
    }

    void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    __global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = std::log(a[gid]);
      }
    }

    void EwiseLog(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }

    __global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = std::exp(a[gid]);
      }
    }

    void EwiseExp(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }

    __global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        out[gid] = std::tanh(a[gid]);
      }
    }

    void EwiseTanh(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }

    __global__ void MatmulKernel(scalar_t *a, scalar_t *b, scalar_t *out, size_t size, uint32_t M,
                                 uint32_t N, uint32_t P)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        size_t i = gid / P;
        size_t j = gid % P;
        scalar_t total = 0;
        for (size_t k = 0; k < N; k++)
          total += a[i * N + k] * b[k * P + j];
        out[gid] = total;
      }
    }

    void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P)
    {
      /**
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   M: rows of a / out
       *   N: columns of a / rows of b
       *   P: columns of b / out
       */
      CudaDims dim = CudaOneDim(out->size);
      MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, M, N, P);
    }

    __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        size_t start = gid * reduce_size;
        scalar_t mmax = a[start];
        for (size_t i = 1; i < reduce_size; i++)
        {
          mmax = (mmax >= a[start + i]) ? mmax : a[start + i];
        }
        out[gid] = mmax;
      }
    }

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */

      CudaDims dim = CudaOneDim(out->size);
      ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
    }

    __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
      {
        size_t start = gid * reduce_size;
        scalar_t ssum = 0.0;
        for (size_t i = 0; i < reduce_size; i++)
        {
          ssum += a[start + i];
        }
        out[gid] = ssum;
      }
    }

    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      CudaDims dim = CudaOneDim(out->size);
      ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
    }

  } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
