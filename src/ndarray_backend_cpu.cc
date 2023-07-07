#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace needle
{
  namespace cpu
  {

#define ALIGNMENT 256
#define TILE 8
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    /**
     * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
     * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
     * here by default.
     */
    struct AlignedArray
    {
      AlignedArray(const size_t size)
      {
        int ret = posix_memalign((void **)&ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0)
          throw std::bad_alloc();
        this->size = size;
      }
      ~AlignedArray() { free(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }
      scalar_t *ptr;
      size_t size;
    };

    void Fill(AlignedArray *out, scalar_t val)
    {
      /**
       * Fill the values of an aligned array with val
       */
      for (int i = 0; i < out->size; i++)
      {
        out->ptr[i] = val;
      }
    }

    void Compact(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory
       *
       * Args:
       *   a: non-compact representation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */
      int length = shape.size();
      size_t total_size = 1;
      std::vector<size_t> sizes;
      sizes.reserve(length + 1);
      for (int idx = length - 1; idx >= 0; idx--)
      {
        sizes.push_back(total_size);
        total_size *= shape[idx];
      }

      for (size_t i = 0; i < total_size; i++)
      {
        int total = offset;
        for (int idx = length - 1; idx >= 0; idx--)
        {
          int num = i / sizes[length - 1 - idx] % shape[idx];
          total += strides[idx] * num;
        }
        out->ptr[i] = a.ptr[total];
      }
    }

    void EwiseSetitem(const AlignedArray &a, AlignedArray *out, std::vector<uint32_t> shape,
                      std::vector<uint32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      int length = shape.size();
      size_t total_size = 1;
      std::vector<size_t> sizes;
      sizes.reserve(length + 1);
      for (int idx = length - 1; idx >= 0; idx--)
      {
        sizes.push_back(total_size);
        total_size *= shape[idx];
      }

      for (size_t i = 0; i < total_size; i++)
      {
        size_t total = offset;
        for (int idx = length - 1; idx >= 0; idx--)
        {
          int num = i / sizes[length - 1 - idx] % shape[idx];
          total += strides[idx] * num;
        }
        out->ptr[total] = a.ptr[i];
      }
    }

    void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, std::vector<uint32_t> shape,
                       std::vector<uint32_t> strides, size_t offset)
    {
      int length = shape.size();
      size_t total_size = 1;
      std::vector<size_t> sizes;
      sizes.reserve(length + 1);
      for (int idx = length - 1; idx >= 0; idx--)
      {
        sizes.push_back(total_size);
        total_size *= shape[idx];
      }

      for (size_t i = 0; i < total_size; i++)
      {
        size_t total = offset;
        for (int idx = length - 1; idx >= 0; idx--)
        {
          int num = i / sizes[length - 1 - idx] % shape[idx];
          total += strides[idx] * num;
        }
        out->ptr[total] = val;
      }
    }

    template <typename Op>
    void EwiseOp(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, Op op)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = op(a.ptr[i], b.ptr[i]);
      }
    }

    template <typename Op>
    void ScalarOp(const AlignedArray &a, scalar_t val, AlignedArray *out, Op op)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = op(a.ptr[i], val);
      }
    }

    void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      EwiseOp(a, b, out, std::plus<scalar_t>());
    }

    void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      ScalarOp(a, val, out, std::plus<scalar_t>());
    }

    void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      EwiseOp(a, b, out, std::multiplies<scalar_t>());
    }

    void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      ScalarOp(a, val, out, std::multiplies<scalar_t>());
    }

    void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      EwiseOp(a, b, out, std::divides<scalar_t>());
    }

    void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      ScalarOp(a, val, out, std::divides<scalar_t>());
    }

    void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = std::pow(a.ptr[i], val);
      }
    }

    void EwiseMaximum(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
      }
    }

    void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = std::max(a.ptr[i], val);
      }
    }

    void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      EwiseOp(a, b, out, std::equal_to<scalar_t>());
    }

    void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      ScalarOp(a, val, out, std::equal_to<scalar_t>());
    }

    void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      EwiseOp(a, b, out, std::greater_equal<scalar_t>());
    }

    void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      ScalarOp(a, val, out, std::greater_equal<scalar_t>());
    }

    void EwiseLog(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = log(a.ptr[i]);
      }
    }

    void EwiseExp(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = exp(a.ptr[i]);
      }
    }

    void EwiseTanh(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = tanh(a.ptr[i]);
      }
    }

    void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
                uint32_t p)
    {
      /**
       * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
       * you can use the "naive" three-loop algorithm.
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: compact 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: columns of b / out
       */

      Fill(out, 0);
      for (int i = 0; i < m; ++i)
      {
        for (int k = 0; k < n; ++k)
        {
          auto s = a.ptr[i * n + k];
          for (int j = 0; j < p; ++j)
          {
            out->ptr[i * p + j] += s * b.ptr[k * p + j];
          }
        }
      }
    }

    inline void AlignedDot(const float *__restrict__ a,
                           const float *__restrict__ b,
                           float *__restrict__ out)
    {

      /**
       * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
       * the result to the existing out, which you should not set to zero beforehand).  We are including
       * the compiler flags here that enable the compile to properly use vector operators to implement
       * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
       * out don't have any overlapping memory (which is necessary in order for vector operations to be
       * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
       * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
       * compiler that the input array will be aligned to the appropriate blocks in memory, which also
       * helps the compiler vectorize the code.
       *
       * Args:
       *   a: compact 2D array of size TILE x TILE
       *   b: compact 2D array of size TILE x TILE
       *   out: compact 2D array of size TILE x TILE to write to
       */

      a = (const float *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
      b = (const float *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
      out = (float *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

      for (int i = 0; i < TILE; ++i)
      {
        for (int k = 0; k < TILE; ++k)
        {
          auto s = a[i * TILE + k];
          for (int j = 0; j < TILE; ++j)
          {
            out[i * TILE + j] += s * b[k * TILE + j];
          }
        }
      }
    }

    void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                     uint32_t n, uint32_t p)
    {
      /**
       * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
       * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
       *   a[m/TILE][n/TILE][TILE][TILE]
       * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
       * function should call `AlignedDot()` implemented above).
       *
       * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
       * assume that this division happens without any remainder.
       *
       * Args:
       *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
       *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
       *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: columns of b / out
       *
       */
      Fill(out, 0);
      int m_tile = m / TILE, n_tile = n / TILE, p_tile = p / TILE;
      int tile_sq = TILE * TILE;
      for (int i = 0; i < m_tile; ++i)
      {
        for (int k = 0; k < n_tile; ++k)
        {
          auto a_ptr = a.ptr + (i * n_tile + k) * tile_sq;
          for (int j = 0; j < p_tile; ++j)
          {
            auto b_ptr = b.ptr + (k * p_tile + j) * tile_sq;
            auto out_ptr = out->ptr + (i * p_tile + j) * tile_sq;
            AlignedDot(a_ptr, b_ptr, out_ptr);
          }
        }
      }
    }

    void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   reduce_size: size of the dimension to reduce over
       */

      for (size_t idx_out = 0; idx_out < out->size; idx_out++)
      {
        size_t start = idx_out * reduce_size;
        scalar_t mmax = a.ptr[start];
        for (size_t i = 1; i < reduce_size; i++)
        {
          mmax = std::max(mmax, a.ptr[start + i]);
        }
        out->ptr[idx_out] = mmax;
      }
    }

    void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking sum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   reduce_size: size of the dimension to reduce over
       */

      for (size_t idx_out = 0; idx_out < out->size; idx_out++)
      {
        size_t start = idx_out * reduce_size;
        scalar_t ssum = 0.0;
        for (size_t i = 0; i < reduce_size; i++)
        {
          ssum += a.ptr[start + i];
        }
        out->ptr[idx_out] = ssum;
      }
    }

  } // namespace cpu
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out)
        { std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE); });

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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
