/**
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/shared_array.hpp>

#include <bob.math/inv.h>

#include <bob.math/linear.h>

#include <bob.core/assert.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>

// Declaration of the external LAPACK function
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( const int *M, const int *N, double *A, const int *lda,
  int *ipiv, int *info);
// Inverse of a general matrix (dgetri)
extern "C" void dgetri_( const int *N, double *A, const int *lda,
  const int *ipiv, double *work, const int *lwork, int *info);

void bob::math::inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  const int N = A.extent(0);
  const blitz::TinyVector<int,2> shapeA(N,N);
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(B);

  bob::core::array::assertSameShape(A,shapeA);
  bob::core::array::assertSameShape(B,shapeA);

  bob::math::inv_(A, B);
}

void bob::math::inv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  const int N = A.extent(0);

  //////////////////////////////////////
  // Prepares to call LAPACK functions
  // Initializes LAPACK variables
  int info = 0;
  const int lda = N;

  // Initializes LAPACK arrays
  boost::shared_array<int> ipiv(new int[N]);

  // Tries to use B directly if possible
  //   Input and output arrays are both column-major order.
  //   Hence, we can ignore the problem of column- and row-major order
  //   conversions.
  bool B_direct_use = bob::core::array::isCZeroBaseContiguous(B);
  blitz::Array<double,2> A_blitz_lapack;
  if (B_direct_use)
  {
    A_blitz_lapack.reference(B);
    A_blitz_lapack = A;
  }
  else
    A_blitz_lapack.reference(bob::core::array::ccopy(A));
  double *A_lapack = A_blitz_lapack.data();


  // Calls the LAPACK functions
  // 1/ Computes the LU decomposition
  dgetrf_( &N, &N, A_lapack, &lda, ipiv.get(), &info);
  // Checks the info variable
  if (info != 0)
    throw std::runtime_error("The LAPACK dgetrf function returned a non-zero value.");

  // TODO: We might consider adding a real invertibility test as described in
  // this thread (Btw, this is what matlab does):
  // http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00778.html

  // 2/ Computes the inverse matrix
  // 2/A/ Queries the optimal size of the working array
  const int lwork_query = -1;
  double work_query;
  dgetri_( &N, A_lapack, &lda, ipiv.get(), &work_query, &lwork_query, &info);
  // 2/B/ Computes the inverse
  const int lwork = static_cast<int>(work_query);
  boost::shared_array<double> work(new double[lwork]);
  dgetri_( &N, A_lapack, &lda, ipiv.get(), work.get(), &lwork, &info);
  // Checks info variable
  if (info != 0)
    throw std::runtime_error("The LAPACK dgetri function returned a non-zero value. The matrix might not be invertible.");

  // Copy back content to B if required
  if (!B_direct_use)
    B = A_blitz_lapack;
}

