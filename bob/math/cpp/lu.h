/**
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_LU_H
#define BOB_MATH_LU_H

#include <blitz/array.h>

namespace bob { namespace math {

  /**
   * @brief Function which performs a LU decomposition of a real
   *   matrix, using the dgetrf LAPACK function: \f$A = P*L*U\f$
   * @param A The A matrix to decompose (size MxN)
   * @param L The L lower-triangular matrix of the decomposition (size Mxmin(M,N))
   * @param U The U upper-triangular matrix of the decomposition (size min(M,N)xN)
   * @param P The P permutation matrix of the decomposition (size min(M,N)xmin(M,N))
   */
  void lu(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
      blitz::Array<double,2>& U, blitz::Array<double,2>& P);
  void lu_(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
      blitz::Array<double,2>& U, blitz::Array<double,2>& P);

  /**
   * @brief Performs the Cholesky decomposition of a real symmetric
   *   positive-definite matrix into the product of a lower triangular matrix
   *   and its transpose. When it is applicable, this is much more efficient
   *   than the LU decomposition. It uses the dpotrf LAPACK function:
   *     \f$A = L*L^{T}\f$
   * @param A The A matrix to decompose (size NxN)
   * @param L The L lower-triangular matrix of the decomposition
   */
  void chol(const blitz::Array<double,2>& A, blitz::Array<double,2>& L);
  void chol_(const blitz::Array<double,2>& A, blitz::Array<double,2>& L);
  /**
   * @}
   */

}}

#endif /* BOB_MATH_LU_H */
