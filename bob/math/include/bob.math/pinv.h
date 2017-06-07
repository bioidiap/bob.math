/**
 * @date Tue Jun 18 18:27:22 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to determine the pseudo-inverse
 * using the SVD method.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_PINV_H
#define BOB_MATH_PINV_H

#include <blitz/array.h>

namespace bob { namespace math {

  /**
   * @brief Function which computes the pseudo-inverse using the SVD method.
   * @warning The output blitz::array B should have the correct
   *   size, with zero base index. Checks are performed.
   * @param A The A matrix to decompose (size MxN)
   * @param B The pseudo-inverse of the matrix A (size NxM)
   * @param rcond Cutoff for small singular values. Singular values smaller
   *   (in modulus) than rcond * largest_singular_value (again, in modulus)
   *   are set to zero.
   */
  void pinv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B,
      const double rcond=1e-15);
  /**
   * @brief Function which computes the pseudo-inverse using the SVD method.
   * @warning The output blitz::array B should have the correct
   *   size, with zero base index. Checks are NOT performed.
   * @param A The A matrix to decompose (size MxN)
   * @param B The pseudo-inverse of the matrix A (size NxM)
   * @param rcond Cutoff for small singular values. Singular values smaller
   *   (in modulus) than rcond * largest_singular_value (again, in modulus)
   *   are set to zero.
   */
  void pinv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B,
      const double rcond=1e-15);

}}

#endif /* BOB_MATH_PINV_H */

