/**
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines function to compute the determinant of
 *   a 2D blitz array matrix.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_DET_H
#define BOB_MATH_DET_H

#include <blitz/array.h>

namespace bob { namespace math {

  /**
   * @brief Function which computes the determinant of a square matrix
   * @param A The A matrix to consider (size NxN)
   */
  double det(const blitz::Array<double,2>& A);
  /**
   * @brief Function which computes the determinant of a square matrix
   * @param A The A matrix to consider (size NxN)
   * @warning Does not check the input matrix
   */
  double det_(const blitz::Array<double,2>& A);

  /**
   * @brief Function which computes the sign and (natural) logarithm of
   *   the determinant of a square matrix
   * @param A The A matrix to consider (size NxN)
   * @param sign The (output) sign of the determinant
   *   (-1 if negative, 0 if zero, +1 if positive)
   */
  double slogdet(const blitz::Array<double,2>& A, int& sign);
  /**
   * @brief Function which computes the sign and (natural) logarithm of
   *   the determinant of a square matrix
   * @param A The A matrix to consider (size NxN)
   * @param sign The (output) sign of the determinant
   *   (-1 if negative, 0 if zero, +1 if positive)
   * @warning Does not check the input matrix
   */
  double slogdet_(const blitz::Array<double,2>& A, int& sign);

}}

#endif /* BOB_MATH_DET_H */
