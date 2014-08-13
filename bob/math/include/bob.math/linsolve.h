/**
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines functions to solve linear systems
 *   A*x=b using LAPACK or a conjugate gradient implementation.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_LINSOLVE_H
#define BOB_MATH_LINSOLVE_H

#include <blitz/array.h>

namespace bob { namespace math {

  /**
   * @brief Function which solves a linear system of equation using the
   *   'generic' dgsev LAPACK function.
   * @param A The A squared-matrix of the system A*x=b (size NxN)
   * @param b The b vector of the system A*x=b (size N)
   * @param x The x vector of the system A*x=b which will be updated
   *   at the end of the function.
   */
  void linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b);
  void linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b);

  /**
   * @brief Function which solves a linear system of equation using the
   *   'generic' dgsev LAPACK function.
   * @param A The A squared-matrix of the system A*X=B (size NxN)
   * @param B The B matrix of the system A*X=B (size NxP)
   * @param X The X matrix of the system A*X=B which will be updated
   *   at the end of the function (size NxP).
   */
  void linsolve(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
      const blitz::Array<double,2>& B);
  void linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
      const blitz::Array<double,2>& B);

  /**
   * @brief Function which solves a symmetric positive definite linear
   *   system of equation using the dposv LAPACK function.
   * @warning No check is performed wrt. to the fact that A should be
   *   symmetric positive definite.
   * @param A The A squared-matrix, symmetric definite positive, of the
   *   system A*x=b (size NxN)
   * @param b The b vector of the system A*x=b (size N)
   * @param x The x vector of the system A*x=b which will be updated
   *   at the end of the function (size N)
   */
  void linsolveSympos(const blitz::Array<double,2>& A,
      blitz::Array<double,1>& x, const blitz::Array<double,1>& b);
  void linsolveSympos_(const blitz::Array<double,2>& A,
      blitz::Array<double,1>& x, const blitz::Array<double,1>& b);

  /**
   * @brief Function which solves a symmetric positive definite linear
   *   system of equation using the dposv LAPACK function.
   * @warning No check is performed wrt. to the fact that A should be
   *   symmetric positive definite.
   * @param A The A squared-matrix, symmetric definite positive, of the
   *   system A*X=B (size NxN)
   * @param B The B vector of the system A*x=b (size NxP)
   * @param X The X vector of the system A*x=b which will be updated
   *   at the end of the function (size NxP)
   */
  void linsolveSympos(const blitz::Array<double,2>& A,
      blitz::Array<double,2>& X, const blitz::Array<double,2>& B);
  void linsolveSympos_(const blitz::Array<double,2>& A,
      blitz::Array<double,2>& X, const blitz::Array<double,2>& B);


  /**
   * @brief Function which solves a symmetric positive-definite linear
   *   system of equation via conjugate gradients.
   * @param A The A symmetric positive-definite squared-matrix of the
   *   system A*x=b (size NxN)
   * @param b The b vector of the system A*x=b (size N)
   * @param x The x vector of the system A*x=b which will be updated
   *   at the end of the function.
   * @param acc The desired accuracy. The algorithm terminates when
   *   norm(Ax-b)/norm(b) < acc
   * @param max_iter The maximum number of iterations
   */
  void linsolveCGSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);
  void linsolveCGSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
      const blitz::Array<double,1>& b, const double acc, const int max_iter);

}}

#endif /* BOB_MATH_LINSOLVE_H */

