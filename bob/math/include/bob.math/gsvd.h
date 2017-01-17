/**
 * @date Tue Jan 10 21:14 2016 +0100
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief This file defines a function to determine the SVD decomposition
 * a 2D blitz array using LAPACK.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_GSVD_H
#define BOB_MATH_GSVD_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Function which performs the Generalized Singular Value Decomposition
 *   using the 'simple' driver routine dggsvd3 of LAPACK.
 *   Just remembering that given A and B as input we have:
 *
 *    A = U C X and B = V S X
 *    where X = [0,R]Q^T
 *
 *
 *
 * @warning The output blitz::array sigma should have the correct 
 *   size, with zero base index. Checks are performed.
 * @param A The A matrix to decompose (size MxP)
 * @param B The B matrix to decompose (size NxP)
 * @warning A and B must have the same number of columns, but may have different numbers of rows. If A is m-by-p and B is n-by-p, then U is m-by-m, V is n-by-n, X is p-by-q, C is m-by-q
 *          and  S is n-by-q, where q = min(m+n,p).
 * @param U The U matrix (MxM)
 * @param V The V matrix (NxN)
 * @param zeroR The X matrix (rxN) 
 * @param Q The Q matrix (Q) 
 * @param C The C matrix (MxQ)
 * @param S The S matrix (NxQ)  
 */
void gsvd(blitz::Array<double,2>& A,
         blitz::Array<double,2>& B,

         blitz::Array<double,2>& U,
         blitz::Array<double,2>& V,
         blitz::Array<double,2>& zeroR,
         blitz::Array<double,2>& Q,
         blitz::Array<double,2>& X,
         blitz::Array<double,2>& C,
         blitz::Array<double,2>& S
         );
  /**
   * @brief Swaping using the LAPACK variable iWork
   *        http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga4a187519e5c71da3b3f67c85e9baf0f2.html#ga4a187519e5c71da3b3f67c85e9baf0f2
   *
   *        @param A 1D Matrix
   *        @param indexes Sorting information
   *        @param n number of operations
   */
    template<typename T>
        void swap_(blitz::Array<T,1>& A, int* indexes, int begin, int end) {
            T aux = 0;
            int fortran_index = 0;
            for (int i=begin; i<A.extent(0); i++){
                fortran_index = indexes[i]-1;
                aux = A(i);
                A(i) = A(fortran_index);
                A(fortran_index) = aux;
            }
        }


  /**
   * @brief Swaping using the LAPACK variable iWork
   *        http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga4a187519e5c71da3b3f67c85e9baf0f2.html#ga4a187519e5c71da3b3f67c85e9baf0f2
   *
   *        @param A D Matrix
   *        @param indexes Sorting information
   *        @param n number of operations
   */
    template<typename T>
        void swap_(blitz::Array<T,2>& A, int* indexes, int begin, int end) {
            blitz::Array<T,1> aux(A.extent(1)); aux = 0;
            int fortran_index = 0;
            for (int i=begin; i<end; i++){
                fortran_index = indexes[i]-1;
                
                if (fortran_index < A.extent(1)){
                    aux = A(blitz::Range::all(), i);
                    A(blitz::Range::all(), i) = A(blitz::Range::all(), fortran_index);
                    A(blitz::Range::all(), fortran_index) = aux;
                }
            }
        }






/**
 * @}
 */
}}

#endif /* BOB_MATH_GSVD_H */
