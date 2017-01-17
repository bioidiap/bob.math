/**
 * @date Tue Jan 10 21:14 2016 +0100
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/shared_array.hpp>

#include <bob.math/gsvd.h>

#include <bob.core/assert.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>
#include <bob.math/linear.h>


// Declaration of the external LAPACK function (Divide and conquer SVD)
extern "C" void dggsvd3_(const char *jobu,
                         const char *jobv, 
                         const char *jobq,
                         const int *M,
                         const int *N,
                         const int *P,
                         const int *K,
                         const int *L,
                         double *A, 
                         const int *lda, 
                         double *B, 
                         const int *ldb,
                         double *alpha,
                         double *beta,
                         double *U,
                         const int *ldu, 
                         double *V,
                         const int *ldv, 
                         double *Q,
                         const int *ldq, 
                         double *work,
                         const int *lwork,
                         int *iwork,
                         int *info);


void bob::math::gsvd( blitz::Array<double,2>& A,
                      blitz::Array<double,2>& B,
                      blitz::Array<double,2>& U,
                      blitz::Array<double,2>& V,
                      blitz::Array<double,2>& zeroR,
                      blitz::Array<double,2>& Q,
                      blitz::Array<double,2>& X,
                      blitz::Array<double,2>& C,
                      blitz::Array<double,2>& S)
{
  const char jobu = 'U';
  const char jobv = 'V';
  const char jobq = 'Q';

  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int P = B.extent(0);
  
  const int lda = std::max(1,M);
  const int ldb = std::max(1,P);
  const int ldu = std::max(1,M);
  const int ldv = std::max(1,P);
  const int ldq = std::max(1,N);
  
  int K = 0; //out
  int L = 0; //out  
  
  // Prepares to call LAPACK function:
  // We will decompose A^T rather than A and B^T rather than B to reduce the required number of copy
  // We recall that FORTRAN/LAPACK is column-major order whereas blitz arrays
  // are row-major order by default.

  //A_lapack = A^T
  blitz::Array<double,2> A_blitz_lapack(bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();


  //B_lapack = B^T
  blitz::Array<double,2> B_blitz_lapack(bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(B).transpose(1,0)));
  double* B_lapack = B_blitz_lapack.data();
  
  
  // U. We will trainpose this one in the end
  double *U_lapack = U.data();

  // V. We will trainpose this one in the end
  double *V_lapack = V.data();

  // For Q, we will return the transpose  
  double *Q_lapack = Q.data();

  // In LAPACK C and S is 1-d. Our code makes it diagonal
  blitz::Array<double,1> C_1d(N); C_1d = 0;
  double *C_lapack = C_1d.data();
  
  blitz::Array<double,1> S_1d(N); S_1d = 0;
  double *S_lapack = S_1d.data();

  const int lwork_query = -1;
  double work_query;

  boost::shared_array<int> iwork(new int[N]);

  int info = 0;
  // A/ Queries the optimal size of the working array
  
  dggsvd3_(&jobu,
          &jobv, 
          &jobq,
          &M,
          &N,
          &P,
          &K,
          &L,
          A_lapack, 
          &lda, 
          B_lapack, 
          &ldb,
          C_lapack,
          S_lapack,
          U_lapack,
          &ldu, 
          V_lapack,
          &ldv, 
          Q_lapack,
          &ldq, 
          &work_query,
          &lwork_query,
          iwork.get(),
          &info);
          
  if (info != 0)
    throw std::runtime_error("The LAPACK dggsvd3 function returned a non-zero value during the checking");

  // B/ Computes
  
  const int lwork = static_cast<int>(work_query);
  boost::shared_array<double> work(new double[lwork]);
  
  dggsvd3_(&jobu,
            &jobv, 
           &jobq,
           &M,
           &N,
           &P,
           &K,
           &L,
           A_lapack, 
           &lda, 
           B_lapack, 
           &ldb,
           C_lapack,
           S_lapack,
           U_lapack,
           &ldu, 
           V_lapack,
           &ldv, 
           Q_lapack,
           &ldq, 
           work.get(),
           &lwork,
           iwork.get(),
           &info);
  if (info != 0)
    throw std::runtime_error("The LAPACK dggsvd3 function returned a non-zero value during the computation.");
  

  /* 
  According to the website http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga4a187519e5c71da3b3f67c85e9baf0f2.html#ga4a187519e5c71da3b3f67c85e9baf0f2
  
  if (M-K-L >= 0)
                      N-K-L  K    L
        ( 0 R ) = K (  0   R11  R12 )
                  L (  0    0   R22 )
                  
                  
        Where R11, R12, R22, = A(1:K+L,N-K-L+1:N)
  
   else
   
                        N-K-L  K   M-K  K+L-M
       ( 0 R ) =     K ( 0    R11  R12  R13  )
                   M-K ( 0     0   R22  R23  )
                 K+L-M ( 0     0    0   R33  )


       where R11, R12, R13, R22, R23 = A(1:M, N-K-L+1:N)
       where R33 = B(M-K+1:L,N+M-K-L+1:N)

  */
  int r = K + L; //Dimension of R
  C.resize(M, r);
  S.resize(P, r);
  S = 0;
  C = 0;
  zeroR.resize(r, N);
  zeroR = 0;
  X.resize(r,N);
  if (M-K-L >= 0){

    //1. First we need the [0 R], which has the shape (N-K-L + K + L, K+L)    
    zeroR(blitz::Range(0, r-1), blitz::Range(N-r, N-1)) = A_blitz_lapack.transpose(1,0)(blitz::Range(0,r-1), blitz::Range(N-r,N-1));
    // 2. Now we have to deal with C and S according to http://www.netlib.org/lapack/lug/node36.html
    //    In the end C is m-by-r and is p-by-r, both are real, nonnegative and diagonal and C'C + S'S = I ,
    //    They have the following structure 
    //    COPY AND PASTE

    //2.1 Preparing C
    //           K  L
    //   C = K ( I  0 )
    //       L ( 0  C )
    //   M-K-L ( 0  0 )
    
    // A - Identity part
    if (K>0){
      blitz::Array<double,2> I (K,K); I= 0;
      bob::math::eye_(I);
      C(blitz::Range(0, K-1), blitz::Range(0, K-1)) = I;
    }
    // B - diag(C) part. Here the C is LxL
    // Swaping
    
    bob::math::swap_(C_1d, iwork.get(), K, std::min(M,r));
    blitz::Array<double,2> C_diag (L,L); C_diag = 0;
    bob::math::diag(C_1d(blitz::Range(K,K+L-1)), C_diag);
    C(blitz::Range(K, M-1), blitz::Range(K, K+L-1)) = C_diag;

    //2.2 Preparing S
    //           K  L
    //D2 =   L ( 0  S )
    //     P-L ( 0  0 )

    // A - diag(S) part
    // Swap
    bob::math::swap_(S_1d, iwork.get(), K, std::min(M,r));
    blitz::Array<double,2> S_diag (L,L); S_diag = 0;
    bob::math::diag(S_1d(blitz::Range(K,K+L-1)), S_diag);
    S(blitz::Range(0, L-1), blitz::Range(K, K+L-1)) = S_diag;

  }
  else{

    //1. First we need the [0 R], which has the shape (N-K-L  K   M-K  K+L-M  ,  k + M-K  K+L-M  )

    //A. First part of R is in A(1:M, N-K-L+1:N) 
    zeroR(blitz::Range(0,M-1), blitz::Range(N-K-L, N-1)) = A_blitz_lapack.transpose(1,0)(blitz::Range(0,M-1) , blitz::Range(N-K-L,N-1));

    //B. Second part of R is in B(M-K+1:L,N+M-K-L+1:N)
    zeroR(blitz::Range(M, r - 1), blitz::Range(N-L+M-K,N-1)) = B_blitz_lapack.transpose(1,0)(blitz::Range(M-K, L-1) , blitz::Range(N+M-K-L, N-1));

    //2. Now we have to deal with C and S according to http://www.netlib.org/lapack/lug/node36.html
    //    In the end C is m-by-r and is p-by-r, both are real, nonnegative and diagonal and C'C + S'S = I ,
    //    They have the following structure 
    //    COPY AND PASTE

    //2.1 Preparing C, where C=diag( ALPHA(K+1), ... , ALPHA(M) ),
    //             K M-K K+L-M
    //  D1 =   K ( I  0    0   )
    //       M-K ( 0  C    0   )

    // A - Identity part
    if (K>0){
      blitz::Array<double,2> I (K,K); I = 0;
      bob::math::eye_(I);
      C(blitz::Range(0, K-1), blitz::Range(0, K-1)) = I;
    }

    // B - diag(C) part
    // Swaping
    blitz::Array<double,1> C_1d_cropped(M-K); C_1d_cropped = 0;
    C_1d_cropped = C_1d(blitz::Range(K,K+M-1));
    bob::math::swap_(C_1d_cropped, iwork.get(), K, std::min(M,r));
    blitz::Array<double,2> C_diag (M,M); C_diag = 0;
    bob::math::diag(C_1d_cropped, C_diag);
    C(blitz::Range(K,M-1), blitz::Range(K,M-1)) = C_diag;


    //2.2 Preparing S
    //               K M-K K+L-M
    //  D2 =   M-K ( 0  S    0  )
    //       K+L-M ( 0  0    I  )
    //         P-L ( 0  0    0  )

    // A - Identity part
    if (K+L-M>0){
      blitz::Array<double,2> I (K+L-M,K+L-M); I= 0;
      bob::math::eye_(I);
      S(blitz::Range(M-K,L-1), blitz::Range(M,K+L-1)) = I;
    }

    // B - diag(S) part
    // Swaping
    blitz::Array<double,1> S_1d_cropped(M-K); S_1d_cropped = 0;
    S_1d_cropped = S_1d(blitz::Range(K,K+M-1));
    
    bob::math::swap_(S_1d_cropped, iwork.get(), K, std::min(M,r));
    blitz::Array<double,2> S_diag (M,M); S_diag = 0;
    bob::math::diag(S_1d_cropped, S_diag);
    S(blitz::Range(0,M-K-1), blitz::Range(K,M-1)) = S_diag;
  }

  // Transposing U and V
  blitz::Array<double,2> Ut(
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(U).transpose(1,0)));
  U = Ut;

  blitz::Array<double,2> Vt(
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(V).transpose(1,0)));
  V = Vt;

  // Swaping U
  bob::math::swap_(U, iwork.get(), K, std::min(M,r));
  // Swaping V
  bob::math::swap_(V, iwork.get(), K, std::min(M,r));

  //Computing X
  bob::math::prod_(zeroR, Q, X);
  blitz::Array<double,2> Xt(
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(X).transpose(1,0)));
  X = Xt;
  bob::math::swap_(X, iwork.get(), K, std::min(M,r));

} 
