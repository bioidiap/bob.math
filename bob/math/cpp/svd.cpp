/**
 * @date Sat Mar 19 22:14:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/shared_array.hpp>

#include <bob.math/svd.h>

#include <bob.core/assert.h>
#include <bob.core/check.h>
#include <bob.core/array_copy.h>

// Declaration of the external LAPACK function (Divide and conquer SVD)
extern "C" void dgesdd_( const char *jobz, const int *M, const int *N,
  double *A, const int *lda, double *S, double *U, const int* ldu, double *VT,
  const int *ldvt, double *work, const int *lwork, int *iwork, int *info);

// Declaration of the external LAPACK function ('Slow' but 'safe' SVD)
extern "C" void dgesvd_( const char *jobu, const char *jobvt, const int *M,
  const int *N, double *A, const int *lda, double *S, double *U,
  const int* ldu, double *VT, const int *ldvt, double *work, const int *lwork,
  int *info);

static void svd_lapack( const char jobz, const int M, const int N,
  double *A, const int lda, double *S, double *U, const int ldu, double *VT,
  const int ldvt, const bool safe)
{
  // Calls the LAPACK function:
  // We use dgesdd by default which is faster than its predecessor dgesvd,
  // when computing the singular vectors.
  //   (cf. http://www.netlib.org/lapack/lug/node71.html)
  // However, dgesdd is failing on some matrices:
  //   see #171: http://github.com/idiap/bob/issues/171
  // Please note that matlab is relying on dgesvd.
  int info = 0;
  if (safe) {
    // A/ Queries the optimal size of the working array
    const int lwork_query = -1;
    double work_query;
    dgesvd_( &jobz, &jobz, &M, &N, A, &lda, S, U, &ldu,
      VT, &ldvt, &work_query, &lwork_query, &info );
    // Check info variable
    if (info != 0)
      throw std::runtime_error("The LAPACK dgesvd function returned a non-zero value.");

    // B/ Computes
    const int lwork = static_cast<int>(work_query);
    boost::shared_array<double> work(new double[lwork]);
    dgesvd_( &jobz, &jobz, &M, &N, A, &lda, S, U, &ldu,
      VT, &ldvt, work.get(), &lwork, &info );
    // Check info variable
    if (info != 0)
      throw std::runtime_error("The LAPACK dgesvd function returned a non-zero value.");
  }
  else {
    // Integer (workspace) array, dimension (8*min(M,N))
    const int l_iwork = 8*std::min(M,N);
    boost::shared_array<int> iwork(new int[l_iwork]);

    // A/ Queries the optimal size of the working array
    const int lwork_query = -1;
    double work_query;
    dgesdd_( &jobz, &M, &N, A, &lda, S, U, &ldu,
      VT, &ldvt, &work_query, &lwork_query, iwork.get(), &info );
    // Check info variable
    if (info != 0)
      throw std::runtime_error("The LAPACK dgesdd function returned a non-zero value. You may consider using LAPACK dgsevd instead (see #171) by enabling the 'safe' option.");

    // B/ Computes
    const int lwork = static_cast<int>(work_query);
    boost::shared_array<double> work(new double[lwork]);
    dgesdd_( &jobz, &M, &N, A, &lda, S, U, &ldu,
      VT, &ldvt, work.get(), &lwork, iwork.get(), &info );
    // Check info variable
    if (info != 0)
      throw std::runtime_error("The LAPACK dgesdd function returned a non-zero value. You may consider using LAPACK dgsevd instead (see #171) by enabling the 'safe' option.");
  }
  
  // Defining the sign of the eigenvectors
  // Approch extracted from page 8 - http://prod.sandia.gov/techlib/access-control.cgi/2007/076422.pdf  
  if(U[0] < 0){    
    int ucol=0; ucol= (jobz=='A')? M : std::min(M,N);
    for (int i=0; i<ldu*ucol; i++){
      U[i] = -1*U[i];
    }

    for (int i=0; i<ldvt*N; i++){
      VT[i] = -1*VT[i];
    }
  }
}

void bob::math::svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Checks zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(U);
  bob::core::array::assertZeroBase(sigma);
  bob::core::array::assertZeroBase(Vt);
  // Checks and resizes if required
  bob::core::array::assertSameDimensionLength(U.extent(0), M);
  bob::core::array::assertSameDimensionLength(U.extent(1), M);
  bob::core::array::assertSameDimensionLength(sigma.extent(0), nb_singular);
  bob::core::array::assertSameDimensionLength(Vt.extent(0), N);
  bob::core::array::assertSameDimensionLength(Vt.extent(1), N);

  bob::math::svd_(A, U, sigma, Vt, safe);
}

void bob::math::svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& Vt, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Prepares to call LAPACK function:
  // We will decompose A^T rather than A to reduce the required number of copy
  // We recall that FORTRAN/LAPACK is column-major order whereas blitz arrays
  // are row-major order by default.
  // If A = U.S.V^T, then A^T = V.S.U^T

  // Initialises LAPACK variables
  const char jobz = 'A'; // Get All left singular vectors
  const int lda = N;
  const int ldu = N;
  const int ldvt = M;
  // Initialises LAPACK arrays
  blitz::Array<double,2> A_blitz_lapack(bob::core::array::ccopy(A));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use U, Vt and S directly to limit the number of copy()
  // S_lapack = S
  blitz::Array<double,1> S_blitz_lapack;
  const bool sigma_direct_use = bob::core::array::isCZeroBaseContiguous(sigma);
  if (!sigma_direct_use) S_blitz_lapack.resize(nb_singular);
  else                   S_blitz_lapack.reference(sigma);
  double *S_lapack = S_blitz_lapack.data();
  // U_lapack = V^T
  blitz::Array<double,2> U_blitz_lapack;
  const bool U_direct_use = bob::core::array::isCZeroBaseContiguous(Vt);
  if (!U_direct_use) U_blitz_lapack.resize(N,N);
  else               U_blitz_lapack.reference(Vt);
  double *U_lapack = U_blitz_lapack.data();
  // V^T_lapack = U
  blitz::Array<double,2> VT_blitz_lapack;
  const bool VT_direct_use = bob::core::array::isCZeroBaseContiguous(U);
  if (!VT_direct_use) VT_blitz_lapack.resize(M,M);
  else                VT_blitz_lapack.reference(U);
  double *VT_lapack = VT_blitz_lapack.data();

  // Call the LAPACK function
  svd_lapack(jobz, N, M, A_lapack, lda, S_lapack, U_lapack, ldu,
    VT_lapack, ldvt, safe);  


  // Copy singular vectors back to U, V and sigma if required
  if (!U_direct_use)  Vt = U_blitz_lapack;
  if (!VT_direct_use) U = VT_blitz_lapack;
  if (!sigma_direct_use) sigma = S_blitz_lapack;
}


void bob::math::svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Checks zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(U);
  bob::core::array::assertZeroBase(sigma);
  // Checks and resizes if required
  bob::core::array::assertSameDimensionLength(U.extent(0), M);
  bob::core::array::assertSameDimensionLength(U.extent(1), nb_singular);
  bob::core::array::assertSameDimensionLength(sigma.extent(0), nb_singular);

  bob::math::svd_(A, U, sigma, safe);
}

void bob::math::svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Prepares to call LAPACK function

  // Initialises LAPACK variables
  const char jobz = 'S'; // Get first min(M,N) columns of U
  const int lda = M;
  const int ldu = M;
  const int ldvt = std::min(M,N);

  // Initialises LAPACK arrays
  blitz::Array<double,2> A_blitz_lapack(bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use U and S directly to limit the number of copy()
  // S_lapack = S
  blitz::Array<double,1> S_blitz_lapack;
  const bool sigma_direct_use = bob::core::array::isCZeroBaseContiguous(sigma);
  if (!sigma_direct_use) S_blitz_lapack.resize(nb_singular);
  else                   S_blitz_lapack.reference(sigma);
  double *S_lapack = S_blitz_lapack.data();
  // U_lapack = U^T
  blitz::Array<double,2> U_blitz_lapack;
  blitz::Array<double,2> Ut = U.transpose(1,0);
  const bool U_direct_use = bob::core::array::isCZeroBaseContiguous(Ut);
  if (!U_direct_use) U_blitz_lapack.resize(nb_singular,M);
  else               U_blitz_lapack.reference(Ut);
  double *U_lapack = U_blitz_lapack.data();
  boost::shared_array<double> VT_lapack(new double[nb_singular*N]);

  // Call the LAPACK function
  svd_lapack(jobz, M, N, A_lapack, lda, S_lapack, U_lapack, ldu,
    VT_lapack.get(), ldvt, safe);

  // Copy singular vectors back to U, V and sigma if required
  if (!U_direct_use) Ut = U_blitz_lapack;
  if (!sigma_direct_use) sigma = S_blitz_lapack;
}


void bob::math::svd(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Checks zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(sigma);
  // Checks and resizes if required
  bob::core::array::assertSameDimensionLength(sigma.extent(0), nb_singular);

  bob::math::svd_(A, sigma, safe);
}

void bob::math::svd_(const blitz::Array<double,2>& A, blitz::Array<double,1>& sigma, bool safe)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Prepares to call LAPACK function

  // Initialises LAPACK variables
  const char jobz = 'N'; // Get first min(M,N) columns of U
  const int lda = M;
  const int ldu = M;
  const int ldvt = std::min(M,N);

  // Initialises LAPACK arrays
  blitz::Array<double,2> A_blitz_lapack(
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use S directly to limit the number of copy()
  // S_lapack = S
  blitz::Array<double,1> S_blitz_lapack;
  const bool sigma_direct_use = bob::core::array::isCZeroBaseContiguous(sigma);
  if (!sigma_direct_use) S_blitz_lapack.resize(nb_singular);
  else                   S_blitz_lapack.reference(sigma);
  double *S_lapack = S_blitz_lapack.data();
  double *U_lapack = 0;
  double *VT_lapack = 0;

  // Call the LAPACK function
  svd_lapack(jobz, M, N, A_lapack, lda, S_lapack, U_lapack, ldu,
    VT_lapack, ldvt, safe);

  // Copy singular vectors back to U, V and sigma if required
  if (!sigma_direct_use) sigma = S_blitz_lapack;
}
