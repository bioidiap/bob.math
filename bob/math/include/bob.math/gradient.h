/**
 * @date Sat Apr 14 21:01:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Computes the gradient of a signal
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_GRADIENT_H
#define BOB_MATH_GRADIENT_H

#include <stdexcept>
#include <boost/format.hpp>

#include <bob.core/assert.h>

namespace bob { namespace math {

  /**
   * @brief Function which computes the gradient of a 1D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param g The output blitz array for the gradient
   * @param dx The sample distance along the x-axis
   * @warning Does not check that g has the same size as input
   */
  template <typename T, typename U>
    void gradient_(const blitz::Array<T,1>& input, blitz::Array<U,1>& g,
        const double dx=1.)
    {
      const int M=input.extent(0);
      // Check input
      if (M<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 0 % M;
        throw std::runtime_error(m.str());
      }
      if (!(dx>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dx % 0;
        throw std::runtime_error(m.str());
      }
      bob::core::array::assertZeroBase(input);
      bob::core::array::assertZeroBase(g);

      // Uncentered gradient at the boundaries
      g(0) = input(1) - input(0);
      g(M-1) = input(M-1) - input(M-2);

      // Centered gradient otherwise
      if(M>2)
      {
        blitz::Range r(1,M-2);
        blitz::Range rp(2,M-1);
        blitz::Range rm(0,M-3);
        g(r) = input(rp) - input(rm);
        g(r) /= 2.;
      }

      // Update scaling if required
      if (dx!=1.) g *= (1./dx);
    }

  /**
   * @brief Function which computes the gradient of a 1D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param dx The sample distance along the x-axis
   * @param g The output blitz array for the gradient
   */
  template <typename T, typename U>
    void gradient(const blitz::Array<T,1>& input, blitz::Array<U,1>& g,
        const double dx=1.)
    {
      // Check input size
      bob::core::array::assertSameShape(input, g);
      gradient_<T,U>(input, g, dx);
    }

  /**
   * @brief Function which computes the gradient of a 2D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param gy The output blitz array for the gradient along the y-axis
   * @param gx The output blitz array for the gradient along the x-axis
   * @param dy The sample distance along the y-axis
   * @param dx The sample distance along the x-axis
   * @warning Does not check that gx and gy have the same size as input
   * @warning For unsigned input datatypes, this function might not work as expected.
   */
  template <typename T, typename U>
    void gradient_(const blitz::Array<T,2>& input, blitz::Array<U,2>& gy,
        blitz::Array<U,2>& gx, const double dy=1., const double dx=1.)
    {
      const int M=input.extent(0);
      const int N=input.extent(1);
      // Check input
      if (M<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 0 % M;
        throw std::runtime_error(m.str());
      }
      if (N<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 1 % N;
        throw std::runtime_error(m.str());
      }
      if (!(dy>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dy % 0;
        throw std::runtime_error(m.str());
      }
      if (!(dx>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dx % 1;
        throw std::runtime_error(m.str());
      }
      bob::core::array::assertZeroBase(input);
      bob::core::array::assertZeroBase(gy);
      bob::core::array::assertZeroBase(gx);

      // Defines 'full' range
      blitz::Range rall = blitz::Range::all();

      // Uncentered gradient at the boundaries
      gy(0,rall) = input(1,rall) - input(0,rall);
      gy(M-1,rall) = input(M-1,rall) - input(M-2,rall);
      gx(rall,0) = input(rall,1) - input(rall,0);
      gx(rall,N-1) = input(rall,N-1) - input(rall,N-2);

      // Centered gradient otherwise
      if (M>2)
      {
        blitz::Range ry(1,M-2);
        blitz::Range ryp(2,M-1);
        blitz::Range rym(0,M-3);
        gy(ry,rall) = input(ryp,rall) - input(rym,rall);
        gy(ry,rall) /= 2.;
      }
      if (N>2)
      {
        blitz::Range rx(1,N-2);
        blitz::Range rxp(2,N-1);
        blitz::Range rxm(0,N-3);
        gx(rall,rx) = input(rall,rxp) - input(rall,rxm);
        gx(rall,rx) /= 2.;
      }

      // Update scaling if required
      if (dy!=1.) gy *= (1./dy);
      if (dx!=1.) gx *= (1./dx);
    }
  /**
   * @brief Function which computes the gradient of a 2D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param gy The output blitz array for the gradient along the y-axis
   * @param gx The output blitz array for the gradient along the x-axis
   * @param dy The sample distance along the y-axis
   * @param dx The sample distance along the x-axis
   * @warning For unsigned input datatypes, this function might not work as expected.
   */
  template <typename T, typename U>
    void gradient(const blitz::Array<T,2>& input, blitz::Array<U,2>& gy,
        blitz::Array<U,2>& gx, const double dy=1., const double dx=1.)
    {
      // Check input size
      bob::core::array::assertSameShape(input, gy);
      bob::core::array::assertSameShape(input, gx);
      gradient_<T,U>(input, gy, gx, dy, dx);
    }

  /**
   * @brief Function which computes the gradient of a 3D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param gz The output blitz array for the gradient along the z-axis
   * @param gy The output blitz array for the gradient along the y-axis
   * @param gx The output blitz array for the gradient along the x-axis
   * @param dz The sample distance along the z-axis
   * @param dy The sample distance along the y-axis
   * @param dx The sample distance along the x-axis
   * @warning Does not check that gx and gy have the same size as input
   * @warning For unsigned input datatypes, this function might not work as expected.
   */
  template <typename T, typename U>
    void gradient_(const blitz::Array<T,3>& input, blitz::Array<U,3>& gz,
        blitz::Array<U,3>& gy, blitz::Array<U,3>& gx, const double dz=1.,
        const double dy=1., const double dx=1.)
    {
      const int M=input.extent(0);
      const int N=input.extent(1);
      const int P=input.extent(2);
      // Check input
      if (M<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 0 % M;
        throw std::runtime_error(m.str());
      }
      if (N<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 1 % N;
        throw std::runtime_error(m.str());
      }
      if (P<2) {
        boost::format m("the dimension %d is of length %d, strictly smaller than 2 - no gradient can be computed");
        m % 2 % P;
        throw std::runtime_error(m.str());
      }
      if (!(dz>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dz % 0;
        throw std::runtime_error(m.str());
      }
      if (!(dy>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dy % 1;
        throw std::runtime_error(m.str());
      }
      if (!(dx>0.)) {
        boost::format m("the sample distance %f for dimension %d is NOT strictly positive - no gradient can be computed");
        m % dx % 2;
        throw std::runtime_error(m.str());
      }
      bob::core::array::assertZeroBase(input);
      bob::core::array::assertZeroBase(gz);
      bob::core::array::assertZeroBase(gy);
      bob::core::array::assertZeroBase(gx);

      // Defines 'full' range
      blitz::Range rall = blitz::Range::all();

      // Uncentered gradient at the boundaries
      gz(0,rall,rall) = input(1,rall,rall) - input(0,rall,rall);
      gz(M-1,rall,rall) = input(M-1,rall,rall) - input(M-2,rall,rall);
      gy(rall,0,rall) = input(rall,1,rall) - input(rall,0,rall);
      gy(rall,N-1,rall) = input(rall,N-1,rall) - input(rall,N-2,rall);
      gx(rall,rall,0) = input(rall,rall,1) - input(rall,rall,0);
      gx(rall,rall,P-1) = input(rall,rall,P-1) - input(rall,rall,P-2);

      // Centered gradient otherwise
      if (M>2)
      {
        blitz::Range rz(1,M-2);
        blitz::Range rzp(2,M-1);
        blitz::Range rzm(0,M-3);
        gz(rz,rall,rall) = input(rzp,rall,rall) - input(rzm,rall,rall);
        gz(rz,rall,rall) /= 2.;
      }
      if (N>2)
      {
        blitz::Range ry(1,N-2);
        blitz::Range ryp(2,N-1);
        blitz::Range rym(0,N-3);
        gy(rall,ry,rall) = input(rall,ryp,rall) - input(rall,rym,rall);
        gy(rall,ry,rall) /= 2.;
      }
      if (P>2)
      {
        blitz::Range rx(1,P-2);
        blitz::Range rxp(2,P-1);
        blitz::Range rxm(0,P-3);
        gx(rall,rall,rx) = input(rall,rall,rxp) - input(rall,rall,rxm);
        gx(rall,rall,rx) /= 2.;
      }

      // Update scaling if required
      if (dz!=1.) gz *= (1./dz);
      if (dy!=1.) gy *= (1./dy);
      if (dx!=1.) gx *= (1./dx);
    }
  /**
   * @brief Function which computes the gradient of a 3D signal
   *   The gradient is computed using central differences in the interior
   *   and first differences at the boundaries.
   *   Similar to NumPy and MATLAB gradient function
   * @param input The input blitz array
   * @param gz The output blitz array for the gradient along the z-axis
   * @param gy The output blitz array for the gradient along the y-axis
   * @param gx The output blitz array for the gradient along the x-axis
   * @param dz The sample distance along the z-axis
   * @param dy The sample distance along the y-axis
   * @param dx The sample distance along the x-axis
   * @warning For unsigned input datatypes, this function might not work as expected.
   */
  template <typename T, typename U>
    void gradient(const blitz::Array<T,3>& input, blitz::Array<U,3>& gz,
        blitz::Array<U,3>& gy, blitz::Array<U,3>& gx, const double dz=1.,
        const double dy=1., const double dx=1.)
    {
      // Check input size
      bob::core::array::assertSameShape(input, gz);
      bob::core::array::assertSameShape(input, gy);
      bob::core::array::assertSameShape(input, gx);
      gradient_<T,U>(input, gz, gy, gx, dz, dy, dx);
    }

}}

#endif /* BOB_MATH_GRADIENT_H */
