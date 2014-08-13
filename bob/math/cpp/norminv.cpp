/**
 * @date Tue Apr 12 21:33:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the inverse normal cumulative distribution
 *   function
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <cmath>
#include <stdexcept>
#include <boost/format.hpp>

#include <bob.math/norminv.h>

double bob::math::norminv(const double p, const double mu, const double sigma)
{
  // Take the mean and sigma (standard deviation) into account
  return sigma * bob::math::normsinv(p) + mu;
}

double bob::math::normsinv(const double p)
{
  // Coefficients in rational approximations
  static const double a1 = -3.969683028665376e+01;
  static const double a2 =  2.209460984245205e+02;
  static const double a3 = -2.759285104469687e+02;
  static const double a4 =  1.383577518672690e+02;
  static const double a5 = -3.066479806614716e+01;
  static const double a6 =  2.506628277459239e+00;

  static const double b1 = -5.447609879822406e+01;
  static const double b2 =  1.615858368580409e+02;
  static const double b3 = -1.556989798598866e+02;
  static const double b4 =  6.680131188771972e+01;
  static const double b5 = -1.328068155288572e+01;

  static const double c1 = -7.784894002430293e-03;
  static const double c2 = -3.223964580411365e-01;
  static const double c3 = -2.400758277161838e+00;
  static const double c4 = -2.549732539343734e+00;
  static const double c5 =  4.374664141464968e+00;
  static const double c6 =  2.938163982698783e+00;

  static const double d1 =  7.784695709041462e-03;
  static const double d2 =  3.224671290700398e-01;
  static const double d3 =  2.445134137142996e+00;
  static const double d4 =  3.754408661907416e+00;

  // Define break-points
  static const double p_low =  0.02425;
  static const double p_high = 1 - p_low;

  // Declare output value
  double x = 0.;

  // Error p should be between 0 and 1
  if (p < 0. || p > 1.) {
    boost::format m("invalid value for parameter `p' (%f) - it should be reside the interval [0.,1.]");
    m % p;
    throw std::runtime_error(m.str());
  }
  // Rational approximation for lower region.
  else if (0. < p && p < p_low)
  {
    double q = sqrt(-2*log(p));
    x =  ( ( ( ( (c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ( ( ( (d1*q+d2)*q+d3)*q+d4)*q+1);
  }
  // Rational approximation for central region.
  else if (p_low <= p && p <= p_high)
  {
    double q = p - 0.5;
    double r = q*q;
    x = ( ( ( ( (a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / ( ( ( ( (b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
  }
  // Rational approximation for upper region.
  else if (p_high < p && p < 1.)
  {
    double q = sqrt(-2*log(1-p));
    x = -( ( ( ( (c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ( ( ( (d1*q+d2)*q+d3)*q+d4)*q+1);
  }

  // This block just improves accuracy and can eventually be commented.
  // The relative error of the approximation has  absolute value less than
  // 1.15 × 10−9.  One iteration of Halley’s rational method (third order)
  // gives full machine precision.
  double e = 0.5 * erfc(-x/sqrt(2)) - p;
  double u = e * sqrt(2*M_PI) * exp(x*x/2);
  x = x - u/(1 + x*u/2);

  return x;
}

