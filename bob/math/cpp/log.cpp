/**
 * @date Fri Feb 10 20:02:07 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <stdexcept>
#include <boost/format.hpp>

#include <bob.math/log.h>

/**
 * Computes log(a+b)=log(exp(log(a))+exp(log(b))) from log(a) and log(b),
 * while dealing with numerical issues
 */
double bob::math::Log::logAdd(double log_a, double log_b)
{
  if(log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }

  double minusdif = log_b - log_a;
  //#ifdef DEBUG
  if(std::isnan(minusdif))
  {
    boost::format m("logadd: minusdif (%f) log_b (%f) or log_a (%f) is nan");
    m % minusdif % log_b % log_a;
    throw std::runtime_error(m.str());
  }
  //#endif
  if(minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(exp(minusdif));
}

/**
 * Computes log(a-b)=log(exp(log(a))-exp(log(b))) from log(a) and log(b),
 * while dealing with numerical issues
 */
double bob::math::Log::logSub(double log_a, double log_b)
{
  double minusdif;

  if(log_a < log_b)
  {
    boost::format m("logsub: log_a (%f) should be greater than log_b(%f)");
    m % log_a % log_b;
    throw std::runtime_error(m.str());
  }

  minusdif = log_b - log_a;
  //#ifdef DEBUG
  if(std::isnan(minusdif))
  {
    boost::format m("logsub: minusdif (%f) log_b (%f) or log_a (%f) is nan");
    m % minusdif % log_b % log_a;
    throw std::runtime_error(m.str());
  }
  //#endif
  if(log_a == log_b) return bob::math::Log::LogZero;
  else if(minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(-exp(minusdif));
}
