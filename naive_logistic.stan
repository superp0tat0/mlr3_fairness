//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=0> N;
  int<lower=0> R;
  int<lower = 0, upper = 1> y[N];
  int<lower=1, upper=R> race[N];
  vector[N] above_45;
  vector[N] below_25;
  vector[N] misdemeanor;
  vector[N] priors;
  vector[N] gender;
}

// The parameters accepted by the model.
parameters {
  real beta_0;
  real beta[6];
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  beta_0 ~ normal(0,1);
  beta ~ normal(0,1);
  
  for(n in 1:N){
    y ~ bernoulli_logit(beta_0 + race[n] * beta[1] 
                                + above_45[n] * beta[2]
                                + below_25[n] * beta[3]
                                + misdemeanor[n] * beta[4]
                                + priors[n] * beta[5]
                                + gender[n] * beta[6]);
  }
}

