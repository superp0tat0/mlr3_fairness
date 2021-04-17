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
  real<lower=0> sigma_y;
  vector[R] eta_race;
  vector<lower=0>[R] sigma_race;
  vector[R] mu_race;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  sigma_race ~ normal(0,1);
  mu_race ~ normal(0,1);
  beta ~ normal(0,1);
  
  for(r in 1:R){
    eta_race[r] ~ normal(mu_race[r], sigma_race[r]);
  }
  
  for(i in 1:N){
    y[i] ~ bernoulli_logit(beta_0 + eta_race[race[i]] * beta[1] 
                                + above_45 * beta[2]
                                + below_25 * beta[3]
                                + misdemeanor * beta[4]
                                + priors * beta[5]
                                + gender * beta[6]);
  }
}

