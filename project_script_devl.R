# Stochastic Processes, STAT551. Yale University
# Final Project:
# Hidden Markov Model - Stock Return Forecasting
# Daniel Snyder, Yibei Chen
# Due: 5/4/2017

# TODO: 9:47 PM 4/30/2017
# Add a comparison to ARMA (autoregressive moving average)
# Done.
#
# Show log daily returns bar plot: both the actual
#   return and the prior day's prediction.
#
# Simulate a stock betting strategy using the forecasts 
#   and determine if such a strategy has positive 
#   expected return over a finite time period.
# Done. 8:15 AM 5/1/2017
#
# Break out the betting strategy as a separate function. Yeah. Do it.
# Done.
#
# Plot conditional distributions of Y given X (see script chunk at bottom of file)
#


rm(list=ls())
setwd("~/Documents/STAT551/Project/")
library(HiddenMarkov)
library(expm)
library(ggplot2)


# --- Pre Processing ----------------------------

pre.process <- function(f.name){
  # Read a csv file of stock price data
  #   having columns: "open", "close", "date"
  # param f.name: string denoting file name to load
  x <- read.csv(f.name)
  x <- x[order(as.POSIXct(x$date)),] # Sort by date
  x$logdiff <- log(x$close) - log(x$open)
  return(x)
}


# --- HMM Fitting -------------------------------

bwelch <- function(hmm){
  # Perform Baum Welch algorithm 
  #   for MLE of model parameters
  # param hmm: dthmm {HiddenMarkov} object
  hmm <- BaumWelch(hmm, control=bwcontrol(prt=F))
  return(hmm)
}


get.hmm <- function(x, Tr, s=2){
  # Instantiate HMM object
  # param x: data frame with column 'logdiff'
  # param Tr: Truncation point 
  #   (index of last observation included in training set)
  # params s: number of hidden Markov States
  Tr <- min(Tr, nrow(x))
  v <- x[1:Tr, "logdiff"]
  k <- 30.
  p <- 1 / k; q <- 1 - p
  A <- matrix(q, s, s)  # Initial guess at Markov Transition Matrix
  diag(A) <- p
  xi <- matrix(1 / s, 1, s)
  m1 <- mean(v[v < 0]); m2 <- mean(v[v >= 0]);
  s1 <- sd(v[v < 0]); s2 <- sd(v[v >= 0]);
  means <- c(m1,m2)
  stds <- c(s1,s2)
  cond.dist <- list(mean = means, 
                      sd = stds)
  hmm <- dthmm(x = v, Pi = A, delta = xi, 
             distn = 'norm', pm = cond.dist)
  hmm <- bwelch(hmm)
  return(hmm)
}


update.hmm <- function(hmm, xnew){
  # Update the Hidden Markov model with new observation. 
  # Re-solve for parameters using Baum-Welch.
  # param hmm: dthmm {HiddenMarkov} object
  # param xnew: float. New observation to append
  hmm$x <- c(hmm$x, xnew)
  hmm <- bwelch(hmm)
  return(hmm)
}


forecast.hmm <- function(hmm, horizon=1){
  # Forecast tomorrow's return (log dollars)
  # param hmm: dthmm{HiddenMarkov} object
  # param horizon: int. how many steps ahead to forecast
  A <- hmm$Pi  # MLE of Markov Chain Transition Matrix
  Eyx <- hmm$pm$mean  # Expectation of Y given X
  t <- length(hmm$x)  # Series length
  px.T0 <- hmm$u[t,]  # Estimated distribution of Xt
  Ah <- A %^% horizon    # Exponentiate MC transition matrix
  px.Th <- px.T0 %*% Ah  # Estimated distribution of X(t+h)
  Ey.Th <- px.Th %*% Eyx  # Expectation of Y(t+h)
  return(Ey.Th)
}


# --- Simulation Methods ------------------------

employ.trading.strategy <- function(x){
  # Investing rule given the predicted next day returns
  # param x: data frame containing column 'predicted'
  # Creates column "position" representing the number
  #   of shares that a simulated trader would own using
  #   a this trading strategy
  N <- nrow(x)
  x$position <- 0 + (x$predicted > 0) - (x$predicted < 0)
  return(x)
}


overall.return <- function(x){
  # Compute the overall return from 
  #   using the HMM trading strategy
  # param x: data frame containing 
  #   columns "open", "close", and "position"
  N <- nrow(x)
  start_ <- min(which(!is.na(x$predicted)))
  D <- 0                  # Initial cash amount
  holdings <- x$position  # How many shares owned, by day
  price <- x$open         # Actual open price
  for(t in start_:N){
    shares <- holdings[t] - holdings[t - 1]
    D <- D - shares * price[t]
  }
  return(D)
}


sequentially.forecast.hmm <- function(x, days, sym){
  # Forecast the log returns for day t using information
  # contained only in days 1 through t - 1 
  # param x: dataframe with column <column="logdiff">
  # param sym: string denoting Stock ticker symbol
  N <- nrow(x)
  Tr <- N - days  # Truncation point
  x$predicted <- rep(NA, N)
  for(t in Tr:(N - 1)){
    if(t == Tr){
      hmm <- get.hmm(x, Tr = t)
    } else {
      xnew <- x$logdiff[t]
      hmm <- update.hmm(hmm, xnew)
    }
    x$predicted[t + 1] <- forecast.hmm(hmm)
  }
  x <- employ.trading.strategy(x)
  return(x)
}


sequentially.forecast.arma <- function(x, days, k=30){
  # Sequentially forecast the next day's returns using 
  #   an moving average of the last k days.
  # param x: data frame with column "logdiff"
  # param days: int. How many days to forecast
  # param k: int. How many previous days to use 
  #   in moving average
  N <- nrow(x)
  Tr <- N - days
  x$predicted <- rep(NA, N)
  for(t in Tr:(N - 1)){
    x$predicted[t + 1] <- predict(arima(x$logdiff[1:t], 
                                        c(1,1,1)), 1)
  }
  x <- employ.trading.strategy(x)
  return(x)
}


forecast.scatter <- function(df){
  # Summarize the sequential forecasting
  #   using a actual by predicted plot
  # param df: dataframe with columns "logdiff" and "predicted"
  x <- df$predicted
  y <- df$logdiff
  x <- x[!is.na(x)]
  y <- y[!is.na(x)]
  axis.min <- min(min(x), min(y))
  axis.max <- max(max(x), max(y))
  ax.lim <- c(axis.min, 
              axis.max)
  plot(logdiff~predicted, df,
       xlim=ax.lim, ylim=ax.lim,
       xlab="Predicted Log Returns",
       ylab="Actual Log Returns")
  abline(0,1)
}


prediction.rsq <- function(x){
  # Calculate squared correlation between 
  #   predictions and actual returns
  y <- x$logdiff
  z <- x$predicted
  y <- y[!is.na(z)]
  z <- z[!is.na(z)]
  rsq <- cor(y, z) ^ 2
  return(rsq)
}


plot.forecast.timeseries <- function(x){
  # Plot the time series of the actual log daily returns
  # as well as the predictions based on prior days.
  # param x: data frame containing columns 
  #   "logdiff", "predicted", "date"
  g <- ggplot(mapping=aes(x=date, y=logdiff), data=x) + geom_col()
  g
}


plot.timeseries <- function(x, sym){
  # Plot time series of log returns along with predictions
  # param x: dataframe with column "logdiff"
  # param sym: string - Stock ticker symbol
  main = sprintf("Log Returns: %s", sym)
  plot(logdiff ~ as.POSIXct(date), x, 
       type="l", col="blue", xlab="Date", 
       ylab="Log Dollars", main=main)
  abline(0,0)
  points(predicted ~ as.POSIXct(date), x, col="green")
}


plot.conditional.dist <- function(hmm){
  # Plot the conditional distributions
  #   of Y given hidden state X
  # param hmm: dthmm {HiddenMarkov} object
  m1 <- hmm$pm$mean[1]
  m2 <- hmm$pm$mean[2]
  s1 <- hmm$pm$sd[1]
  s2 <- hmm$pm$sd[2]
  xlim <- c(min(m1 - 3  * s1,
                m2 - 3  * s2),
            max(m1 + 3  * s1,
                m2 + 3  * m2))
  x <- seq(xlim[1],xlim[2],.001)
  y1 <- dnorm(x, m1, s1)
  y2 <- dnorm(x, m2, s2)
  ymax <- max(max(y1), max(y2))
  plot(x, y1, type="l", col="red",
       ylim=c(0,ymax), main="Conditional Distribution: P[Y|X]",
       ylab="Probability Density", xlab="y")
  abline(v = m1, col='red')
  lines(x, y2, col="blue")
  abline(v = m2, col="blue")
  legend(xlim[1], ymax / 2, c('X=1', 'X=2'), lty=c(1,1),
         lwd=c(2,2), col=c('red','blue'))
}


analyze <- function(f.name, sym, method="hmm", days=332){
  # param f.name: name of csv file containing 
  #   stock price data
  # param method: string in {"hmm", "arma"}
  x <- pre.process(f.name)
  x.hmm <- sequentially.forecast.hmm(x, days=days, sym=sym)
  x.arma <- sequentially.forecast.arma(x, days=days, sym=sym)
  r.hmm <- prediction.rsq(x.hmm)
  r.arma <- prediction.rsq(x.arma)
  print(sprintf("HMM R^2: %1.4f, ARMA R^2: %1.4f", 
                r.hmm, r.arma))
  overall.return(x.hmm)
  overall.return(x.arma)
}


# Read csv files and sort datewise
# Use business day "date", "open", "close"
# To create "logdiff" column
analyze("./stock_data/utx_decade.csv")
analyze("./stock_data/ford_decade.csv")
analyze("./stock_data/aapl_decade.csv")
analyze("./stock_data/msft_decade.csv")
analyze("./stock_data/sbux_decade.csv")


# # See what the conditional distributions of Y look like in a case
# hmm <- get.hmm(ford, Tr=2400)
# xlim <- c(min(hmm$pm$mean[1] - 3  * hmm$pm$sd[1], hmm$pm$mean[2] - 3  * hmm$pm$sd[1]),
#           max(hmm$pm$mean[1] + 3  * hmm$pm$sd[1], hmm$pm$mean[2] + 3  * hmm$pm$sd[1]))
# 
# x <- seq(xlim[1],xlim[2],.001)
# y1 <- dnorm(x, hmm$pm$mean[1], hmm$pm$sd[1])
# y2 <- dnorm(x, hmm$pm$mean[2], hmm$pm$sd[2])
# ymax <- max(max(y1), max(y2))
# plot(x,y1, type="l", 
#      col="red", ylim=c(0,ymax))
# abline(v=hmm$pm$mean[1], col='red')
# lines(x,y2, col="blue")
# abline(v=hmm$pm$mean[2], col="blue")









