#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:20:44 2018

@author: YY
"""

#crank nicolson

import datetime
import numpy as np
import pandas as pd
import math #recently added
import scipy.stats as stats #recently added

Stock = 50
Exercise_Price = 50
Interest_rate = 4 # percent
Volatility = 40 # percent
Yield_rate = 1 # percent
Value_date = datetime.date(2018,10,24)
Expiration_date = datetime.date(2018,12,1)

S = float(Stock)
K = float(Exercise_Price)
r = float(Interest_rate)/100
sigma = float(Volatility)/100
q = float(Yield_rate)/100
T = (Expiration_date-Value_date).days/365

N = 50
M = 50

Smax = 2*K
deltaT = T/N
deltaS = Smax/M

aj = [((1/4)*(deltaT)*((sigma**2)*(j**2) - (r-q)*j)) for j in range(1,M)]
bj = [(-(1/2)*(deltaT)*((sigma**2)*(j**2) + r)) for j in range(1,M)]
cj = [((1/4)*(deltaT)*((sigma**2)*(j**2) + (r-q)*j)) for j in range(1,M)]

#creating M2   
M2 = np.zeros((M+1,M+1))
M2[0,0], M2[M,M]=1,1
for j in range(1,M):
    M2[j,j-1:j+2]=aj[j-1], 1+bj[j-1], cj[j-1]
    
#creating M1
M1 = np.zeros((M+1,M+1))
M1[0,0], M1[M,M]=1,1
for j in range(1,M):
    M1[j,j-1:j+2]=-aj[j-1], 1-bj[j-1], -cj[j-1]

#fNj matrix for call and put options
fc,fp=np.zeros((N+1,M+1)), np.zeros((N+1,M+1))
for j in range(M+1):
    fc[N][j]=np.maximum(j*deltaS - K,0)
    fp[N][j]=np.maximum(K - j*deltaS,0)

#matrix multiplication to get a matrix of b
for i in range(N-1,0-1,-1):
    bc , bp = [] , []
    # 3.1 create b
    bc = np.dot(M2,fc[i+1])
    bp = np.dot(M2,fp[i+1]) 
    
    # 3.2 modify b
    bc[0],bp[M] = 0,0 
    bc[M],bp[0] = (Smax - K*(np.exp(-r*(N-i)*deltaT))), (K*(np.exp(-r*(N-i)*deltaT))) #modify b
    
    # 3.3 compute F
    fc[i], fp[i] = np.dot(np.linalg.inv(M1),bc), np.dot(np.linalg.inv(M1),bp)

k=np.int32(np.floor(S/deltaS))
calloption, putoption=[],[]

calloption.append(fc[0][k]+((fc[0][k+1]-fc[0][k])/deltaS)*(S-k*deltaS))
putoption.append(fp[0][k]+((fp[0][k+1]-fp[0][k])/deltaS)*(S-k*deltaS))

print(calloption, putoption)

#greeks calculation

d1 = (math.log(S/K)+(r-q+(sigma**2)/2)*(T))/(sigma*math.sqrt(T))
d2 = d1-sigma*math.sqrt(T)
print(d1,d2)

#delta
delta_call, delta_put = math.exp(-q*T)*stats.norm.cdf(d1), -math.exp(-q*T)*stats.norm.cdf(-d1)

#delta 100
delta_call100, delta_put100 = delta_call*100, delta_put*100

#lambda
lambda_call, lambda_put = delta_call*S/calloption, delta_put*S/putoption

#gamma
gamma_call = gamma_put = math.exp(-q*T)*stats.norm.pdf(d1)/(S*sigma*math.sqrt(T))

#vega
vega_call = vega_put = S*math.exp(-q*T)*stats.norm.pdf(d1)*math.sqrt(T)/100

#rho
rho_call = K*T*math.exp(-r*T)*stats.norm.cdf(d2)/100
rho_put = -K*T*math.exp(-r*T)*stats.norm.cdf(-d2)/100

#Psi
psi_call = (-S*T*math.exp(-q*T)*stats.norm.cdf(d1)-
            S*math.sqrt(T)*math.exp(-q*T)*stats.norm.pdf(-d1)/sigma +
            K*math.sqrt(T)*math.exp(-r*T)*stats.norm.pdf(-d2)/sigma)/100
psi_put = (S*T*math.exp(-q*T)*stats.norm.cdf(-d1)-
            S*math.sqrt(T)*math.exp(-q*T)*stats.norm.pdf(-d1)/sigma +
            K*math.sqrt(T)*math.exp(-r*T)*stats.norm.pdf(-d2)/sigma)/100

#strike sensitivity
SSc, SSp = (calloption-S*delta_call)/K, (putoption-S*delta_put)/K

#intrinsic value
IVc, IVp = np.maximum(S-K,0), np.maximum(K-S,0)

#time value
TVc, TVp = calloption-IVc, putoption-IVp

#zero volatility
ZVc, ZVp = np.maximum(S*math.exp((r-q)*T)-K,0)*math.exp(-r*T), np.maximum(K-S*math.exp((r-q)*T),0)*math.exp(-r*T)
print(ZVc, ZVp)
