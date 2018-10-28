# -*- coding: utf-8 -*-


import datetime
import numpy as np
import pandas as pd

def pre_process_input(Stock, Exercise_Price, Interest_rate, Volatility, Yield_rate, Value_date, Expiration_date):
    
    S = float(Stock)
    K = float(Exercise_Price)
    r = float(Interest_rate)/100
    sigma = float(Volatility)/100
    q = float(Yield_rate)/100
    T = (Expiration_date-Value_date).days/365
    
    return S, K, r, sigma, q, T

def initialization_parameters(T, K):

    N = 50
    M = 50

    Smax = 2*K
    deltaT = T/N
    deltaS = Smax/M
    
    return N, M, Smax, deltaT, deltaS

def obtain_matrix_a(deltaT, sigma, r, q, M):

    aj = [((1/2)*(deltaT)*((sigma**2)*(j**2) - (r-q)*j)) for j in range(1,M)]
    bj = [(1-(deltaT)*((sigma**2)*(j**2) + r)) for j in range(1,M)]
    cj = [((1/2)*(deltaT)*((sigma**2)*(j**2) + (r-q)*j)) for j in range(1,M)]

    #creating A
    a = np.zeros((M+1,M+1))
    a[0,0], a[M,M]=1,1
    for j in range(1,M):
        a[j,j-1:j+2]=aj[j-1], bj[j-1], cj[j-1]

    return a

def obtain_forward_matrices(N, M, deltaS, Smax, K, r, deltaT, a):

    fc,fp = np.zeros((N+1,M+1)), np.zeros((N+1,M+1))

    for j in range(M+1):
        fc[N][j]=np.maximum(j*deltaS - K,0)
        fp[N][j]=np.maximum(K - j*deltaS,0)

    for i in range(N-1,0-1,-1):
        fc[i]=np.dot(a,fc[i+1])
        fp[i]=np.dot(a,fp[i+1])
        fc[i][0],fp[i][M]= 0,0
        fc[i][M],fp[i][0]=(Smax - K*(np.exp(-r*(N-i)*deltaT))), (K*(np.exp(-r*(N-i)*deltaT)))

    return fc, fp

def obtain_call_put_option(S, deltaS, fc, fp):

    k = np.int32(np.floor(S/deltaS))
    calloption, putoption = [],[]
    calloption.append(fc[0][k]+((fc[0][k+1]-fc[0][k])/deltaS)*(S-k*deltaS))
    putoption.append(fp[0][k]+((fp[0][k+1]-fp[0][k])/deltaS)*(S-k*deltaS))

    return calloption, putoption

def option_price_calculator(
    Stock = 50,
    Exercise_Price = 50,
    Interest_rate = 4,
    Volatility = 40,
    Yield_rate = 1,
    Value_date = datetime.date(2018, 10, 24),
    Expiration_date = datetime.date(2018, 12, 1)
    ):

    S, K, r, sigma, q, T = pre_process_input(Stock, Exercise_Price, Interest_rate, Volatility, Yield_rate, Value_date, Expiration_date)
    
    N, M, Smax, deltaT, deltaS = initialization_parameters(T, K)
    
    a = obtain_matrix_a(deltaT, sigma, r, q, M)
    
    fc, fp = obtain_forward_matrices(N, M, deltaS, Smax, K, r, deltaT, a)
    
    calloption, putoption = obtain_call_put_option(S, deltaS, fc, fp)

    print(calloption, putoption)

option_price_calculator()