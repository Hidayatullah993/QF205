#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:41:32 2018

@author: YY
"""

import datetime
import numpy as np
import pandas as pd
import math #recently added
import scipy.stats as stats #recently added

def pre_process_input(
    Stock, 
    Exercise_Price, 
    Interest_rate, 
    Volatility, 
    Yield_rate, 
    Expiration_date, 
    Value_date
):
    S = float(Stock)
    K = float(Exercise_Price)
    r = float(Interest_rate)/100
    sigma = float(Volatility)/100
    q = float(Yield_rate)/100
    T = (Expiration_date - Value_date).days/365
    
    return S, K, r, sigma, q, T

def initialization_parameters(T, K):

    N = 50
    M = 50

    Smax = 2*K
    deltaT = T/N
    deltaS = Smax/M
    
    return N, M, Smax, deltaT, deltaS

def obtain_matrix_a(deltaT, sigma, r, q, M, algo = 'explicit'):

    if algo == 'explicit':
        aj = [((1/2)*(deltaT)*((sigma**2)*(j**2) - (r-q)*j)) for j in range(1,M)]
        bj = [(1-(deltaT)*((sigma**2)*(j**2) + r)) for j in range(1,M)]
        cj = [((1/2)*(deltaT)*((sigma**2)*(j**2) + (r-q)*j)) for j in range(1,M)]

    elif algo == 'implicit':
        aj = [((1/2)*(deltaT)*(((r-q)*j) - (sigma ** 2) * (j ** 2))) for j in range(1,M)]
        bj = [(1 + (deltaT) * ((sigma ** 2)*(j ** 2) + r)) for j in range(1,M)]
        cj = [(-(1/2) * (deltaT) * ((sigma ** 2)*(j ** 2) + (r - q) * j)) for j in range(1,M)]        
    
    elif algo == 'crank nicolson':
        aj = [((1/4)*(deltaT)*((sigma**2)*(j**2) - (r-q)*j)) for j in range(1,M)]
        bj = [(-(1/2)*(deltaT)*((sigma**2)*(j**2) + r)) for j in range(1,M)]
        cj = [((1/4)*(deltaT)*((sigma**2)*(j**2) + (r-q)*j)) for j in range(1,M)]
    
    #creating A
    a = np.zeros((M+1,M+1))
    a[0,0], a[M,M]=1,1
    for j in range(1,M):
        a[j,j-1:j+2]=aj[j-1], bj[j-1], cj[j-1]
    
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
    
    return a, M1, M2

def obtain_forward_matrices(N, M, deltaS, Smax, K, r, deltaT, a, M1, M2, algorithm):
    fc, fp = np.zeros((N+1,M+1)), np.zeros((N+1,M+1))
    for j in range(M+1):
        fc[N][j] = np.maximum(j*deltaS - K,0)
        fp[N][j] = np.maximum(K - j*deltaS,0)

    if algorithm == 'explicit':
        
        for i in range(N - 1, 0 - 1, - 1):
            fc[i] = np.dot(a,fc[i+1])
            fp[i] = np.dot(a,fp[i+1])
            fc[i][0], fp[i][M] = 0, 0
            fc[i][M], fp[i][0] = (Smax - K*(np.exp(-r*(N-i)*deltaT))), (K*(np.exp(-r*(N-i)*deltaT)))

    elif algorithm == 'implicit':

        for i in range(N - 1,0 - 1, -1):
            fc[i + 1][0], fp[i + 1][M] = 0, 0
            fc[i + 1][M], fp[i + 1][0] = (Smax - K*(np.exp(-r*(N-i)*deltaT))), (K*(np.exp(-r*(N-i)*deltaT)))
            
            # TODO: Inverse required ++
            
            fc[i] = np.dot(np.linalg.inv(a),fc[i+1])
            fp[i] = np.dot(np.linalg.inv(a),fp[i+1])
            
    elif algorithm == 'crank nicolson':

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

    return fc, fp

def obtain_call_put_option(S, deltaS, fc, fp):

    k = np.int32(np.floor(S/deltaS))
    calloption, putoption = [],[]
    calloption.append(fc[0][k]+((fc[0][k+1]-fc[0][k])/deltaS)*(S-k*deltaS))
    putoption.append(fp[0][k]+((fp[0][k+1]-fp[0][k])/deltaS)*(S-k*deltaS))

    return calloption, putoption

def option_price_calculator(
    Stock,
    Exercise_Price,
    Interest_rate,
    Volatility,
    Yield_rate,
    T,
    algorithm,
    type
    ):

    # NOTE values should already be pre-processed by the get_all_values() method prior to calling this method.
    # S, K, r, sigma, q, T = pre_process_input(Stock, Exercise_Price, Interest_rate, Volatility, Yield_rate, T)
        
    N, M, Smax, deltaT, deltaS = initialization_parameters(T, Exercise_Price)
    
    a, M1, M2 = obtain_matrix_a(deltaT, Volatility, Interest_rate, Yield_rate, M, algorithm)

    fc, fp = obtain_forward_matrices(N, M, deltaS, Smax, Exercise_Price, Interest_rate, deltaT, a, M1, M2, algorithm)
    
    calloption, putoption = obtain_call_put_option(Stock, deltaS, fc, fp)

    if type == 'call':
        return calloption
    elif type == 'put':
        return putoption

def delta_calculation(S, K, r, q, T, sigma, optiontype):
    #have to reassign d1 and d2 for new S values
    d1gamma = (math.log(S/K)+(r-q+(sigma**2)/2)*(T))/(sigma*math.sqrt(T))
    d2gamma = d1gamma-sigma*math.sqrt(T)
    delta_call, delta_put = math.exp(-q*T)*stats.norm.cdf(d1gamma), -math.exp(-q*T)*stats.norm.cdf(-d1gamma)
    if optiontype == 'call':
        return delta_call 
    elif optiontype == 'put':
        return delta_put

 # this method should called by the GUI side        
def get_all_values(
    Stock,
    Exercise_Price,
    Interest_rate,
    Volatility,
    Yield_rate,
    Expiration_date,
    Value_date,
    algorithm,
    option_type
):    

    # S = 50
    # K = 50
    # r = 4
    # sigma = 40
    # q = 1
    # Value_date = datetime.date(2018, 10, 24)
    # Expiration_date = datetime.date(2018, 12, 1)
    # algo = 'crank nicolson'
    # option_type = 'call'

    #process data first, so can apply to greeks (or can change the flow later on)

    S, K, r, sigma, q, T = pre_process_input(Stock, Exercise_Price, Interest_rate, Volatility, Yield_rate, Expiration_date, Value_date)

    #need to assign for 
    calloption = option_price_calculator(S, K, r, sigma, q, T, algorithm, 'call')
    putoption = option_price_calculator(S, K, r, sigma, q, T, algorithm, 'put')

    #generate for user to see values (or can show both)
    if option_type == 'call':
        print('option price = '+ str(calloption))
    elif option_type == 'put':
        print('option price = '+ str(putoption))

    #greeks calculation
    d1 = (math.log(S/K)+(r - q + ( sigma ** 2)/2) * (T))/(sigma*math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    #delta

    delta_call, delta_put = delta_calculation(S, K, r, q, T, sigma, 'call'), delta_calculation(S, K, r, q, T, sigma, 'put')

    #delta 100
    delta_call100, delta_put100 = delta_call*100, delta_put*100

    #lambda
    lambda_call, lambda_put = delta_call*S/calloption, delta_put*S/putoption

    #gamma
    gamma_call = gamma_put = math.exp(-q*T)*stats.norm.pdf(d1)/(S*sigma*math.sqrt(T))

    #gamma 1%
    gamma_call_onepercent = (delta_calculation(1.01*S, K, r, q, T, sigma, 'call')-
                            delta_calculation(0.99*S, K, r, q, T, sigma, 'call'))/2
    gamma_put_onepercent = (delta_calculation(1.01*S, K, r, q, T, sigma, 'put')-
                            delta_calculation(0.99*S, K, r, q, T, sigma, 'put'))/2

    #theta 1 day
    theta_call_oneday = (option_price_calculator(S, K, r, sigma, q, T-1/365, algorithm, 'call')[0]-
                option_price_calculator(S, K, r, sigma, q, T, algorithm, 'call')[0])
    theta_put_oneday = (option_price_calculator(S, K, r, sigma, q, T-1/365, algorithm, 'put')[0]-
                        option_price_calculator(S, K, r, sigma, q, T, algorithm, 'put')[0])


    #theta 7 day
    theta_call_sevenday = (option_price_calculator(S, K, r, sigma, q, T-7/365, algorithm, 'call')[0]-
                option_price_calculator(S, K, r, sigma, q, T, algorithm, 'call')[0])
    theta_put_sevenday = (option_price_calculator(S, K, r, sigma, q, T-7/365, algorithm, 'put')[0]-
                        option_price_calculator(S, K, r, sigma, q, T, algorithm, 'put')[0])

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
    TVc, TVp = calloption - IVc, putoption-IVp

    #zero volatility
    ZVc, ZVp = np.maximum(S*math.exp((r-q)*T)-K,0)*math.exp(-r*T), np.maximum(K-S*math.exp((r-q)*T),0)*math.exp(-r*T)

    return {
        'call': {
            'value': calloption[0],
            'delta': delta_call,
            'delta_100': delta_call100,
            'lambda': lambda_call[0],
            'gamma': gamma_call,
            'gamma_1%': gamma_call_onepercent,
            'theta': theta_call_oneday,
            'theta_7d': theta_call_sevenday,
            'vega': vega_call,
            'rho': rho_call,
            'psi': psi_call,
            'strike_sensitivity': SSc[0],
            'intrinsic_value': IVc,
            'time_value': TVc[0],
            'zero_volatility': ZVc,
        },
        'put': {
            'value': putoption[0],
            'delta': delta_put,
            'delta_100': delta_put100,
            'lambda': lambda_put[0],
            'gamma': gamma_put,
            'gamma_1%': gamma_put_onepercent,
            'theta': theta_put_oneday,
            'theta_7d': theta_put_sevenday,
            'vega': vega_put,
            'rho': rho_put,
            'psi': psi_put,
            'strike_sensitivity': SSp[0],
            'intrinsic_value': IVp,
            'time_value': TVp[0],
            'zero_volatility': ZVp,
        }
    }

Stock = 50
Exercise_Price = 50
Interest_rate = 4
Volatility = 40
Yield_rate = 1
Value_date = datetime.date(2011, 1, 1)
Expiration_date = datetime.date(2011, 7, 3)
algorithm = 'crank nicolson'
option_type = 'call'

print(get_all_values(
    Stock,
    Exercise_Price,
    Interest_rate,
    Volatility,
    Yield_rate,
    Expiration_date,
    Value_date,
    algorithm,
    option_type
))