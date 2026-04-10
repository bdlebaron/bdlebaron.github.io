#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to generate all figures for:
Dynamic Order Dispersion and Volatility Persistence in a Simple
Limit Order Book Model

This is the turnkey script that runs all simulations and generates all figures.
"""

import numpy as np
from random import choices
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm
from statsmodels.tsa.arima.model import ARIMA
from time import time
import csv
import os

# Import everything we need from LOBModel
from LOBModel import (
    modelIteration, 
    fastautocorr, 
    fastautocorr1, 
    fastxcorr,
    archAdjust,
    getSummedBids,
    getSummedAsks
)

# Import plotting function from make_plots4
from make_plots4 import plot_four_panel, process_data_file

# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

def get_base_parameters():
    """Base case parameters from Table 1"""
    return {
        'nAgents': 2000,
        'Tinit': 10000,
        'Tmax': 200000,
        'Lmin': 10,
        'Lmax': 500,
        'pf': 1000.,
        'deltaP': 0.025,
        'sigmae': 0.0001,
        'kMax': 0.1,
        'adaptiveK': True,
        'multipleFundamentals': True,
        'simpleDemands': True,
        'maxTrade': 1,
        'tau': 25,
        'sigmaF': 0.10,
        'sigmaM': 0.00,
        'sigmaN': 0.90,
        'portfolioAdj': False,
        'deltaT': 50,
        'lam': 1.00,
        'maxHold': 50.,
        'beta': 3.,
        'volLag': 150,
        'agentsPerPeriod': 30,
        'rhoBar': 0.999,
        'orderSigma': 0.0,
        'trimVol': False,
        'probMarketOrder': 0.05
    }

# ============================================================================
# SIMULATION RUNNERS
# ============================================================================

def run_simulation(run_name, params, seed=4242):
    """Run a single simulation"""
    print(f"\n{'='*70}")
    print(f"Running: {run_name}")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    
    t0 = time()
    results = modelIteration(**params)
    t1 = time()
    
    print(f"Completed in {(t1-t0)/60:.2f} minutes")
    return results

def save_simulation_csv(filename, results, params, seed):
    """Save simulation results to CSV"""
    (price, ret, rret, rRV, rRV2, totalV, rPrice, rSpread, rbidDepth, 
     raskDepth, portDev, rportDev, holdings, dholdings, orders, 
     rportDev_non_nan, Xsmooth_centered_non_nan, Xsmooth_trailing_non_nan,
     holdingsDiff, holdingsDiffABS, holdingsDist, holdingsDistABS, rVol, 
     rOrders, marketBook, agentList, forecastSet, rdholdingsts, bidsdf, 
     rbidSlope, asksdf, raskSlope, wealthByType, rpf, logPriceFund, rCRV,
     rtotalOrdersOnBook, orderFlow, pft) = results
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["price", "ret", "rret", "rRV", "rRV2", "rCRV",
                        "autocorrelation sum", "rvol", "rPrice", "spread",
                        "totalOrders", "bid depth", "bidslope", "ask depth",
                        "askslope", "tau", "sigmaE", "seed", "fundamental"])
        
        for i in range(len(rret)):
            writer.writerow([
                rPrice[i], 0, rret[i], rRV[i], rRV2[i], rCRV[i],
                0.0, rVol[i], rPrice[i], rSpread[i],
                rtotalOrdersOnBook[i], rbidDepth[i], rbidSlope[i],
                raskDepth[i], raskSlope[i], params['tau'], params['sigmae'], 
                seed, pft[i]
            ])
    
    print(f"Saved: {filename}")

# ============================================================================
# PLOTTING FUNCTIONS (using existing code)
# ============================================================================

# vol distribution facts

def create_vol_facts_plot(df, output_file, title="", start=500):
    
    start = 0
    
    rret = df['rret'].values
    rRV  = df['rRV'].values
    
    ret = rret[start:]
    vol = rRV[start:]
    svol = np.sqrt(vol)
    lvol = np.log(vol)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    # Panel 1 : Normalized returns
    normret = ret/svol

    nbins = 50

    n, bins, patches = ax[0].hist(normret, nbins, density=True,facecolor='green', alpha=0.5)
    ax[0].grid()
    mu = np.mean(normret)
    sigma = np.std(normret)
    xmin, xmax = ax[0].get_xlim()

    # Generate normal pdf from scipy.stats
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
      
    ax[0].plot(x, p, 'k', linewidth=2)

    # y = plt.mlab.normpdf(bins,mu,sigma)
    # ax[0, 1].plot(bins, y, 'r')
    ax[0].set_title('Standardized returns, kurtosis = '+str(round(kurtosis(normret, fisher=False, bias=False),1)))
    
    
    # panel 2 : log vol histogram
                               
    nbins = 50

    n, bins, patches = ax[1].hist(lvol, nbins, density=True,facecolor='green', alpha=0.5)
    ax[1].grid()
    mu = np.mean(lvol)
    sigma = np.std(lvol)
    xmin, xmax = ax[1].get_xlim()

    # Generate normal pdf from scipy.stats
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
      
    ax[1].plot(x, p, 'k', linewidth=2)

    # y = plt.mlab.normpdf(bins,mu,sigma)
    # ax[0, 1].plot(bins, y, 'r')
    ax[1].set_title('Log(RV), kurtosis = '+str(round(kurtosis(lvol, fisher=False, bias=False),1))) 
    
    plt.tight_layout()
    plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')
    plt.savefig(output_file+".pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()    
    
    

def create_long_memory_plotold(rRV, output_file, title="", start=500):
    """Create long memory variance scaling plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    start = 500
    rv = rRV[start:]
    log_rv = np.log(rv + 1e-10)
    
    # Variance scaling
    max_agg = min(75, len(log_rv) // 10)
    aggregations = np.arange(1, max_agg)
    variances = np.zeros(len(aggregations))
    
    for i, K in enumerate(aggregations):
        n_blocks = len(log_rv) // K
        block_sums = np.array([np.sum(log_rv[j*K:(j+1)*K]) 
                                for j in range(n_blocks)])
        variances[i] = np.var(block_sums)
    
    log_agg = np.log(aggregations)
    log_var = np.log(variances)
    slope = np.polyfit(log_agg, log_var, 1)[0]
    
    ax1.loglog(aggregations, variances, 'o-', 
               label=f'Slope={slope:.2f}')
    ax1.set_xlabel('Aggregation (Days)')
    ax1.set_ylabel('var(log(RV))_h')
    ax1.set_title('Variance scaling')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Autocorrelations
    max_lag = min(70, len(log_rv) // 3)
    acf = fastautocorr(log_rv, max_lag)
    lags = np.arange(max_lag + 1)
    
    ax2.plot(lags[1:], acf[1:], 'o-')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('Log(RV) autocorrelations')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

# Find non overlapping 1 to k horizon variances of vol estimates
# Part of long memory scaling  (see equation 16 in paper)
def varhorizon(vol,k):
    svol = np.cumsum(vol)
    varh = []
    varn = []
        
    for j in range(1,k):
        dvol = np.diff( svol[::j])
        varh.append(np.var(dvol))
        varn.append(j)
    # make sure return numpy arrays.  I think this routine can data pandas dataframes as args, but not sure
    return np.array(varh), np.array(varn)

def ols(y,x):
    y = y-np.mean(y)
    x = x-np.mean(x)
    vx = np.var(x)
    cvxy = np.mean( x * y)
    beta = cvxy/vx
    return(beta)
    
def create_long_memory_plot(rRV, output_file, title="", start=500):
    
    from random import choices
        
    Delta = 1
    # Start = 25000
    rRV = rRV[start::1]
    # Use log(rv) as is standard
    vol = np.log(rRV)
    
    # generate bootstrap volatility series
    voln = np.array(choices(vol,k=len(vol)+1))
  

    # Range for ACF estimations
    timeRange = 75
    timeRangehalf = 25
    varRange = 75
    xindex =  range(1,varRange+1)
    xindex2 = range(1,timeRange+1)


    # first order autocorrelation
    acf1 = fastautocorr(vol,timeRange)[1:]
    # theoretical ACF for an AR(1) proces
    acf1th = acf1[1]**range(1,timeRange+1)

    # Find summed variances for possible scaling
    varh, varn = varhorizon(vol,varRange)

    # Do the same for bootstrap scrambled sample
    varhn, varnn = varhorizon(voln,varRange)

    # Find slope of log/log regression (this is equal 2H, H=self similarity)
    # Also, d = H-(1/2)
    beta = ols(np.log(varh),np.log(varn))
    betastr = str(round(beta,2))
    leg1 = 'Base case: Slope ='+betastr

    # Same for bootstrap
    betarw = ols(np.log(varhn),np.log(varnn))
    betastr = str(round(betarw,2))
    leg2 = 'IID Bootstrap: Slope ='+betastr


    # Find H and d, and gamma (see equations 17-19)
    H = 0.5*beta
    d = H - 0.5
    gamma = 2-2*H
    # gamma = 0.41

    print("H,d,gamma = ",H,d,gamma)

    # Line up long memory ACF so that it crosses data ACF at timeRangeHalf
    timeRangehalf = 30
    A = acf1[timeRangehalf-1]*timeRangehalf**(gamma)
    # Long memory ACF (see equation 18)
    theoryACF = A * varn**(-gamma)

    # Now get ARMA(1,1) comparison
    arma = ARIMA(vol, order=(1,0,1)).fit()
    print(arma.summary())

    theta = arma.params[2]
    rho   = arma.params[1]
    armaA = (1+rho*theta)*(rho+theta)/(  1+2*rho*theta + theta**2)
    acfARMA = armaA * rho**(varn)


    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    # fig.suptitle('Returns for tau = '+str(tau)+", simgae = "+str(sigmae))
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].plot(varn,varh)
    ax[0].plot(varnn,varhn)
    ax[0].grid()
    ax[0].set_title('Variance scaling')
    ax[0].set_xlabel('Aggregation (Days)')
    ax[0].set_ylabel('$var(\log(RV)_h)$')
    ax[0].legend([leg1,leg2])


      
    ax[1].plot(xindex2,acf1)
    ax[1].plot(xindex2,acf1th)
    ax[1].grid()
    ax[1].set_title("Log(RV) autocorrelations")
    # ax[1].set_xlabel('Lag(j)')
    ax[1].set_ylabel('Autocorrelation')
    ax[1].set_xlabel('Lag')



    ax[1].plot(varn,theoryACF)
    leg1 = 'Base case'
    ax[1].legend([leg1,'AR(1)','Long memory'])

    ax[1].plot(varn,acfARMA)

    ax[1].legend([leg1,'AR(1)','Long memory','ARMA(1,1)'])
    
    plt.tight_layout()
    plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')
    plt.savefig(output_file+".pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()  
    
    

def create_orderbook_dynamics_plot(rret, rSpread, rPrice, rbidSlope, raskSlope,
                                   rbidDepth, raskDepth, output_file,
                                   title="", start=500):
    """Create 4-panel order book dynamics plot"""
    # fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig, axes = plt.subplots(4,1,figsize=(6, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    """
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    """
    returns = rret[start:]
    spread = rSpread[start:]
    spread = 100*spread/rPrice
    slope = 0.5 * (np.abs(rbidSlope[start:]) + np.abs(raskSlope[start:]))
    depth = rbidDepth[start:] + raskDepth[start:]
    days = np.arange(len(returns))
    
    axes[0].plot(days, returns)
    axes[0].set_title("Returns")
    axes[0].set_ylabel('')
    axes[0].grid(True)
    
    axes[1].plot(days, spread)
    axes[1].set_title("Bid/Ask Spread")
    axes[1].set_ylabel('% of price')
    axes[1].grid(True)
    
    axes[2].plot(days, slope)
    axes[2].set_title("Order Book Slope")
    axes[2].grid(True)
    
    axes[3].plot(days, depth)
    axes[3].set_title("Depth")
    axes[3].set_ylabel('Shares')
    axes[3].set_xlabel('Period')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')
    plt.savefig(output_file+".pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_cross_correlation_plot(rret, rRV, rCRV, rSpread, rbidSlope, raskSlope,
                                   rbidDepth, raskDepth, rVol,volatilityXC,
                                   output_file, title="", max_lag=50, start=500):
    """Create cross-correlation plot"""
    abs_ret = np.abs(rret[start:])
    rv = rRV[start:]
    rv = np.sqrt(rv)
    crv = rCRV[start:]
    spread = rSpread[start:]
    slope = 0.5 * (np.abs(rbidSlope[start:]) + np.abs(raskSlope[start:]))
    depth = rbidDepth[start:] + raskDepth[start:]
    volume = rVol[start:]
    
    print(rv[0:10])
    print(spread[0:10])
    
    """
    xcorr_spread = fastxcorr(abs_ret, spread, max_lag)
    xcorr_slope = fastxcorr(abs_ret, slope, max_lag)
    xcorr_depth = fastxcorr(abs_ret, depth, max_lag)
    xcorr_volume = fastxcorr(abs_ret, volume, max_lag)
    """
    
    T = len(rv)
    T2 = int(T/2)
  
    """
    rv = rv[T2:]
    crv = crv[T2:]
    spread = spread[T2:]
    slope  = slope[T2:]
    depth  = depth[T2:]
    volume = volume[T2:]
    """
    
    xcorr_spread = fastxcorr(spread,rv, max_lag)
    xcorr_slope = fastxcorr(slope,rv, max_lag)
    xcorr_depth = fastxcorr(depth,rv, max_lag)
    xcorr_volume = fastxcorr(rv,volume, max_lag)
    xcorr_slopevol = fastxcorr(slope,volume,max_lag)
    xcorr_slopecvar = fastxcorr(slope,crv,max_lag)
    
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    lags = np.arange(-max_lag, max_lag + 1)
    
    if(volatilityXC):
        ax.plot(lags, xcorr_slope, lw=2, label='slope(t),vol(t+j)', alpha=0.7)
        ax.plot(lags, xcorr_spread, lw=2, label='spread(t),vol(t+j)', alpha=0.7)
        ax.plot(lags, xcorr_depth, lw=2, label='depth(t),vol(t+j)', alpha=0.7)
        # ax.plot(lags, xcorr_volume, lw=2, label='|r(t)| vs Volume', alpha=0.7)
    else : # vikyne cross correlations
        ax.plot(lags, xcorr_volume, lw=2, label='vol(t),volume(t+j)', alpha=0.7)
        ax.plot(lags, xcorr_slopevol, lw=2, label='slope(t),volume(t+j)', alpha=0.7)
        ax.plot(lags, xcorr_slopecvar, lw=2, label='slope(t),cvar(t+j)', alpha=0.7)
    
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Cross Correlation Lag (j)')
    ax.set_ylabel('Correlation')
    # No title for paper
    """
    if title:
        ax.set_title(title)
    """
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')
    plt.savefig(output_file+".pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: run all experiments and generate all figures"""
    print("\n" + "="*70)
    print("LIMIT ORDER BOOK MODEL - FIGURE GENERATION")
    print("="*70)
    
    os.makedirs("figures", exist_ok=True)
    
    seed = 4242
    seed = 14
    experiments = {}
    
    # ========================================================================
    # RUN SIMULATIONS
    # ========================================================================
    
    print("\n" + "="*70)
    print("RUNNING SIMULATIONS")
    print("="*70)
    
    # Base case
    params_base = get_base_parameters()
    experiments['base'] = run_simulation("Base Case", params_base, seed)
    save_simulation_csv("dataOutputFile_base.csv", experiments['base'], 
                       params_base, seed)
    
    # Fixed shading
    params_fixed = get_base_parameters()
    params_fixed['adaptiveK'] = False
    experiments['fixed'] = run_simulation("Fixed Shading", params_fixed, seed)
    save_simulation_csv("dataOutputFile_fixed.csv", experiments['fixed'],
                       params_fixed, seed)
    
    # Homogeneous fundamental
    params_homFund = get_base_parameters()
    params_homFund['multipleFundamentals'] = False
    experiments['homFund'] = run_simulation("Homogeneous Fundamentals", params_homFund, seed)
    save_simulation_csv("dataOutputFile_homFund.csv", experiments['homFund'],
                       params_homFund, seed)
    
    # Additional experiments (optional - comment out to save time)
    if True :  # Set to False to skip these
        params_rc = get_base_parameters()
        params_rc['tau'] = 75
        experiments['reduced_cancel'] = run_simulation("Reduced Cancelations",
                                                       params_rc, seed)
        save_simulation_csv("dataOutputFile_reduced_cancel.csv",
                           experiments['reduced_cancel'], params_rc, seed)
        
        params_tv = get_base_parameters()
        params_tv['trimVol'] = True
        experiments['trimmed'] = run_simulation("Trimmed Volatility",
                                               params_tv, seed)
        save_simulation_csv("dataOutputFile_trimmed.csv",
                           experiments['trimmed'], params_tv, seed)
        
        params_noise = get_base_parameters()
        params_noise['sigmae'] = 0.001
        experiments['noise'] = run_simulation("Increased Noise",
                                             params_noise, seed)
        save_simulation_csv("dataOutputFile_noise.csv",
                           experiments['noise'], params_noise, seed)
        
        # Long base run for vol scaling, double standard run
        params_baseLong = get_base_parameters()
        params_baseLong['Tmax'] = 400000
        experiments['baseLong'] = run_simulation("Base Long",
                                             params_baseLong, seed)
        save_simulation_csv("dataOutputFile_baseLong.csv",
                           experiments['baseLong'], params_baseLong, seed)
        
        # Long base run for vol scaling, double standard run, heterogeneous vol forecasts
        params_baseHetVol = get_base_parameters()
        params_baseHetVol['Tmax'] = 400000
        params_baseHetVol['volLag'] = 0 # set for heterogeneous vol forecast
        experiments['baseHetVol'] = run_simulation("Base VolHet",
                                             params_baseHetVol, seed)
        save_simulation_csv("dataOutputFile_baseHetVol.csv",
                           experiments['baseHetVol'], params_baseHetVol, seed)
    
    # ========================================================================
    # GENERATE FIGURES
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    # Unpack base results
    (price, ret, rret, rRV, rRV2, totalV, rPrice, rSpread, rbidDepth,
     raskDepth, portDev, rportDev, holdings, dholdings, orders,
     rportDev_non_nan, Xsmooth_centered_non_nan, Xsmooth_trailing_non_nan,
     holdingsDiff, holdingsDiffABS, holdingsDist, holdingsDistABS, rVol,
     rOrders, marketBook, agentList, forecastSet, rdholdingsts, bidsdf,
     rbidSlope, asksdf, raskSlope, wealthByType, rpf, logPriceFund, rCRV,
     rtotalOrdersOnBook, orderFlow, pft) = experiments['base']
    
    # Figure 4: Base case
    print("\nGenerating Figure 4...")
    df_base = process_data_file("dataOutputFile_base.csv")
    plot_four_panel(df_base, params_base['tau'], params_base['sigmae'],
                   "figures/Figure4_base_case")
    
    # Figure 6: Fixed shading
    print("\nGenerating Figure 6...")
    df_fixed = process_data_file("dataOutputFile_fixed.csv")
    plot_four_panel(df_fixed, params_fixed['tau'], params_fixed['sigmae'],
                   "figures/Figure6_fixed_shading")
    
    # Figure 7: Homogeneous fundamentals
    print("\nGenerating Figure 7...")
    df_homFund = process_data_file("dataOutputFile_homFund.csv")
    plot_four_panel(df_homFund, params_homFund['tau'], params_homFund['sigmae'],
                   "figures/Figure7_homFund")
    
    
    # Figure 10: Vol distribution facts
    print("\nGenerating Figure 10...")
    df_baseLong = process_data_file("dataOutputFile_baseLong.csv")
    create_vol_facts_plot(df_baseLong, "figures/Figure10_vol_facts", 'Figure 10: Realized volatility distribution facts', start=500)
    
    # Figure 11: Long memory
    print("\nGenerating Figure 11...")
    df_baseLong = process_data_file("dataOutputFile_baseLong.csv")
    rRVLong = df_baseLong['rRV'].values
    create_long_memory_plot(rRVLong, "figures/Figure11_long_memory",
                           "Figure 11: Long memory tests", start=0)
    
    # Figure 13: Long memory, het volatility forecasts
    print("\nGenerating Figure 13...")
    df_baseHetVol = process_data_file("dataOutputFile_baseHetVol.csv")
    rRVLongHetVol = df_baseHetVol['rRV'].values
    create_long_memory_plot(rRVLongHetVol, "figures/Figure13_long_memoryHetVol",
                           "Figure 13: Long memory tests (Heterogeneous memory)", start=0)

    
    # Figure 20: Order book dynamics
    print("\nGenerating Figure 20...")
    df_base = process_data_file("dataOutputFile_base.csv", Start=0)
    rret = df_base['rret'].values
    rRV = df_base['rRV'].values
    rCRV = df_base['rCRV'].values
    rVol = df_base['rvol'].values
    rSpread = df_base['spread'].values
    rPrice  = df_base['rPrice'].values
    rbidSlope = df_base['bidslope'].values
    raskSlope = df_base['askslope'].values
    rbidDepth = df_base['bid depth'].values
    raskDepth = df_base['ask depth'].values
    create_orderbook_dynamics_plot(rret, rSpread, rPrice, rbidSlope, raskSlope,
                                   rbidDepth, raskDepth,
                                   "figures/Figure20_orderbook_dynamics",
                                   "Figure 20: Order Book Dynamics",start=0)
    
    # Figure 21: Cross-correlations
    print("\nGenerating Figure 21...")
    volatilityXC = True
    create_cross_correlation_plot(rret, rRV, rCRV, rSpread, rbidSlope, raskSlope,
                                  rbidDepth, raskDepth, rVol,volatilityXC,
                                  "figures/Figure21_cross_correlations",
                                  "Figure 21: Volatility Cross-Correlations",start=0)
    
    # Figure 22: Cross-correlations
    print("\nGenerating Figure 22...")
    volatilityXC = False
    create_cross_correlation_plot(rret, rRV, rCRV, rSpread, rbidSlope, raskSlope,
                                  rbidDepth, raskDepth, rVol,volatilityXC,
                                  "figures/Figure22_cross_correlations",
                                  "Figure 22: Slope Cross-Correlations",start=0)
    
    # Figure 23: Order book dynamics, fixed shading)
    
    print("\nGenerating Figure 23...")
    df_fixed = process_data_file("dataOutputFile_fixed.csv",Start=0)
    rretFix = df_fixed['rret'].values
    rRVFix = df_fixed['rRV'].values
    rCRVFix = df_fixed['rCRV'].values
    rVolFix = df_fixed['rvol'].values
    rSpreadFix = df_fixed['spread'].values
    rPriceFix  = df_fixed['rPrice'].values
    rbidSlopeFix = df_fixed['bidslope'].values
    raskSlopeFix = df_fixed['askslope'].values
    rbidDepthFix = df_fixed['bid depth'].values
    raskDepthFix = df_fixed['ask depth'].values
    create_orderbook_dynamics_plot(rretFix, rSpreadFix, rPriceFix, rbidSlopeFix, raskSlopeFix,
                                   rbidDepthFix, raskDepthFix,
                                   "figures/Figure23_orderbook_dynamics",
                                   "Figure 23: Order Book Dynamics (static shading)",start=0)
    
    
    # Figure 24: Cross-correlations (fixed shading)
    print("\nGenerating Figure 24...")
    volatilityXCFix = True
    create_cross_correlation_plot(rretFix, rRVFix, rCRVFix, rSpreadFix, rbidSlopeFix, raskSlopeFix,
                                  rbidDepthFix, raskDepthFix, rVolFix,volatilityXCFix,
                                  "figures/Figure24_cross_correlations",
                                  "Figure 24: Volatility Cross-Correlations (static shading)",start=0)
    
    
    
    # Additional figures if experiments were run
    if 'reduced_cancel' in experiments:
        print("\nGenerating Figure 14...")
        df_rc = process_data_file("dataOutputFile_reduced_cancel.csv")
        plot_four_panel(df_rc, params_rc['tau'], params_rc['sigmae'],
                       "figures/Figure14_reduced_cancelations.png")
    
    if 'trimmed' in experiments:
        print("\nGenerating Figure 15...")
        df_tv = process_data_file("dataOutputFile_trimmed.csv")
        plot_four_panel(df_tv, params_tv['tau'], params_tv['sigmae'],
                       "figures/Figure15_trimmed_volatility.png")
    
    if 'noise' in experiments:
        print("\nGenerating Figure 16...")
        df_noise = process_data_file("dataOutputFile_noise.csv")
        plot_four_panel(df_noise, params_noise['tau'], params_noise['sigmae'],
                       "figures/Figure16_increased_noise.png")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nGenerated figures are in the 'figures/' directory")
    print("Generated CSV files are in the current directory")

if __name__ == '__main__':
    main()