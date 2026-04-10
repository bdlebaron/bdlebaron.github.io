# -*- coding: utf-8 -*-
"""


Python code to go with 

Dynamic Order Dispersion and Volatility Persistence in a Simple
Limit Order Book Model
Andrew Hawley, Blake LeBaron, Mark Paddrik, Nathan Palmer

Third revision:  February, 2026

Code date: February 2026

This code and parameters runs basic benchmark simulation and should replicate (exactly) figure 4, our base case
model and parameters.

This program generates the four panel plots from the paper.




Simple code to plot time series 


Notes from Blake:
    
Remember the program builds aggregate series from the tick based series.  Ret are tick based, rret are aggregate
All series with r in front of them rPrice, rSpread, rbidDepth, raskDepth are aggregate.
 
Aggregate are supposed to represent daily series (they do in terms of standard deviations (roughly)). 
That means each tick is about 5 minutes.  Nice, BUT there are only 5 to 10 percent executed orders.  There is one order per tick, so activity in the market is still VERY low in terms of trades.  Seems silly to calibrate this until we get closer to tick level frequency.
 
retvol is a rolling high frequency vol series.
retcorr is the rolling profitability of liquidity (reversal) trades




"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# This import is deprecated in newer pandas versions
#from pandas.tools.plotting import autocorrelation_plot
from scipy.stats import kurtosis
from scipy.stats import norm
from scipy.stats import jarque_bera

# Import functions from LOBModel instead of redefining them?
#from LOBModel import fastautocorr, fastautocorr1, fastxcorr, archAdjust

#Main Result
#path_for_output = 'dataOutputFile0.csv'

# Plotting functions
def plot_four_panel(df, tau, sigmae, output_file=None):
    """
    Create the standard four-panel plot from paper.
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns including 'rret', 'rPrice', etc.
    tau : float
        Tau value for this simulation
    sigmae : float  
        Sigma_e value for this simulation
    output_file : str, optional
        If provided, save to this file
    """
    acflags = 50
    nbins = 50
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    temp_returns = df['rret'].values
    temp_prices = df['rPrice'].values
    
    # Panel 1: Returns time series
    ax[0, 0].plot(temp_returns)
    ax[0, 0].grid()
    ax[0, 0].set_title('Returns = r(t)')
    
    # Panel 2: Histogram with kurtosis
    T = len(temp_returns)
    temp_returns_clean = temp_returns[1:]
    
    n, bins, patches = ax[0, 1].hist(temp_returns_clean, nbins, density=True,
                                      facecolor='green', alpha=0.5)
    ax[0, 1].grid()
    mu = np.mean(temp_returns_clean)
    sigma = np.std(temp_returns_clean)
    xmin, xmax = ax[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax[0, 1].plot(x, p, 'k', linewidth=2)
    
    kurt = kurtosis(temp_returns_clean, fisher=False, bias=False)
    ax[0, 1].set_title(f'Histogram\nkurtosis = {kurt:.1f}')
    
    # Panel 3: Price with ACF(1)
    ax[1, 0].plot(temp_prices)
    ax[1, 0].grid()
    zz = fastautocorr(temp_prices, 2)[1]
    ax[1, 0].set_xlabel('Day')
    ax[1, 0].set_title(f"Price: ACF(1) = {zz:.3f}")
    
    # Panel 4: Autocorrelations
    retacf = fastautocorr(temp_returns_clean, acflags)
    aretacf = fastautocorr(np.abs(temp_returns_clean), acflags)
    xpts = np.arange(1, acflags + 1, 1)
    
    ax[1, 1].plot(xpts, retacf[1:], label="r(t)")
    ax[1, 1].plot(xpts, aretacf[1:], label="|r(t)|")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel('Lag')
    
    bolBands = np.ones(50) * 1.96 / np.sqrt(T)
    ax[1, 1].plot(xpts, bolBands, 'r--')
    ax[1, 1].plot(xpts, -bolBands, 'r--')
    ax[1, 1].set_title('Autocorrelations')
    ax[1, 1].grid()
    
    if output_file:
        plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')
        plt.savefig(output_file+".pdf", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    plt.close()

def process_data_file(path_for_output, Delta=1, Start=500):
    """
    Load and process a data file.
    
    Parameters:
    -----------
    path_for_output : str
        Path to CSV file
    Delta : int
        Downsampling interval
    Start : int
        Starting row to use
        
    Returns:
    --------
    df : DataFrame
        Processed data
    """
    df = pd.read_csv(path_for_output)
    
    df.columns = ["price", "ret", "rret", "rRV", "rRV2", "rCRV",
                  "autocorrelation sum", "rvol", "rPrice", "spread",
                  "totalOrders", "bid depth", "bidslope", "ask depth",
                  "askslope", "tau", "sigmaE", "seed", "fundamental"]
    
    # df['rvol'] = df['rvol'].rolling(50).sum()
    
    # Downsample if needed
    df = df.iloc[Start::Delta, :]
    df['abs_rret'] = np.abs(df['rret'])
    
    return df



def plot_normal_histogram_helper(xdata,nbins=50):
    """
    Helper function that creates histogram data and computes normal distribution parameters.
    
    Parameters:
    -----------
    xdata : array-like
        Data to plot in histogram
    nbins : int, default=50
        Number of bins for the histogram
    
    Returns:
    --------
    bins : array
        Bin edges
    y : array
        Normal PDF values corresponding to bins
    curve_color : str
        Color to use for the normal curve ('r' for red)
    """
    n, bins, patches = plt.hist(xdata,nbins,normed=1,facecolor='green', alpha=0.5)
    mu = np.mean(xdata)  # Calculate mean of data
    sigma = np.std(xdata)  # Calculate standard deviation
    y = plt.mlab.normpdf(bins,mu,sigma)  # Generate normal PDF values
    return bins, y, 'r'
    
def plot_normal_histogram(xdata,nbins=50):
    """
    Plot a histogram of data with a normal distribution curve overlay.
    
    Parameters:
    -----------
    xdata : array-like
        Data to plot in histogram
    nbins : int, default=50
        Number of bins for the histogram
    """
    binsval, yval, curve_color =  plot_normal_histogram_helper(xdata,nbins=50)
    plt.plot(binsval, yval, curve_color)

def plot_time_series(prices, label='Price'):
    """
    Plot a time series with proper labels and grid.
    
    Parameters:
    -----------
    prices : array-like
        Time series data to plot
    label : str, default='Price'
        Y-axis label for the plot
    """
    plt.clf()
    plt.plot(prices)
    plt.xlabel('Period(t)')
    plt.ylabel(label)
    plt.grid()
    
# helper routine for basic autocorrelations
def autocorr(x,m):
    """
    Compute autocorrelation function for lags 0 to m.
    
    Parameters:
    -----------
    x : array-like
        Time series data
    m : int
        Maximum lag to compute
    
    Returns:
    --------
    array
        Autocorrelation values from lag 0 to m
    """
    n = len(x)
    v = x.var()  # Variance of the series
    x2 = x-x.mean()  # Demean the series
    r = np.correlate(x2,x2,mode="full")[(n-1):(n+m+1)]  # Compute correlation using numpy's correlate function
    result = r/(n*v)  # Normalize by n*variance
    return result

def fastxcorr(x,y,m):
    """
    Compute cross-correlation between two series for lags -m to m.
    This manually implements cross-correlation for better performance.
    
    Parameters:
    -----------
    x : array-like
        First time series
    y : array-like
        Second time series
    m : int
        Maximum lag in both directions
    
    Returns:
    --------
    array
        Cross-correlation values from lag -m to m
    """
    nx = len(x)
    z = np.zeros(2*m+1)  # Initialize array to hold cross-correlation values
    stdprod = np.std(x)*np.std(y)  # Product of standard deviations for normalization
    mx = np.mean(x)
    my = np.mean(y)
    for i in range(m+1):
        z[i+m] = np.mean( (x[0:(nx-i)]-mx)*(y[i:nx]-my))  # Compute positive lags
        # print(z[i+m])
    for i in range(m):
        j = m-i
        z[i] = np.mean( (x[j:nx]-mx)*(y[0:(nx-j)]-my))  # Compute negative lags
    # print(z)
    # print(stdprod)
    z = z/stdprod  # Normalize to get correlation coefficients
    return z

def fastautocorr(x,m):
    """
    Compute autocorrelation function for lags 0 to m.
    This is a manual implementation for better performance.
    
    Parameters:
    -----------
    x : array-like
        Time series data
    m : int
        Maximum lag to compute
    
    Returns:
    --------
    array
        Autocorrelation values from lag 0 to m
    """
    nx = len(x)
    z = np.zeros(m+1)  # Initialize array for autocorrelation values
    v = np.var(x)  # Variance for normalization
    mx = np.mean(x)
    for i in range(0,m+1):
        z[i] = np.mean( (x[0:(nx-i)]-mx)*(x[i:nx]-mx))  # Compute autocorrelation at lag i
    z = z/v  # Normalize by variance
    return z

def fastautocorr1(x):
    """
    Compute lag-1 autocorrelation efficiently.
    
    Parameters:
    -----------
    x : array-like
        Time series data
    
    Returns:
    --------
    float
        Autocorrelation at lag 1
    """
    # assuming mean = 0
    n = len(x)
    v = x.var()  # Variance for normalization
    me = x.mean()
    cv =np.mean( ( x[0:(n-1)]-me)*(x[1:n]-me))  # Compute covariance at lag 1
    rho = cv/v  # Normalize by variance to get correlation
    return rho

def archAdjust(x,k):
    """
    Compute ARCH adjustment factors for autocorrelation confidence bands.
    Accounts for heteroskedasticity in the time series.
    
    Parameters:
    -----------
    x : array-like
        Time series data
    k : int
        Maximum lag to compute adjustment for
    
    Returns:
    --------
    array
        ARCH adjustment factors for lags 0 to k
    """
    T = len(x)
    x = x-np.mean(x)  # Demean the series
    adj = np.zeros(k+1)  # Initialize adjustment factors array
    for i in range(1,k):
        num = np.mean( x[0:(T-i)]**2 * x[i:T]**2)  # Numerator: mean product of squared terms
        den = np.mean( x**2 )**2  # Denominator: square of mean squared term
        # print(num/den)
        adj[i]= 1./np.sqrt(T)*num/den  # Compute adjustment factor
    # print(adj.shape)
    return adj
        
def hill_estimate(x,frac):
    """
    Returns the Hill Estimators for some 1D data set.
    Used to estimate the tail exponent of the distribution.
    
    Parameters:
    -----------
    x : array-like
        Data to estimate tail exponent for
    frac : float
        Fraction of data (from the tail) to use for estimation
        
    Returns:
    --------
    float
        Hill estimator (inverse of the tail exponent)
    """    
    # sort data in such way that the smallest value is first and the largest value comes last:
    Y = np.sort(x)
    n = len(Y)
    k = int(frac*n)  # Number of observations to use from the tail
    Y = Y[-k:]  # Extract the tail observations
    
    base = (1./k)* np.sum ( np.log(Y) - np.log(Y[0]))  # Hill estimator formula
    return 1./base


if __name__ == '__main__':
    """
    Main execution: process data files and create plots.
    """
    import sys
    
    # Default input file
    input_file = 'dataOutputFile0.csv'
    
    # Check if file provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print(f"Processing: {input_file}")
    
    # Load and process data
    df = process_data_file(input_file, Delta=1, Start=500)
    
    print(f"Sample length: {len(df)}")
    print(f"Annualized std: {np.sqrt(250) * np.std(df['rret']):.4f}")
    print(f"Kurtosis: {kurtosis(df['rret'], fisher=False, bias=False):.2f}")
    
    # Get unique parameter combinations
    tau_list = df['tau'].unique()
    sigmae_list = df['sigmaE'].unique()
    
    # Create plots for each combination
    for tau in tau_list:
        for sigmae in sigmae_list:
            tau_idx = df['tau'] == tau
            sigmae_idx = df['sigmaE'] == sigmae
            tau_sigmae_idx = np.logical_and(tau_idx, sigmae_idx)
            
            # Create subset
            df_subset = df[tau_sigmae_idx].reset_index(drop=True)
            
            # Create filename
            sigmae_str = str(sigmae).replace('.', '_')
            tau_str = str(tau).replace('.', '_')
            output_file = f'fourplot_tau_{tau_str}_sigmae_{sigmae_str}.png'
            
            print(f"\nGenerating plot for tau={tau}, sigmae={sigmae}")
            plot_four_panel(df_subset, tau, sigmae, output_file)
            
            # Print statistics
            print(f"  Std: {np.sqrt(250) * np.std(df_subset['rret']):.4f}")
            print(f"  Kurtosis: {kurtosis(df_subset['rret'], fisher=False, bias=False):.2f}")
    
    print("\nDone!")

# # First read in all the data: 

# path_for_output = 'dataOutputFile0.csv' #dataOutputFile0.csv
# df = pd.read_csv(path_for_output)

# print(df.shape)
# print(len(df))

# df.columns=["price",
          # "ret",
          # "rret",
          # "rRV",
          # "rRV2","rCRV",
          # "autocorrelation sum",
          # "rvol",
          # "rPrice",
          # "spread","totalOrders",
          # "bid depth","bidslope",
          # "ask depth","askslope",
          # "tau",
          # "sigmaE","seed","fundamental" ]

# df['rvol']=df['rvol'].rolling(50).sum()
# print("sample length=",len(df))

# Delta = 50
# Delta = 1
# Start = 25000
# Start = 500
# drraw = df.copy()
# df=df.iloc[Start::Delta,:]
# df['abs_rret']=np.abs(df['rret'])

# # Extract the tau and sigma_e values:
# tau_list = df['tau'].unique()
# sigmae_list = df['sigmaE'].unique()

# print("sample length=",len(df))

# for tau in tau_list:
    # for sigmae in sigmae_list:
        # acflags = 50
        # tau_idx = df['tau'] == tau
        # sigmae_idx = df['sigmaE'] == sigmae
        # tau_sigmae_idx = np.logical_and(tau_idx, sigmae_idx)
        
        # # Let's remove the extra '.' from the name:
        # sigmae_str = str(sigmae)
        # sigmae_str = sigmae_str.replace('.', '_')
        # tau_str = str(tau)
        # tau_str = tau_str.replace('.', '_')
        
        # filename = 'fourplot_tau_'+tau_str+'sigmae_'+sigmae_str+".png"
        
        # nbins = 50
        # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        # fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # # fig.suptitle('Returns for tau = '+str(tau)+", simgae = "+str(sigmae))
        # temp_returns = df['rret'][tau_sigmae_idx]
        # temp_returns = temp_returns.reset_index(drop=True)
        # temp_prices = df['rPrice'][tau_sigmae_idx]
        # # temp_prices = df['rPrice'][tau_sigmae_idx]                                                            a
        # temp_prices = temp_prices.reset_index(drop=True)
        # temp_vol = df['rvol'][tau_sigmae_idx]
        # temp_vol = temp_vol.reset_index(drop='True')
        # #fig, ax = plt.subplots(2, 2)
        # ax[0, 0].plot(temp_returns)
        # ax[0, 0].grid()
        # ax[0, 0].set_title('Returns = r(t)')

        
        # T = len(temp_returns.values)
        # temp_returns = temp_returns[1:len(temp_returns)]
        
        # # get tail exponent estimate
        # # kappa = hill_estimate(np.abs(temp_returns.values),0.10)
        # # kappaView = round(kappa,1)
        # print('kappa')
        # # print(kappa,len(temp_returns))
        
        # n, bins, patches = ax[0, 1].hist(temp_returns, nbins, density=True,facecolor='green', alpha=0.5)
        # ax[0,1].grid()
        # mu = np.mean(temp_returns)
        # sigma = np.std(temp_returns)
        # xmin, xmax = ax[0,1].get_xlim()

        # # Generate normal pdf from scipy.stats
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, sigma)
          
        # ax[0,1].plot(x, p, 'k', linewidth=2)


        
        # # y = plt.mlab.normpdf(bins,mu,sigma)
        # # ax[0, 1].plot(bins, y, 'r')
        # # ax[0, 1].set_title('Histogram \n kurtosis, tail = '+str(round(kurtosis(temp_returns, fisher=False, bias=False),1))+', '+str(kappaView))
        
        # ax[0, 1].set_title('Histogram \n kurtosis = '+str(round(kurtosis(temp_returns, fisher=False, bias=False),1)))
        
        # # plot_acf(temp_returns, ax=ax[2,1], lags=50,  zero=False)
        # retacf = fastautocorr(temp_returns.values,acflags)
        # T = len(temp_returns.values)
        # adj = archAdjust(temp_returns.values,50)
        # # print(adj)
        # # bolBands = np.ones(50)*1.96/np.sqrt(T)
        # # bolBands = 1.96*adj[1:]
        # xpts = np.arange(1,acflags+1,1)
        # ax[1,1].plot(xpts,retacf[1:],label="r(t)")
        # # ax[2,1].plot(xpts,bolBands,'r--')
        # # ax[2,1].plot(xpts,-bolBands,'r--')
        # # ax[2,1].set_title('Autocorrelation; R')
        # # ax[2,1].grid()
        # # plot_acf(np.abs(temp_returns), ax=ax[1,1], lags=50,  zero=False, title='Autocorrelation; |R|')
        # aretacf = fastautocorr(np.abs(temp_returns.values),acflags)
        # ax[1,1].plot(xpts,aretacf[1:],label="|r(t)|")
        # ax[1,1].legend()
        # ax[1,1].set_xlabel('Lag')
        # bolBands = np.ones(50)*1.96/np.sqrt(T)
        # ax[1,1].plot(xpts,bolBands,'r--')
        # ax[1,1].plot(xpts,-bolBands,'r--')
        # ax[1,1].set_title('Autocorrelations')
        # ax[1,1].grid()
        # ax[1,0].plot(temp_prices)
        # ax[1,0].grid()
        # zz = fastautocorr(temp_prices.values,2)[1]
        # ax[1,0].set_xlabel('Day')
        # ax[1,0].set_title("Price: ACF(1) = "+str(round(zz,3)))
        # # ax[1,0].plot(temp_vol)
        # # ax[1,0].grid()
        # # ax[1,0].set_title("Volume")
        # # plt.close()
        # plt.show()
        
        
        # print("Std: ",np.sqrt(250)*np.std(temp_returns))
        # print("Kurtosis: ", kurtosis(temp_returns))
        # from scipy.stats import kstest
        # x = np.random.standard_normal(3500)
        # print(kstest(temp_returns,'norm',alternative='two-sided'))
        # print(kstest(x, 'norm',alternative='two-sided'))

        # print(jarque_bera(temp_returns))
        # print(jarque_bera(x))
        
        
# plt.show()


