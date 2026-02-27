#!/usr/bin/env python3
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

"""

# import usual Python helpers
from __future__ import print_function, division
import sys
import numpy as np
import numpy.random as rnd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import statsmodels.tsa.stattools as stattools
import statsmodels.api as sm
import pandas as pd
import csv
import copy
from time import time
import cProfile
from random import random

#Let's look at some correlations
from scipy.stats import pearsonr
from matplotlib.pyplot import clf

slurm_it=int(1)
slurm_array=50

#Class for forecasts
class forecasts:
    """
    A class that manages price forecast components for market agents.
    
    This class calculates and stores various forecasting components that agents use
    to form their price expectations, including:
    - Fundamental forecasts (based on perceived fundamental value)
    - Chartist forecasts (based on historical price patterns)
    - Noise components (random elements that create trading heterogeneity)
    - Volatility estimates at different time scales

    Note on chartists: the fraction of chartist agents are set to zero for the 
    paper, as they were discovered to not heavily influence results. These have 
    been left in the code for the sake of potential experimentation.

    The class also tracks volatility ratios which inform agent behavior during
    periods of changing market conditions.
    """
    # initialize forecast components
    def __init__(self,Lmax,pf,sigmae,volLag):
        """
        Initialize the forecast components and parameters.
        
        Parameters:
        -----------
        Lmax : int
            Maximum lookback period for chartist strategies
        pf : float
            Initial fundamental price
        sigmae : float
            Standard deviation of noise term in forecasts
        volLag : int
            Base period for volatility calculations
        """
        # Set up fundamental forecasts with multiple versions (low/mid/high)
        fundamentalSpread = 0.10
        deltaSpread = fundamentalSpread/3.
        self.fundamentalAdjust = np.array([1.-fundamentalSpread/2., 1.0, 1.+fundamentalSpread/2.])
        self.nFundamental = len(self.fundamentalAdjust)

        # Initialize arrays for current and previous fundamental forecasts
        self.fundamental = np.zeros(self.nFundamental)
        self.fundamentalold = np.zeros(self.nFundamental)

        # chartist is vector for different lengths
        self.chartist    = np.zeros(Lmax)
        self.noise       = 0.
        self.v = 0.
        self.spread = 1.
        self.vShort = 0.

        # Main benchmark
        self.volLag =   [int(0.1*volLag),int(0.5*volLag),volLag,2*volLag,4*volLag,8*volLag,16*volLag,32*volLag]
        # self.volLag = [int(0.1*volLag), volLag,2*volLag,4*volLag,8*volLag,16*volLag,32*volLag]
        # self.volLag = [ 32*volLag] 
        
        # Configure the lookback windows for different volatility estimates
        # The model uses multiple timescales to capture market dynamics
        self.volLag = [volLag]
        # self.volLag = [int(0.5*volLag),volLag,2*volLag,3*volLag,5*volLag,8*volLag,13*volLag]
        # self.volLag = [int(0.33*volLag),volLag,3*volLag,9*volLag,27*volLag]
        # self.volLag = [volLag,2*volLag,3*volLag,5*volLag,8*volLag,13*volLag]
        # self.volLag = [50, 250, 1000]
        # self.volLag = [int(0.33*volLag),volLag,7*volLag]
        self.nVol = len(self.volLag)
        
        # Volatility ratios measure relative volatility between different timescales
        self.vRatio = np.ones(self.nVol)
        
        # Store fundamental price and parameters
        self.pf = pf
        self.pfold = pf
        self.Changet = 0
        self.sigmae = sigmae
        self.Lmax = Lmax
        # Maximum forecast values for each lookback period
        self.fMax = 0.01*np.ones(Lmax)         

    def changeFundamental(self,t):
        """
        Change the fundamental price in a regime-switching manner.
        
        Adjusts the fundamental price up or down by 10% based on current level
        and random selection.
        
        Parameters:
        -----------
        t : int
            Current time step
        """
        # Store old fundamental price and update change time
        self.pfold=self.pf
        self.Changet=t
        print(t)
        # Calculate potential new price levels (10% higher or lower)
        highP = self.pf*1.1
        lowP  = self.pf*0.9
        # Logic for selecting new fundamental price
        if(self.pf>1300.):
            newpf = lowP
        if(self.pf<700):
            newpf=highP
        if(self.pf>= 700 and self.pf<=1300):
            if(np.random.randint(0,2)==1):
              newpf=highP
            else:
              newpf=lowP
        self.pf = newpf
        print(newpf)
    def changeFundamentalRW(self,minp,maxp,rwstd):
        """
        Change the fundamental price following a random walk with boundaries.
        
        Parameters:
        -----------
        minp : float
            Minimum allowed price
        maxp : float
            Maximum allowed price
        rwstd : float
            Standard deviation of the random walk innovation
        """
        # Apply log-normal random walk to fundamental price
        logp = np.log(self.pf)
        newlogp = logp + rwstd*np.random.standard_normal(1)
        self.pf = np.exp(newlogp)
        # Enforce price boundaries
        if(self.pf>maxp):
            self.pf = maxp
        if(self.pf<minp):
            self.pf = minp
            
    def changeFundamentalSpread(self):
        """
        Change the spread between different fundamental price adjustments.
        
        Increases the diversity of fundamental value perceptions across agents.
        """
        # Wider fundamental spread (0.4 vs initial 0.1)
        fundamentalSpread = 0.4
        deltaSpread = fundamentalSpread/self.nFundamental
        # Create evenly spaced adjustment factors centered at 1.0
        self.fundamentalAdjust = np.arange(1.-fundamentalSpread/2.,1.+fundamentalSpread/2.,deltaSpread)               
    # update forecasts   
    def updateForecasts(self,t,Tinit,price,ret,rho,trimVol,spread,deltaP,rhoadjust,noiseOn):
        """
        Update all forecast components based on current market conditions.
        
        This core method updates fundamental, chartist, and noise components
        of forecasts, as well as volatility estimates at different time scales.
        
        Note on chartists: the fraction of chartist agents are set to zero for the 
        paper, as they were discovered to not heavily influence results. These have 
        been left in the code for the sake of potential experimentation.

        
        Parameters:
        -----------
        t : int
            Current time step
        Tinit : int
            Initial time period (for calibration/warmup)
        price : float
            Current market price
        ret : ndarray
            Array of historical returns
        rho : float
            Mean reversion parameter for fundamental component
        trimVol : bool
            Whether to trim outliers when calculating volatility
        spread : ndarray
            Historical bid-ask spreads
        deltaP : float
            Minimum price increment
        rhoadjust : bool
            Whether to override rho parameter
        noiseOn : bool
            Whether to include noise component in forecasts
        """
        # This is a bit of an annoyming parameter
        # could be set to 1, or a reasonable tau in common use (5 or 10)
        forecastHorizon = 1.
        # Store current spread for reference
        self.spread = np.mean(spread[(Tinit-100):(t+1)])
        
        
        # Update fundamental forecasts (previous and current)
        # The forecast is based on mean reversion toward fundamental price
        self.fundamentalold = forecastHorizon*(rho-1.)*np.log(price/(self.pfold*self.fundamentalAdjust))
        # Adjust rho parameter if needed
        if(rhoadjust):
          rho=0.98
        # Calculate new fundamental forecasts
        self.fundamental = forecastHorizon*(rho-1.)*np.log(price/(self.pf*self.fundamentalAdjust))
        # Noise parameters
        noiseStd = self.sigmae
        noiseVar = noiseStd**2
        rho = 0.00
        adjVar = (1.-rho**2)*noiseVar
        adjStd = np.sqrt(adjVar)
        
        # Generate random noise term
        self.noise = adjStd*np.random.randn()
        # gen short volatility estimates
        # Calculate long-term volatility based on recent history
        self.v = np.var(ret[(Tinit-100):(t+1)])
        # Calculate volatility at different time scales
        for i in range(self.nVol):
            # Limit lookback to available history
            lag = min(t,self.volLag[i])
            # Set maximum allowed deviation for volatility calculation
            vMax = 2.0*np.sqrt(self.v)*np.ones(lag)
            # Get most recent returns
            retTrim = ret[t-lag+1:t+1]
            # Optionally trim outliers for more stable estimates
            if(trimVol):
                retTrim = np.minimum(retTrim,vMax)
                retTrim = np.maximum(retTrim,-vMax)
            # Calculate short-term volatility
            self.vShort = np.var(retTrim)
            # Calculate ratio of short to long-term volatility
            # This ratio indicates changing market conditions
            self.vRatio[i] = np.sqrt(self.vShort/self.v)
        # Calculate chartist (trend) components at different time scales
        # Take returns in reverse order for easier cumulative calculation
        revrets = ret[t:(t-self.Lmax):-1]
        # Calculate moving averages of different lengths
        self.chartist = np.cumsum(revrets)/np.arange(1.,float(self.Lmax+1))
        # Convert to forecast over the horizon
        self.chartist = forecastHorizon*np.log(1.+self.chartist)
        
    def updateForecastsNoiseOnly(self):
        """
        Update only the noise component of forecasts.
        
        Used for efficiency when only noise needs to be refreshed.
        """
        self.noise = self.sigmae*np.random.randn()
        
    def updateVolOnly(self,t,ret):
        """
        Update only the volatility estimates.
        
        Parameters:
        -----------
        t : int
            Current time step
        ret : ndarray
            Array of historical returns
        """
        # Calculate volatility for shortest lookback period
        lag = min(t,self.volLag)
        vMax = 1.5*2.*np.sqrt(self.v)*np.ones(lag)
        retTrim = ret[t-lag+1:t+1]
        # Update long-term volatility based on all history
        self.v = np.var(ret[0:t+1])
        # Update short-term volatility based on recent history
        self.vShort = np.var(retTrim)
        # Update volatility ratio
        self.vRatio = np.sqrt(self.vShort/self.v)
        # Refresh noise term
        self.noise = self.sigmae*np.random.randn()


#Class to define the orderbook 
class orderBook:
    """
    A class that implements a limit order book for the ABM simulation.
    
    The order book maintains separate lists for bid (buy) and ask (sell) orders at
    discrete price levels. It handles the addition of new orders, execution of trades
    when orders cross, and periodic cleaning of expired orders.
    
    Attributes:
        minPrice (float): Minimum price allowed in the order book
        maxPrice (float): Maximum price allowed in the order book
        midPrice (float): Initial midpoint between min and max prices
        deltaPrice (float): Price increment for discrete price levels
        priceVec (numpy.array): Array of all possible price levels
        nPrice (int): Number of discrete price levels
        bids (list): List of lists storing bid orders at each price level
        asks (list): List of lists storing ask orders at each price level
        bestBidDex (int): Index of the current best (highest) bid price
        bestBid (float): The current best bid price
        bestAskDex (int): Index of the current best (lowest) ask price
        bestAsk (float): The current best ask price
    """
    def __init__(self,minP,maxP,deltaP,agentList,nAgents,Tinit,tau):
        """
        Initialize the order book with specified parameters and populate with initial orders.
        
        Args:
            minP (float): Minimum price allowed in the book
            maxP (float): Maximum price allowed in the book
            deltaP (float): Price increment for discrete price levels
            agentList (list): List of agent objects that can place orders
            nAgents (int): Number of agents in the simulation
            Tinit (int): Initial time period
            tau (int): Maximum order lifetime
        """
        # price ranges on order book
        self.minPrice = minP
        self.maxPrice = maxP
        self.midPrice = (maxP+minP)/2.
        # discreteness in book
        self.deltaPrice = deltaP
        # discrete prices
        self.priceVec = np.arange(minP,maxP+deltaP,deltaP)
        self.nPrice = len(self.priceVec)
        # set up lists of lists for bids and asks
        self.bids = []
        self.asks = []
        for i in range(self.nPrice):
            self.bids.append([])
            self.asks.append([])
        # generate best bid
        pmid = self.discretePrice(self.midPrice)
        self.bestBidDex = pmid[0]-1
        self.bestBid = self.realPrice(self.bestBidDex)
        # drop 5 orders in at best bid
        for i in range(2):
             # order length for initial orders
            startTau = np.random.randint(tau)
            randomAgent = agentList[np.random.randint(nAgents)]
            self.bids[self.bestBidDex].append((1.,Tinit+startTau,randomAgent))
        # drop 45 random orders near best bid
        for i in range(150):
             # order length for initial orders
            startTau = np.random.randint(tau)
            randomAgent = agentList[np.random.randint(nAgents)]
            self.bids[self.bestBidDex-np.random.randint(1,self.bestBidDex/4)].append((1.,Tinit+startTau,randomAgent))
        # now set up best asks
        self.bestAskDex = pmid[0]+1 
        self.bestAsk = self.realPrice(self.bestAskDex)
        for i in range(2):
             # order length for initial orders
            startTau = np.random.randint(tau)
            randomAgent = agentList[np.random.randint(nAgents)]
            self.asks[self.bestAskDex].append((1.,Tinit+startTau,randomAgent))
        for i in range(150):
             # order length for initial orders
            startTau = np.random.randint(tau)
            randomAgent = agentList[np.random.randint(nAgents)]
            self.asks[self.bestAskDex+np.random.randint(1,self.bestAskDex/4)].append((1.,Tinit+startTau  ,randomAgent))
    # take price and return (index, descrete price)      
    def discretePrice(self,price):
        """
        Convert a continuous price to a discrete price index and value.
        
        Args:
            price (float): The continuous price to convert
            
        Returns:
            tuple: (index, discrete_price) where index is the array position and 
                  discrete_price is the corresponding price value
        """
        iPrice = int((price-self.minPrice)/self.deltaPrice)
        iPrice = max(iPrice,0)
        iPrice = min(iPrice,self.nPrice-1)
        dPrice = self.minPrice + self.deltaPrice*iPrice
        return (iPrice, dPrice)
    # take discrete price and return real price    
    def realPrice(self,iPrice):
        """
        Convert a discrete price index to the corresponding price value.
        
        Args:
            iPrice (int): The index of the price in the price vector
            
        Returns:
            float: The corresponding price value
        """
        return self.minPrice + self.deltaPrice*iPrice
    # add bid orders to the book    
    def addBid(self,trader1,quant,t,tnoise):
        """
        Add a bid (buy) order to the book or execute a trade if it crosses the spread.
        
        If the bid price is below the best ask price, add it to the order book.
        If the bid price is equal to or above the best ask price, execute a trade
        with the first order at the best ask price.
        
        Args:
            trader1 (agent): The agent placing the bid
            quant (float): The quantity to bid
            t (int): Current time step
            tnoise (float): Price noise to add to the bid
            
        Returns:
            float: Trade price if a trade was executed, -1 otherwise
        """
        # default no trade flag
        price = trader1.bid+tnoise
        tradePrice = -1.
        # discretize prices
        ptup   = self.discretePrice(price)
        iPrice = ptup[0]
        dPrice = ptup[1]
        # price < best ask then add to bid side
        # print(dPrice,self.bestAsk)
        if dPrice < self.bestAsk:
            # push (order,t) onto book at iPrice
            self.bids[iPrice].append((quant,t+trader1.tau,trader1))
            trader1.orders += quant
            # if better than bestBid, then update best
            if(dPrice>self.bestBid):
                self.bestBid = dPrice
                self.bestBidDex = iPrice
        # price > best ask, then execute trade at best ask
        else:
            # pop first in order off best ask
            tradeInfo = self.asks[self.bestAskDex].pop(0)
            tradePrice = self.bestAsk
            tradeQ     = tradeInfo[0]
            trader2    = tradeInfo[2]
            tradeTime  = t-(tradeInfo[1]-trader2.tau)
            # print('buy')
            trader1.processBuy(trader2,tradePrice,tradeQ,tradeTime)
            # walk up the book to find new best ask
            for j in range(self.bestAskDex,self.nPrice):
                if( self.asks[j]!=[]):
                    break
            # reached end of book, set price there, and fill with order (emergency)
            if (self.asks[j] == [] ):
                self.bestAsk = self.priceVec[j]
                self.bestAskDex = j
                self.asks[j].append((1.,t+trader1.tau,trader1))
            # best ask at first occupied order
            else:
                self.bestAsk = self.priceVec[j]
                self.bestAskDex = j
        return tradePrice
    # repeat all this for adding an ask     
    def addAsk(self,trader1,quant,t,tnoise):
        """
        Add an ask (sell) order to the book or execute a trade if it crosses the spread.
        
        If the ask price is above the best bid price, add it to the order book.
        If the ask price is equal to or below the best bid price, execute a trade
        with the first order at the best bid price.
        
        Args:
            trader1 (agent): The agent placing the ask
            quant (float): The quantity to ask
            t (int): Current time step
            tnoise (float): Price noise to add to the ask
            
        Returns:
            float: Trade price if a trade was executed, -1 otherwise
        """
        tradePrice = -1.
        price = trader1.ask + tnoise
        ptup   = self.discretePrice(price)
        iPrice = ptup[0]
        dPrice = ptup[1]
        if dPrice > self.bestBid:
            # push (order,t) onto book at iPrice
            self.asks[iPrice].append((quant,t+trader1.tau,trader1))
            trader1.orders -= quant
            if(dPrice < self.bestAsk):
                self.bestAsk = dPrice
                self.bestAskDex = iPrice
        else:
            tradeInfo = self.bids[self.bestBidDex].pop(0)
            tradePrice = self.bestBid
            tradeQ     = tradeInfo[0]
            trader2    = tradeInfo[2]
            tradeTime  = t-(tradeInfo[1]-trader2.tau)
            trader1.processSell(trader2,tradePrice,tradeQ,tradeTime)
            for j in range(self.bestBidDex,-1,-1):
                if( self.bids[j]!=[]):
                    break
            if (self.bids[j] == [] ):
                self.bestBid = self.priceVec[j]
                self.bestBidDex = j
                self.bids[j].append((1.,t+trader1.tau,trader1))
            else:
                self.bestBid = self.priceVec[j]
                self.bestBidDex = j
        return tradePrice
    # cleanse book of old orders   
    def cleanBook(self,t,tau,agentList):
        """
        Remove expired orders from the order book and ensure the book remains valid.
        
        Go through all price levels and remove any orders that have expired (their
        lifetime exceeded). Also ensure there is at least one order at the maximum
        and minimum prices to keep the book valid.
        
        Args:
            t (int): Current time step
            tau (int): Maximum order lifetime
            agentList (list): List of agent objects
        """
        # sweep through book
        for i in range(self.nPrice):
            # check time on first (oldest) order
            if(self.bids[i]!=[]):
                # if old, then pop it off, sweep through all
                nremove = 0
                for j in range(len(self.bids[i])):
                # activate this line for just first order
                # for j in range(1):
                    if(self.bids[i][j-nremove][1])<t:
                        nremove += 1
                        # remember to reduce agent order by q
                        canceledOrder = self.bids[i].pop(j-nremove)
            # same for ask side
            if(self.asks[i]!=[]):
                nremove = 0
                for j in range(len(self.asks[i])):
                # for j in range(1):
                    if(self.asks[i][j-nremove][1]<t):
                        nremove += 1
                        canceledOrder = self.asks[i].pop(j-nremove)
        # make sure there is some order at the end
        randAgent = agentList[np.random.randint(len(agentList))]
        if self.bids[0]==[]:
            self.bids[0].append((1.,randAgent.tau+t,randAgent))
        if self.asks[self.nPrice-1]==[]:
            self.asks[self.nPrice-1].append((1.,randAgent.tau+t,randAgent))
        # reset best bid and ask prices
        i = self.nPrice-1
        while self.bids[i]==[]:
            i -= 1
        self.bestBidDex = i
        self.bestBid = self.realPrice(i)
        i = 0
        while self.asks[i]==[]:
            i += 1
        self.bestAskDex = i
        self.bestAsk = self.realPrice(i)
        
    # crude depth estimate
    def bookDepth(self,forecast):
        """
        Calculate the depth of the order book within a certain price range.
        
        Depth is the total volume of orders (bids and asks) within a specified price range
        around the midpoint price. This is used as a measure of market liquidity.
        
        Args:
            forecast (forecasts): The forecast object containing volatility information
            
        Returns:
            tuple: (bidVol, askVol) total volume of bids and asks within the range
        """
        bidVol = 0.
        askVol = 0.
        # set range to 1/2 std
        nearPriceRange = 1000*2.*np.sqrt(forecast.v)
        midPoint = (self.bestBid + self.bestAsk)/2.
        for i in range(self.nPrice):
            if ( self.bids[i]!=[]):
                if(abs(self.realPrice(i)-midPoint)<nearPriceRange):
                    for j in range(len(self.bids[i])):
                        bidVol += self.bids[i][j][0]
            if ( self.asks[i]!=[]):
                if(abs(self.realPrice(i)-midPoint)<nearPriceRange):
                    for j in range(len(self.asks[i])):
                        askVol += self.asks[i][j][0]
        return bidVol, askVol

    # total orders on book
    def totalOrders(self):
        """
        Calculate the total number of orders in the order book.
        
        Counts all bid and ask orders across all price levels.
        
        Returns:
            float: Total number of orders in the book
        """
        bidVol = 0.
        askVol = 0.
        midPoint = (self.bestBid + self.bestAsk)/2.
        for i in range(self.nPrice):
            if ( self.bids[i]!=[]):
                for j in range(len(self.bids[i])):
                    bidVol += self.bids[i][j][0]
            if ( self.asks[i]!=[]):
                for j in range(len(self.asks[i])):
                    askVol += self.asks[i][j][0]
        totalOrders = askVol + bidVol
        return totalOrders

                    
    # utility to print the order book           
    def printBook(self):
        """
        Print a representation of the current state of the order book.
        
        Displays all non-empty price levels with their orders for both bid and ask sides.
        """
        for i in range(self.nPrice):
            if(self.bids[i] != []):
                print(i,self.realPrice(i),self.bids[i])
        print("------")
        for i in range(self.nPrice):
            if(self.asks[i] != []):
                print(i,self.realPrice(i),self.asks[i])


class agent:
    """
    A trading agent that participates in the limit order book market.
    
    This class models a market participant who makes forecasts about future prices
    using a combination of fundamental, chartist, and noise strategies. Based on these
    forecasts, the agent decides on desired holdings and places orders accordingly.
    
    The agent can follow different trading strategies based on the weights assigned to
    fundamental, chartist, and noise components. Pure strategy agents can be created by
    setting one weight to 1 and others to 0.
    """
    def __init__(self,sigmaF,sigmaM,sigmaN,kmax,Lmin,Lmax,forecastSet,beta,maxHold,tau,adaptiveK,simpleDemands,maxTrade,sigmaAgent,probMarketOrder):
        """
        Initialize a trading agent with specific strategy parameters and constraints.
        
        Parameters:
        sigmaF, sigmaM, sigmaN - Weights for fundamental, momentum/chartist, and noise strategies
        kmax - Maximum distance between order price and expected price
        Lmin, Lmax - Min/max range for trend following time horizons
        forecastSet - Reference to shared forecast information
        beta - Intensity of choice parameter that affects demand sensitivity
        maxHold - Maximum long or short position the agent can take
        tau - Order lifespan/cancellation period
        adaptiveK - Whether to adjust order pricing based on volatility
        simpleDemands - If True, use simple demand model from CI(2002)
        maxTrade - Maximum number of shares to trade at once
        sigmaAgent - Individual agent noise component standard deviation
        probMarketOrder - Probability of placing a market order versus limit order
        """
        # set strategy weights
        # set all positive :  this diverges from the paper a little
        a = 2.
        uRangeStart = 0.2
        if(sigmaF>0):
            # uniform range not less than zero
            urange = min(uRangeStart,sigmaF)
            urange = min(urange,1.-sigmaF)
            self.fundWeight = float(np.random.uniform(sigmaF-urange,sigmaF+urange,1))
        else:
            self.fundWeight = 0.
        if(sigmaM>0):
            urange = min(uRangeStart,sigmaM)
            urange = min(urange,1.-sigmaM)
            self.chartWeight = float(np.random.uniform(sigmaM-urange,sigmaM+urange,1))
        else:
            self.chartWeight = 0.
        if(sigmaN>0):
            urange = min(uRangeStart,sigmaN)
            urange = min(urange,1.-sigmaN)
            self.noiseWeight = float(np.random.uniform(sigmaN-urange,sigmaN+urange,1))
        else:
           self.noiseWeight = 0. 
        # Build simple agents
        self.type = 0
        simple = True  # simple/pure strat agents

        if(simple):
            xrnd = np.random.uniform(low=0.,high=1.,size=1)
            self.fundWeight = float(xrnd<sigmaF)
            self.chartWeight = float( (xrnd>sigmaF) and (xrnd<sigmaF+sigmaM))
            self.noiseWeight = float(xrnd>(sigmaF+sigmaM))
            self.type = int(0*self.fundWeight + 1*self.chartWeight + 2*self.noiseWeight)

         
        if(self.noiseWeight==1.0):
            self.probMarket = probMarketOrder
        else:
            self.probMarket = probMarketOrder
            
        normalize = self.fundWeight + self.chartWeight + self.noiseWeight
        self.fundWeight /= normalize
        self.chartWeight /= normalize
        self.noiseWeight /= normalize
        # horizons for momentum rules 
        self.l = np.random.randint(Lmin,Lmax)
        # Chart can be momentum or reversal
        self.chartSign = 1.
        self.fundamentalIndex = int(np.random.randint(forecastSet.nFundamental))
        self.volIndex = int(np.random.randint(forecastSet.nVol))
        self.beta = beta # intensity of choice
        # random component of spread
        self.k = kmax*np.random.rand()
        # Note:  k is not used:  Kappa is the final say on demand shading
        # Here it is allowed to cross the book (>0 value)
        # It is modified to ktilde in the order generation part
        self.kappa = float(np.random.uniform(low=-kmax, high =0*kmax, size=1))
        self.adaptiveK = adaptiveK
        self.simpleDemands = simpleDemands
        self.ktilde = self.k * np.random.rand()
        self.tau = tau
        # use this next line for heterogeneous tau (this works fine)
        # self.tau = np.random.randint(5,tau)
        self.fcast = 0.
        self.pfcast = 0.
        self.holdings = 0.
        self.cash = 0.
        self.orders   = 0.
        self.desHoldings = 0.
        # max long or short position
        self.maxHold = maxHold
        self.wealth = 0.
        self.bid = 0.
        self.ask = 0.
        # forecast adjustment weight
        self.fcastAdjust = 1.
        self.buying = False
        self.selling = False
        self.atHoldings = True
        self.maxTrade = maxTrade
        self.numTrade = 0
        self.timeToExecute = []
        self.agentNoise = 0.0
        # Fraction of aggregate noise term
        self.agentNoiseStd = sigmaAgent
        
    def updateFcast(self,forecast,price,tau,t):
        """
        Update agent's forecast based on fundamental, chartist and noise components.
        
        The forecast is a weighted sum of:
        1. Fundamental component - based on deviation from fundamental value
        2. Chartist component - based on recent price trends
        3. Noise component - random term
        
        If a recent fundamental change has occurred, agents gradually transition
        to the new fundamental value.
        
        Note: simple agents have weights that are in {0, 1} such that only 
        one forecast type is realized.

        Parameters:
        forecast - Forecast object with market forecast information
        price - Current market price
        tau - Time horizon
        t - Current time step
        """
        # weighted forecast value
        # self.fcast = ((1./self.tau)*self.fundWeight*forecast.fundamental+self.chartWeight*forecast.chartist[self.l] + \
        #     self.noiseWeight*forecast.noise)
        # self.fcast = (self.fundWeight*forecast.fundamental+self.chartWeight*forecast.chartist[self.l] + \
        #     self.noiseWeight*forecast.noise)
        # Generate agent-specific noise
        self.agentNoise = self.agentNoiseStd*np.random.standard_normal(1)[0]
        
        # Gradually transition to new fundamental value after a change
        if(np.random.uniform()< ((t-forecast.Changet)/250)):
            self.fcast = (self.fundWeight*forecast.fundamental[self.fundamentalIndex]+self.chartWeight*self.chartSign*forecast.chartist[self.l] + \
                          self.noiseWeight*(forecast.noise+self.agentNoise))
        else:
            self.fcast = (self.fundWeight*forecast.fundamentalold[self.fundamentalIndex]+self.chartWeight*self.chartSign*forecast.chartist[self.l] + \
                      self.noiseWeight*(forecast.noise+self.agentNoise))
        # BL Overide:  bound the forecast
        # self.fcast = min(self.fcast,0.05)
        # self.fcast = max(self.fcast,-0.05)
        # exponentiate the forecast to get future price forecast 
        # note:  this could have a variance adjustment, but it doesn't at the moment
        # self.pfcast = price*np.exp(self.tau*(self.fcast+0.5*forecast.v))
        
    def updateFcastold(self,forecast,price,tau):
        """
        Legacy forecast update method (kept for backward compatibility).
        
        Parameters:
        forecast - Forecast object
        price - Current market price
        tau - Time horizon
        """
        # weighted forecast value
        self.fcast = self.fcastAdjust*(self.fundWeight*forecast.fundamental+self.chartWeight*forecast.chartist[self.l] + \
            self.noiseWeight*forecast.noise)
        # bound the forecast
        # self.fcast = min(self.fcast,0.5)
        # self.fcast = max(self.fcast,-0.5)
        # exponentiate the forecast to get future price forecast 
        # note:  this could have a variance adjustment, but it doesn't at the moment
        self.pfcast = price*np.exp(self.tau*self.fcast+0.0*forecast.v)
        
    def shareDemand(self):
        """
        Calculate the agent's desired share position based on forecast.
        
        Note that the paper is only about agents with "simple" demand;
        this has been left in for future work.
        
        Returns:
        demand - The number of shares the agent wishes to hold
        """
        # simple demands replicate earlier CI paper, but on increase
        # and sell on decrease
        if(self.simpleDemands):
            # simple demand world should have zero orders
            self.orders = 0.
            if(self.fcast>0):
                demand = self.holdings + np.random.randint(1,self.maxTrade+1)
            else:
                demand = self.holdings - np.random.randint(1,self.maxTrade+1)
        else:
            # generate desired holding from hyperbol tangent
            # demand = self.maxHold*np.tanh(self.beta/self.maxHold*self.tau*self.fcast)
            start = np.sign(float(self.fcast))
            demand = 0.*start + self.beta*self.fcast
        if(demand>self.maxHold):
            demand = self.maxHold
        if(demand<-self.maxHold):
            demand = -self.maxHold
        return demand
        
    def processBuy(self,counterParty,tradePrice,quant,tradeTime):
        """
        Process a buy transaction with a counterparty.
        
        Updates agent and counterparty holdings, cash, and trade statistics.
        
        Parameters:
        counterParty - The agent on the other side of the trade
        tradePrice - Price at which the trade occurs
        quant - Quantity traded
        tradeTime - Time taken to execute the trade
        """
        self.holdings += quant
        self.cash -= tradePrice*quant
        self.numTrade += 1
        counterParty.holdings  -= quant
        counterParty.cash += tradePrice*quant
        counterParty.orders    += quant
        counterParty.numTrade += 1
        counterParty.timeToExecute.append(tradeTime)
        
    def processSell(self,counterParty,tradePrice,quant,tradeTime):
        """
        Process a sell transaction with a counterparty.
        
        Updates agent and counterparty holdings, cash, and trade statistics.
        
        Parameters:
        counterParty - The agent on the other side of the trade
        tradePrice - Price at which the trade occurs
        quant - Quantity traded
        tradeTime - Time taken to execute the trade
        """
        self.holdings -= quant
        self.cash += tradePrice*quant
        self.numTrade += 1
        counterParty.holdings += quant
        counterParty.cash -= tradePrice*quant
        counterParty.orders   -= quant
        counterParty.numTrade += 1
        counterParty.timeToExecute.append(tradeTime)
        
    def getAgentOrder(self,forecast,price,portfolioAdj,orderSigma,entryProb,t,marketBook):
        """
        Generate an order based on the agent's forecast and desired position.
        
        Note that `tauHorizon` is the forecast horizon (h) in Table 1: Model parameters
        in the paper.
        
        Determines:
        1. Whether to buy or sell
        2. Order price (limit or market)
        3. Number of shares to trade
        
        Parameters:
        forecast - Forecast object with market forecast information
        price - Current market price
        portfolioAdj - Whether to adjust based on portfolio
        orderSigma - Standard deviation for order price noise
        entryProb - Probability of entering the market
        t - Current time step
        marketBook - Reference to the order book
        
        Returns:
        nShares - Number of shares to trade
        """
        # Calculate future price forecast
        tauHorizon = 0.33*self.tau
        vratioAdjust = [1.0,1.25,2.]
        self.pfcast = price*np.exp(tauHorizon*(self.fcast+0.5*forecast.v))
        # Get desired holdings from forecast
        self.desHoldings = self.shareDemand()
        # Calculate order price adjustment (ktilde) based on volatility
        if(self.adaptiveK):
            self.ktilde = self.kappa*forecast.vRatio[self.volIndex]*(1+orderSigma*np.random.randn())
        else :
            self.ktilde = self.kappa*(1+orderSigma*np.random.randn())
        # Reset buy/sell flags
        self.buying = False
        self.selling = False
        # desired holdings > holdings -> buy order
        # if ( (portfolioAdj) and ( self.desHoldings > self.holdings)) or ( (~portfolioAdj) and (self.fcast>0)):
        # For simple demands desHoldings = holdings + 1 for buy and holdings - 1 for sell
        # Determine whether to buy or sell based on desired position change
        if (self.desHoldings-(self.holdings+0.*self.orders)) > 0.5 :
        # if self.fcast>0.:
            self.buying = True
            nShares = int(round( self.desHoldings - (self.holdings+0.*self.orders)))
            nShares = min(nShares,self.maxTrade)
            self.atHoldings = False
            # Determine bid price - market order or limit order
            if((np.random.uniform()<self.probMarket) and (self.fundWeight==0)):
                # Market order - bid above best ask
                self.bid = 1.01*marketBook.bestAsk
            else:  
                # Limit order - bid based on forecast
                self.bid = self.pfcast * (1.+self.ktilde)
            # Special case for large price-forecast deviation with fundamental traders
            if((abs(price-self.pfcast)>25) and (np.random.uniform()<1) and (self.fundWeight==1)):
                #print(self.pfcast)
                self.bid = 1.01*marketBook.bestAsk
            # buying in falling market
            # place order at current price forecast
        elif (self.desHoldings-(self.holdings+0.*self.orders)) < -0.5 :
            self.selling = True
            nShares = int(round( (self.holdings+0.*self.orders) - self.desHoldings))
            nShares = min(nShares,self.maxTrade)
            self.atHoldings = False
            # Determine ask price - market order or limit order
            if((np.random.uniform()<self.probMarket) and (self.fundWeight==0)):
                # Market order - ask below best bid
                self.ask = 0.99*marketBook.bestBid
            else:
                # Limit order - ask based on forecast
                self.ask = self.pfcast*(1.-self.ktilde)  
            # Special case for large price-forecast deviation with fundamental traders
            if((abs(price-self.pfcast)>25) and (np.random.uniform()<1) and (self.fundWeight==1)):
                self.ask = 0.99*marketBook.bestBid
        else:
            # No position change needed
            self.atHoldings = True
            nShares = 0
        # print(self.buying,self.bid,self.ask,price,self.pfcast)
        return nShares


# helper routine for basic autocorrelations
def autocorr(x,m):
    """
    Calculate autocorrelation for a time series up to lag m.
    
    Parameters:
    -----------
    x : array_like
        Input time series data
    m : int
        Maximum lag for autocorrelation calculation
        
    Returns:
    --------
    result : ndarray
        Array of autocorrelation values from lag 0 to m
    """
    n = len(x)
    v = x.var()
    x2 = x-x.mean()
    r = np.correlate(x2,x2,mode="full")[(n-1):(n+m+1)]
    result = r/(n*v)
    return result

def fastxcorr(x,y,m):
    """
    Calculate cross-correlation between two time series up to lag m efficiently.
    
    Parameters:
    -----------
    x : array_like
        First time series
    y : array_like
        Second time series
    m : int
        Maximum lag for cross-correlation calculation
        
    Returns:
    --------
    z : ndarray
        Array of normalized cross-correlation values from lag -m to m
    """
    nx = len(x)
    z = np.zeros(2*m+1)
    stdprod = np.std(x)*np.std(y)
    mx = np.mean(x)
    my = np.mean(y)
    for i in range(m+1):
        z[i+m] = np.mean( (x[0:(nx-i)]-mx)*(y[i:nx]-my))
    for i in range(m):
        j = m-i
        z[i] = np.mean( (x[j:nx]-mx)*(y[0:(nx-j)]-my))
    z = z/stdprod
    return z

def fastautocorr(x,m):
    """
    Calculate autocorrelation for a time series up to lag m efficiently.
    
    Parameters:
    -----------
    x : array_like
        Input time series data
    m : int
        Maximum lag for autocorrelation calculation
        
    Returns:
    --------
    z : ndarray
        Array of autocorrelation values from lag 0 to m
    """
    nx = len(x)
    z = np.zeros(m+1)
    v = np.var(x)
    mx = np.mean(x)
    for i in range(0,m+1):
        z[i] = np.mean( (x[0:(nx-i)]-mx)*(x[i:nx]-mx))
    z = z/v
    return z

def fastautocorr1(x):
    """
    Calculate lag-1 autocorrelation for a time series efficiently.
    
    Parameters:
    -----------
    x : array_like
        Input time series data
        
    Returns:
    --------
    rho : float
        Lag-1 autocorrelation coefficient
    """
    # assuming mean = 0
    n = len(x)
    v = x.var()
    me = x.mean()
    cv =np.mean( ( x[0:(n-1)]-me)*(x[1:n]-me))
    rho = cv/v
    return rho
    
    
def normhist(xdata,nbins):
    """
    Create a histogram of data with a normal distribution overlay.
    
    This function plots a histogram of the provided data and overlays a normal
    distribution curve with the same mean and standard deviation.
    
    Parameters
    ----------
    xdata : array_like
        The data to be plotted in the histogram.
    nbins : int
        Number of bins for the histogram.
    
    Returns
    -------
    None
        The function produces a plot but does not return any values.
    """
    n, bins, patches = plt.hist(xdata,nbins,normed=1,facecolor='green', alpha=0.5)
    mu = np.mean(xdata)
    sigma = np.std(xdata)
    y = mlab.normpdf(bins,mu,sigma)
    plt.plot(bins,y,'r')
    
def reversion(xdata):
    """
    Calculate the mean reversion coefficient for a time series.
    
    This function computes the correlation coefficient between the series level
    and its first difference, which is a common measure of mean reversion.
    A negative correlation indicates mean reversion.
    
    Parameters
    ----------
    xdata : array_like
        Time series data to analyze for mean reversion.
    
    Returns
    -------
    float
        Correlation coefficient between the series and its first difference.
        Negative values indicate mean reversion.
    """
    xdiff = np.diff(xdata)
    rstat = np.corrcoef(xdata[0:-1],xdiff)
    return rstat[0,1]
    
def pltPrice(prices):
    """
    Create a simple time series plot of prices.
    
    Parameters
    ----------
    prices : array_like
        Price data to be plotted.
    
    Returns
    -------
    None
        The function produces a plot but does not return any values.
    """
    plt.clf()
    plt.plot(prices)
    plt.xlabel('Period(t)')
    plt.ylabel('Price')
    plt.grid()

def printHoldings(agentList):
    """
    Print the financial positions of all agents.
    
    Displays each agent's holdings, cash balance, desired holdings, 
    kappa (price adjustment parameter), buying status, and number of trades.
    
    Parameters
    ----------
    agentList : list
        List of agent objects to display information for.
    
    Returns
    -------
    None
        Output is printed to console.
    """
    for i in range(len(agentList)):
        print(agentList[i].holdings,agentList[i].cash,agentList[i].desHoldings,agentList[i].kappa,agentList[i].buying,agentList[i].numTrade)
 
def printActivity(agentList):
    """
    Print detailed activity information for all agents.
    
    Displays each agent's strategy weights (fundamental, chartist, noise),
    kappa parameter, buying status, number of trades, and execution times.
    
    Parameters
    ----------
    agentList : list
        List of agent objects to display information for.
    
    Returns
    -------
    None
        Output is printed to console.
    """
    for i in range(len(agentList)):
        print(agentList[i].fundWeight,agentList[i].chartWeight,agentList[i].noiseWeight,agentList[i].kappa,agentList[i].buying,agentList[i].numTrade,agentList[i].timeToExecute) 
        

def executionSummary(agentList):
    """
    Collect all trade execution times across all agents.
    
    This function gathers the execution times of all trades made by all agents
    into a single array for analysis of market execution dynamics.
    
    Parameters
    ----------
    agentList : list
        List of agent objects to collect execution times from.
    
    Returns
    -------
    numpy.ndarray
        Array containing all execution times from all agents.
    """
    xTime = []
    for i in range(len(agentList)):
        xTime.extend(agentList[i].timeToExecute)
        xTimeVec = np.array(xTime)
    return xTimeVec
        
        
def printAgentSummary(agentList):
    """
    Calculate the average strategy weights across all agents.
    
    This function computes the mean values of noise, fundamental, and trend
    (chartist) weights across the entire population of agents.
    
    Parameters
    ----------
    agentList : list
        List of agent objects to summarize.
    
    Returns
    -------
    tuple
        Three float values representing the average noise weight,
        average fundamental weight, and average trend weight.
    """
    n = len(agentList)
    noise = np.zeros(n)
    trend = np.zeros(n)
    fund  = np.zeros(n)
    for i in range(n):
        noise[i] = agentList[i].noiseWeight
        trend[i] = agentList[i].chartWeight
        fund[i]  = agentList[i].fundWeight
    return np.mean(noise), np.mean(fund), np.mean(trend)
 
def printAgents(fileName) :
    """
    Export agent data to a CSV file.
    
    This function writes information about each agent to a CSV file,
    including cash, holdings, strategy weights, and other parameters.
    
    Parameters
    ----------
    fileName : str
        Name of the file to write the data to.
    
    Returns
    -------
    None
        Data is written to a file.
    """
    fileName='baseAgents.csv'
    writeFile = open(fileName,'w')
    tsWrite = csv.writer(writeFile,delimiter=',',lineterminator='\n')
    tsWrite.writerow(['iagent','cash','shares','noise','fund','momentum','tau','k'])
    n = len(agentList)
    for i in range(n):
        tsWrite.writerow([i, agentList[i].cash, agentList[i].holdings, agentList[i].noiseWeight, agentList[i].fundWeight, agentList[i].chartWeight,float(agentList[i].tau),agentList[i].kappa])
    del tsWrite
    writeFile.close()
  
def printTS(fileName):
    """
    Export time series data to a CSV file.
    
    This function writes various time series data to a CSV file, including
    prices, returns, volatility measures, and market microstructure metrics.
    
    Parameters
    ----------
    fileName : str
        Name of the file to write the data to.
    
    Returns
    -------
    None
        Data is written to a file.
    """
    writeFile = open(fileName,'w')
    tsWrite = csv.writer(writeFile,delimiter=',',lineterminator='\n')
    tsWrite.writerow(['tindex','rPrice','rret','rRV','rRV2','rVol','rSpread','rbidDepth','raskDepth'])
    T = len(rret)
    for t in range(T):
        tsWrite.writerow([t,rPrice[t],rret[t],rRV[t],rRV2[t],rVol[t],rSpread[t],rbidDepth[t],raskDepth[t]])
    del tsWrite
    writeFile.close()
    
def getSlope(x,y):
    """
    Calculate the slope of the linear relationship between x and y.

    This function computes the slope of the best-fit line for the given x and y data points,
    using the formula: slope = covariance(x,y) / variance(x).

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.

    Returns:
    float: The calculated slope. Returns 0 if the variance of x is 0.
    """
    xc = x.copy()
    yc = y.copy()
    xc -= np.mean(x)
    yc -= np.mean(y)
    cv = np.sum( xc*yc )
    v  = np.sum( xc**2.)
    if (v>0):
        slope = cv/v
    else:
        slope = 0.
    return slope

    
def getSummedBids(bidsdf):
    """
    Process and analyze the bid side of the order book.

    This function takes the bid data from the order book, processes it to create a
    cumulative volume profile, and calculates various metrics including the slope
    of the volume curve.

    Parameters:
    bidsdf (array-like): The bid data from the order book.

    Returns:
    tuple: A tuple containing:
        - priceRange (array): The range of prices analyzed
        - volume (array): The cumulative volume at each price level
        - dvolume (array): The change in volume between price levels
        - slope (float): The average slope of the volume curve
    """
    bidVec = np.array(bidsdf)
    # Throw out boundary
    # bidVec = bidVec[1:]
    bidVec = bidVec[bidVec>0.]
    bestBid = np.max(bidVec)
    # Arbitrarily cut the book at its mid point out into bid range (do the same for asks below)
    # minBid  = np.min(bidVec[1:])/2.
    nMidOrder = int(len(bidVec)/2)
    # Measure slope out to midpoint
    minBid = bidVec[nMidOrder]
    # Use this line for full book
    fullBook = False
    if (fullBook):
        minBid = np.min(bidVec)
    # minBid  = np.min(bidVec[1:])
    priceRange = np.arange(bestBid,minBid,-5.)
    priceLen = len(priceRange)
    volume  = np.zeros(priceLen)
    for i in range(priceLen):
        volume[i] = np.sum(bidVec>=priceRange[i])  
    # volume[volume==0.] = 1.
    # Normalize volume 
    volume /= np.sum(volume)
    # Or logs as in N/S
    # 2 as log in N/S
    # volume = np.log(volume)
    # dvolume = np.append(np.diff(volume),0.)
    dvolume = np.diff(volume)
    # dvolume = (volume[1:]/volume[:-1]) - 1.
    # dvolume = np.diff(volume)/volume[0:-1]
    # dprice  = np.diff(priceRange)/priceRange[0:-1]
    # as in N/S
    # dprice = np.diff(np.log(priceRange))
    dprice = (priceRange[1:]/priceRange[0:-1])-1.
    if(len(dvolume)>0):
        slope = np.mean(dvolume/dprice)
    else:
        slope = 0
    # slope = getSlope(priceRange[:-1]/bestBid-1.,dvolume)
    # slope = getSlope(dprice,dvolume)
    # slope = getSlope(priceRange,volume)
    return priceRange, volume, dvolume, slope
        

def getSummedAsks(asksdf):
    """
    Process and analyze the ask side of the order book.

    This function is similar to getSummedBids, but for the ask side of the order book.
    It processes the ask data, creates a cumulative volume profile, and calculates
    various metrics including the slope of the volume curve.

    Parameters:
    asksdf (array-like): The ask data from the order book.

    Returns:
    tuple: A tuple containing:
        - priceRange (array): The range of prices analyzed
        - volume (array): The cumulative volume at each price level
        - dvolume (array): The change in volume between price levels
        - slope (float): The average slope of the volume curve
    """
    askVec = np.array(asksdf)
    askVec = askVec[askVec>0.]
    # Throw out boundary max ask 
    # askVec = askVec[:-1]
    bestAsk = np.min(askVec)
    nMidOrder = int(len(askVec)/2)
    # Measure slope out to midpoint
    maxAsk = askVec[nMidOrder]
    # Use this line for full book
    fullBook = False
    if(fullBook):
        maxAsk = np.max(askVec)
    # maxAsk = (np.max(askVec)-bestAsk)/2.+bestAsk
    priceRange = np.arange(bestAsk,maxAsk,5.)
    # print(priceRange)
    priceLen = len(priceRange)
    volume  = np.zeros(priceLen)
    for i in range(priceLen):
        volume[i] = np.sum(askVec<=priceRange[i]) 
    # volume[volume==0.] = 1.
    # normalize volume
    # 1 By total volume on book
    volume /= np.sum(volume)
    # 2 as log in N/S
    # volume = np.log(volume)
    # dvolume = np.append(np.diff(volume),0.)
    dvolume = np.diff(volume)
    # dvolume = (volume[1:]/volume[:-1]) - 1.
    # dvolume = np.diff(volume)/volume[0:-1]
    # percentage price changes
    # dprice  = np.diff(priceRange)/priceRange[0:-1]
    # as in N/S
    # dprice = np.diff(np.log(priceRange))
    dprice = (priceRange[1:]/priceRange[0:-1])-1.
    
    if(len(dvolume)>0):
        slope = np.mean(dvolume/dprice)
    else :
        slope = 0
    # slope = getSlope(priceRange,dvolume)
    # slope = getSlope(dprice,dvolume)
    # slope = getSlope(priceRange[:-1]/bestAsk-1.,dvolume)
    # slope = getSlope(priceRange,volume)
    return priceRange, volume, dvolume, slope       

    
def downSampleMean(x,istart,iend,delta):
    """
    Downsample a time series by taking the mean over fixed-size windows.

    This function reduces the resolution of a time series by averaging values
    over non-overlapping windows of size 'delta'.

    Parameters:
    x (array-like): The input time series to be downsampled.
    istart (int): The starting index of the range to be downsampled.
    iend (int): The ending index of the range to be downsampled.
    delta (int): The size of each downsampling window.

    Returns:
    numpy.ndarray: The downsampled time series.
    """
    n = len(x)
    j = 0
    xm = np.zeros(len(np.arange(istart,iend,delta)))
    for i in range(istart,iend,delta):
        if( (i+delta) < (iend) ):
            xm[j] = np.mean(x[i:(i+delta)])
        else:
            xm[j] = x[i]
        j+=1
    return(xm)



def getSummary(rret,price):
    """
    Calculate summary statistics for returns and price data.

    This function computes various statistical measures and properties
    of the input return and price time series.

    Parameters:
    rret (array-like): Array of returns.
    price (array-like): Array of prices.

    Returns:
    list: A list containing the following summary statistics:
        - Mean of returns
        - Annualized standard deviation of returns
        - Kurtosis of returns
        - Proportion of autocorrelations outside confidence bands
        - Mean absolute autocorrelation of absolute returns
        - First-order autocorrelation of prices
    """
    T = int(len(rret)/2)
    T2 = int(len(price)/2)
    sret = rret[T:]
    sprice = price[T:]
    m = np.mean(rret)
    s = np.sqrt(250.)*np.std(sret)
    k =  kurtosis(sret, fisher=False, bias=False)
    retacf = fastautocorr(sret,20)[1:]
    bbands = 1.96*archAdjust(sret,20)[1:]
    outsideBands = np.mean((np.abs(retacf)>=bbands))
    
    retacfsum = np.mean(np.abs(retacf))
    retacf = fastautocorr(np.abs(sret),20)[1:]
    retacfasum = np.mean(retacf)
    
    import statsmodels.api as sm
    lbstats = sm.stats.acorr_ljungbox(rret, lags=[10], return_df=True)
    lbpvalue = lbstats["lb_pvalue"].values[0]

    priceacf = fastautocorr(price,1)[1]
    
    return [m, s, k, outsideBands, retacfasum, priceacf]
        
# Adjust Bol bands for ARCH/GARCH  
def archAdjust(x,k):
    """
    Adjust autocorrelation confidence bands for ARCH/GARCH effects.

    This function calculates adjustment factors for autocorrelation
    confidence bands to account for ARCH/GARCH effects in the time series.

    Parameters:
    x (array-like): The input time series.
    k (int): The number of lags to calculate adjustments for.

    Returns:
    numpy.ndarray: Array of adjustment factors for each lag up to k.
    """
    T = len(x)
    x = x-np.mean(x)
    adj = np.zeros(k+1)
    for i in range(1,k):
        num = np.mean( x[0:(T-i)]**2 * x[i:T]**2)
        den = np.mean( x**2 )**2
        adj[i]= 1./np.sqrt(T)*num/den
    return adj


def modelIteration(nAgents, Tinit, Tmax, Lmin, Lmax, pf, deltaP, sigmae, kMax, adaptiveK, simpleDemands, maxTrade, tau, sigmaF, sigmaM, sigmaN, portfolioAdj, deltaT, lam, maxHold, beta, volLag,agentsPerPeriod,rhoBar,orderSigma,trimVol,probMarketOrder):
    """
    Execute a complete simulation of the limit order book model.
    
    This function implements a market simulation based on the Chiarella/Iori model (Quant Finance, 2002).
    It creates agents, initializes an order book, runs the simulation for the specified time period,
    and collects various market metrics throughout the process.
    
    Parameters:
    -----------
    nAgents : int
        Number of agents in the simulation
    Tinit : int
        Initial time periods for market warm-up/burn-in
    Tmax : int
        Maximum simulation time steps
    Lmin : int
        Minimum trend-following horizon for chartist strategies
    Lmax : int
        Maximum trend-following horizon for chartist strategies
    pf : float
        Initial fundamental price
    deltaP : float
        Price tick size for the order book
    sigmae : float
        Standard deviation of noise in forecasts
    kMax : float
        Maximum order placement deviation from forecast price
    adaptiveK : bool
        Whether to adapt order placement based on local volatility
    simpleDemands : bool
        Whether to use simple demand functions as in CI(2002)
    maxTrade : int
        Maximum number of shares per trade
    tau : int
        Order lifetime before cancellation
    sigmaF : float
        Weight for fundamental strategy component
    sigmaM : float
        Weight for momentum/chartist strategy component
    sigmaN : float
        Weight for noise trading component
    portfolioAdj : bool
        Whether agents adjust based on portfolio holdings
    deltaT : int
        Time period for downsampling/aggregation
    lam : float
        Probability of random agent selection
    maxHold : int
        Maximum position size (long or short) for agents
    beta : float
        Intensity of choice parameter for demand function
    volLag : int
        Lag window for volatility estimation
    agentsPerPeriod : int
        Number of agents activated each period
    rhoBar : float
        Target autocorrelation for price process
    orderSigma : float
        Noise in order placement
    trimVol : bool
        Whether to trim outliers in volatility estimation
    probMarketOrder : float
        Probability of placing market orders instead of limit orders
    
    Returns:
    --------
    Multiple time series including:
        price - Price series
        ret - Returns
        rret - Downsampled returns
        rRV - Realized volatility
        rRV2 - Alternative realized volatility measure
        And many other market metrics and agent information
    """
    agentList = []
    # Create dataframes to store the order book state
    bidsdf=np.zeros((3000,1))
    asksdf=np.zeros((3000,1))
    # price, return, and volume time series
    # Initialize time series arrays for tracking various metrics
    price = pf*np.ones(Tmax+1)
    pft    = np.ones(Tmax+1)
    entryProb = float(agentsPerPeriod)/float(nAgents)
    logPriceFund = np.zeros(Tmax+1)
    ret   = np.zeros(Tmax+1)
    spread = np.zeros(Tmax+1)
    bidDepth = np.zeros(Tmax+1)
    askDepth = np.zeros(Tmax+1)
    rbidDepth = np.zeros(int((Tmax+1)/100))
    raskDepth = np.zeros(int((Tmax+1)/100))
    bidSlope = np.zeros(Tmax+1)
    askSlope = np.zeros(Tmax+1)
    totalV = np.zeros(Tmax+1)
    totalRV = np.zeros(Tmax+1)
    totalRV2 = np.zeros(Tmax+1)
    totalCRV = np.zeros(Tmax+1)
    totalOrder = np.zeros(Tmax+1)
    totalOrdersOnBook = np.zeros(Tmax+1)
    orderFlow = np.zeros(Tmax+1)
    rPrice = np.zeros(int((Tmax+1)/100))
    rRV    = np.zeros(int((Tmax+1)/100))
    rCRV    = np.zeros(int((Tmax+1)/100))
    rRV2   = np.zeros(int((Tmax+1)/100))
    rSpread = np.zeros(int((Tmax+1)/100))
    portDev = np.zeros(Tmax+1)
    rportDev = np.zeros(int((Tmax+1)/100))
    holdings = np.zeros(nAgents)
    orders = np.zeros(nAgents)
    dholdings = np.zeros(nAgents)
    holdingsts = np.zeros(Tmax+1)
    dholdingsts = np.zeros(Tmax+1)
    rholdingsts = np.zeros(int((Tmax+1)/100))
    rdholdingsts = np.zeros(int((Tmax+1)/100))
    holdingsDiff = np.zeros(nAgents)
    holdingsDiffABS = np.zeros(nAgents)
    holdingsDist = np.zeros(shape=(Tmax+1,250))
    holdingsDistABS = np.zeros(shape=(Tmax+1,250))
    ac = np.zeros(10)
    print('beta',beta)

    # Create set of forecasts with shared parameters
    forecastSet = forecasts(Lmax,pf,sigmae,volLag)
    # Create agents with heterogeneous parameters
    for i in range(nAgents):
        agentList.append(agent(sigmaF,sigmaM,sigmaN,kMax,Lmin,Lmax,forecastSet,beta,maxHold,tau,adaptiveK,simpleDemands,maxTrade,0.*sigmae,probMarketOrder))
    # Create order book with initial state
    marketBook = orderBook(600.,1400.,deltaP,agentList,nAgents,Tinit,tau)
    # Set up initial prices during burn-in period
    price[0:Tinit] = pf*(1.+0.001*np.random.randn(Tinit))
    logPriceFund[0:Tinit] = np.log(price[0:Tinit]/pf)
    ret[0:Tinit] = 0.001*np.random.randn(Tinit)
    
    print("At the loop...")
    randomAgent = agentList[np.random.randint(0,nAgents)]
    # Calculate initial price autocorrelation
    rho = fastautocorr1(logPriceFund[0:Tinit])
    rho = rhoBar
    # Main simulation loop
    for t in range(Tinit,(Tmax-1)):

        # Check for potential fundamental value changes (uncommon events)
        pfChange = 0.000
        if(np.random.rand()<pfChange and t>80000 and t<150000):
            print("Adjust")
            forecastSet.changeFundamental(t)
        pft[t] = forecastSet.pf
        # Zero out agents after burnin period
        # Reset agent holdings after initial burn-in period if needed
        if(t == (Tinit+2000)):
            for agentx in agentList:
                agentx.holdings=0.
                agentx.cash=0.
        # Determine if fundamental has recently changed
        if(pft[t]!=pft[t-250]):
            rhoadjust=True
        else:
            rhoadjust=False
        # Periodically update the price autocorrelation estimate
        if((np.random.rand()<0.01) and (t>(Tinit+100))):
            if( t < Tinit+10000):
                acLag = t-Tinit
            else :
                acLag = 10000
            trueRho = fastautocorr1(logPriceFund[acLag:t])
            rho = trueRho
            if(rho<0.98):
                rho = 0.98
            if(rho>rhoBar):
                rho = rhoBar
        # Update forecasts for all agents based on current state
        forecastSet.updateForecasts(t,Tinit,price[t],ret,rho,trimVol,spread,deltaP,rhoadjust,noiseOn=True)
        tradePrice = -1
        # Initialize desired holdings for agents at beginning
        if(t==Tinit) :
            for i in range(0,nAgents):
                agentList[i].desHoldings = agentList[i].shareDemand()
        # Trading activity for this period
        inPeriodPrice = price[t]
        volSum = 0
        volSumBuy = 0
        volSumSell = 0
        orderSum = 0
        nBuys = 0
        nSells = 0
        # Loop through multiple agents per period
        for iagent in range(0,agentsPerPeriod):            
            randomAgent = agentList[np.random.randint(0,nAgents)]
            # Update agent forecast and determine trading decision
            randomAgent.updateFcast(forecastSet,inPeriodPrice,tau,t)
            nShares = randomAgent.getAgentOrder(forecastSet,inPeriodPrice,portfolioAdj,orderSigma,entryProb,t,marketBook)   
            tradePrice = -1
            # potential buyer
            lastTradePrice = -1.
            # Process agent's buy orders if applicable
            if randomAgent.buying:
                # add bid or market order for all share demand
                for iShare in range(0,nShares):
                    tnoise = 0.0
                    orderSum += 1
                    tradePrice = marketBook.addBid(randomAgent,1.,t,tnoise)
                    if(tradePrice != -1):
                        volSum += 1
                        volSumBuy += 1
                        lastTradePrice = tradePrice
                    
            # Process agent's sell orders if applicable
            if randomAgent.selling:
                # seller: add ask, or market order
                for iShare in range(0,nShares):
                    tnoise = 0.0
                    orderSum += 1
                    tradePrice = marketBook.addAsk(randomAgent,1.,t,tnoise)
                    if(tradePrice != -1):
                        volSum += 1
                        volSumSell += 1
                        lastTradePrice = tradePrice
            # Update in-period price based on trading activity
            if tradePrice == -1:
                inPeriodPrice = (marketBook.bestBid+marketBook.bestAsk)/2.
            else:
                inPeriodPrice = tradePrice
                # Test for bid/ask bounce effect
                # inPeriodPrice = (marketBook.bestBid+marketBook.bestAsk)/2.
            # Calculate return within period
            ret[t+1] = np.log(inPeriodPrice/price[t])
            spread[t+1] = marketBook.bestAsk - marketBook.bestBid
        # After all agents have traded, update price for next period
        if lastTradePrice == -1:
            price[t+1]=(marketBook.bestBid + marketBook.bestAsk)/2.
            # totalV[t+1]=totalV[t]
        else:
            # trade
            price[t+1] = lastTradePrice
        # Update metrics
        logPriceFund[t+1] = np.log(price[t+1]/forecastSet.pf)
        totalV[t+1] = volSum+totalV[t]
        totalOrder[t+1] = orderSum + totalOrder[t]
        orderFlow[t+1] = volSumBuy - volSumSell
        
        spread[t+1] = marketBook.bestAsk - marketBook.bestBid
        # returns
        ret[t+1]=np.log(price[t+1]/price[t])
        # Realized vol measures 
        # build up sums over time
        # these can then be differenced to get the RV over any given period (like volume)
        # Volatility measures
        totalRV[t+1] = totalRV[t] + ret[t+1]**2.
        totalCRV[t+1] = totalCRV[t] + (ret[t+1]**2.)*volSum
        totalRV2[t+1] = totalRV2[t] + np.pi/2.*np.abs(ret[t])*np.abs(ret[t+1])
        # clear book
        # Clean stale orders from the book (with probability 1.0)
        if(rnd.rand()<1.0):
            marketBook.cleanBook(t,tau,agentList)
        # Measure current order book depth
        bidDepth[t+1], askDepth[t+1] = marketBook.bookDepth(forecastSet)
        totalOrdersOnBook[t+1] = marketBook.totalOrders()
        
        # This portion will output the dynamics order book
        # This code records the entire book at time t, and stores in bidsdf[:,t], asksdf[:,t]
        orderbookbids=marketBook.bids
        bids=[]
        for prc in range(0,len(orderbookbids)):
            if orderbookbids[prc] != []:
                for numOrd in range(0,len(orderbookbids[prc])):
                    bids.append(prc)
        #Turn descrete to real bids
        realbids=[(i*deltaP)+600 for i in bids]     
        
        for i in range(0,3000-len(realbids)):
            realbids.append(0)
        
        # Store only current state to save memory
        bidsdf[:,0]=realbids
        
        orderbookasks=marketBook.asks
        asks=[]
        for prc in range(0,len(orderbookasks)):
            if orderbookasks[prc] != []:
                for numOrd in range(0,len(orderbookasks[prc])):
                    asks.append(prc)
        #Turn descrete to real bids
        realasks=[(i*deltaP)+600 for i in asks]     
        
        for i in range(0,3000-len(realasks)):
            realasks.append(0)
        
        # Store current ask side
        asksdf[:,0]=realasks
        
        # Estimate Bid and Ask slopes on the book (again modified to only store slope for each t)
        
        # Calculate order book slope metrics
        priceRange, volume, dvolume, bidSlope[t] = getSummedBids(bidsdf[:,0])
        priceRange, volume, dvolume, askSlope[t] = getSummedAsks(asksdf[:,0])

        # ***** End order book data creation

        # find mean deviation holdings, desired holdings
        # Track agent holdings and desired positions
        for i in range(0, nAgents):
            holdings[i] = agentList[i].holdings
            dholdings[i] = agentList[i].desHoldings
            orders[i] = agentList[i].orders
            # this is here in case someone hit one of agent i's orders
            # moving agent i in range
            # Check if agent's current position matches desired position
            if abs((holdings[i]+orders[i])-dholdings[i])<0.5:
                agentList[i].atHoldings = True
            # Track distribution of position deviations
            holdingsDiff[i] = (int(agentList[i].holdings) - int(agentList[i].desHoldings))
            holdingsDiffABS[i] = abs(int(agentList[i].holdings) - int(agentList[i].desHoldings))
            holdingsDist[t][int(holdingsDiff[i]) + 10] = holdingsDist[t][int(holdingsDiff[i]) + 10] + 1
            holdingsDistABS[t][int(holdingsDiffABS[i])] = holdingsDistABS[t][int(holdingsDiffABS[i])] + 1
        
        # Periodically report progress
        if (t % 1000) == 0:
            print(" T is currently = ", t)
        
        # Calculate aggregate metrics for this period
        # 1. Position deviation
        # 2. Average holdings
        # 3. Average desired holdings
        portDev[t] = np.mean(np.abs(holdings+orders-dholdings))
        holdingsts[t] = np.mean(holdings)
        dholdingsts[t] = np.mean(dholdings)

    # generate long run values for time series    
    # Create downsampled/aggregated time series for analysis
    rVol    = np.diff(totalV[range(Tinit+deltaT,Tmax,deltaT)])
    rOrders    = np.diff(totalOrder[range(Tinit+deltaT,Tmax,deltaT)])
    rPrice = price[ range(Tinit+deltaT,Tmax,deltaT)]
    price = price[  range(Tinit+deltaT,Tmax,1)]
    ret   = np.diff(np.log(price))
    rpf    = pft[ range(Tinit+deltaT,Tmax,deltaT)]
    rSpread = spread[ range(Tinit+deltaT,Tmax,deltaT)]
    rbidDepth = bidDepth[ range(Tinit+deltaT,Tmax,deltaT)]
    raskDepth = askDepth[ range(Tinit+deltaT,Tmax,deltaT)]
    rtotalOrdersOnBook = totalOrdersOnBook[range(Tinit+deltaT,Tmax,deltaT)]
    # Calculate downsampled slope metrics
    rbidSlope = downSampleMean(bidSlope,Tinit+deltaT,Tmax,deltaT)
    raskSlope = downSampleMean(askSlope,Tinit+deltaT,Tmax,deltaT)
    
    # Downsampled agent position metrics
    rholdingsts = holdingsts[ range(Tinit+deltaT,Tmax,deltaT)]
    rdholdingsts = dholdingsts[ range(Tinit+deltaT,Tmax,deltaT)]
    rportDev = portDev[ range(Tinit+deltaT,Tmax,deltaT)]
    rportDev = rportDev[1:]
    # Calculate returns and volatility on downsampled series
    rret   = np.diff(np.log(rPrice))
    rRV    = np.diff(totalRV[range(Tinit+deltaT,Tmax,deltaT)])
    rRV2    = np.diff(totalRV2[range(Tinit+deltaT,Tmax,deltaT)]) 
    rCRV  = np.diff(totalCRV[range(Tinit+deltaT,Tmax,deltaT)]) - np.mean(rRV)*np.mean(rVol)
    # Autocorrelation of returns
    ac  = autocorr(rret,2)[1]
    rretKurt = kurtosis(rret)
    # pricing and depth information is now off by 1 large period
    # Since return is return t->t+1, and all of these refer to time t information
    # need to readjust:  this is important for timing
    # Adjust timing of certain metrics to align with returns
    # (Since return is t->t+1, but other metrics are for time t)
    rPrice = rPrice[1:]
    rSpread = rSpread[1:]
    rbidDepth = rbidDepth[1:]
    raskDepth = raskDepth[1:]
    raskSlope = raskSlope[1:]
    rbidSlope = rbidSlope[1:]

    # We are looking at the volatility vs the returns 
    # Volatility measure is noisy, so take a moving average 

    # Analyze volatility clustering
    X = rret**2
    
    T_window = 10

    # Calculate moving averages of volatility with different window sizes
    X=pd.DataFrame(X)
    Xsmooth_trailing=X.rolling(window=T_window, center=False).mean()  
    Xsmooth_centered=X.rolling(window=T_window, center=True).mean()  
    Xsmooth_centered=Xsmooth_centered.to_numpy()
    Xsmooth_trailing=Xsmooth_trailing.to_numpy()
    
    #fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    #ax1.plot(rportDev, 'b-')
    #ax2.plot(Xsmooth_trailing, 'r-')
    #ax2.plot(Xsmooth_centered, 'r-.')
    #ax2.plot(X, 'g-.')
    ##ax2.set_ylabel('sin', color='r');  ax1.set_xlabel('time (s)'); ax1.set_ylabel('exp', color='b')
    # plt.show()

    #fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    #ax1.plot(rportDev, 'b-')
    #ax2.plot(Xsmooth_centered, 'r-')
    #ax2.plot(Xsmooth_trailing, 'r-.')
    ##ax2.set_ylabel('sin', color='r');  ax1.set_xlabel('time (s)'); ax1.set_ylabel('exp', color='b')
    # plt.show()
    
    # Extract non-NaN values for correlation analysis
    rportDev=pd.DataFrame(rportDev)
    rportDev=rportDev.to_numpy()
    #non_nan_idx = np.isfinite(Xsmooth_centered)
    
    # Calculate correlation between portfolio deviation and volatility
    rportDev_non_nan = rportDev[np.isfinite(Xsmooth_centered)]
    Xsmooth_centered_non_nan = Xsmooth_centered[np.isfinite(Xsmooth_centered)]

    #fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    #ax1.plot(rportDev_non_nan, 'b-')
    #ax2.plot(Xsmooth_centered_non_nan, 'r-')
    #ax2.set_ylabel('sin', color='r');  ax1.set_xlabel('time (s)'); ax1.set_ylabel('exp', color='b')
    #plt.title("Centered MA")
    #plt.show()
    
    print("Centered MA: pearsonr:", stats.pearsonr(rportDev_non_nan, Xsmooth_centered_non_nan))


    #****Trailing:****
    # Repeat for trailing moving average
    rportDev_non_nan = rportDev[np.isfinite(Xsmooth_trailing)]
    Xsmooth_trailing_non_nan = Xsmooth_trailing[np.isfinite(Xsmooth_trailing)]

    # wealth summary over pure strategy types
    # Calculate average wealth by agent type
    wealthByType = np.zeros(3)
    typeCount = np.zeros(3)
    lastPrice = price[-1]
    for agentx in agentList:
        wealthByType[agentx.type] += agentx.holdings * lastPrice + agentx.cash
        typeCount[agentx.type] += 1.0
    for i in range(3):
        if(typeCount[i]>0):
            wealthByType[i] /= typeCount[i]
        
    
    #fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    #ax1.plot(rportDev_non_nan, 'b-')
    #ax2.plot(Xsmooth_centered_non_nan, 'r-')
    ##ax2.set_ylabel('sin', color='r');  ax1.set_xlabel('time (s)'); ax1.set_ylabel('exp', color='b')
    #plt.title("Trailing MA")
    #plt.show()
    print("Trailing MA: pearsonr:", pearsonr(rportDev_non_nan, Xsmooth_trailing_non_nan))
    
    print("DeltaT:",deltaT,Tinit,Tmax)

    # Return large collection of results and metrics
    return price, ret, rret, rRV, rRV2, totalV, rPrice, rSpread, rbidDepth, raskDepth, portDev, rportDev, holdings, dholdings, orders, rportDev_non_nan, Xsmooth_centered_non_nan, Xsmooth_trailing_non_nan, holdingsDiff, holdingsDiffABS, holdingsDist, holdingsDistABS, rVol, rOrders, marketBook, agentList, forecastSet, rdholdingsts, bidsdf, rbidSlope, asksdf, raskSlope, wealthByType, rpf, logPriceFund, rCRV, rtotalOrdersOnBook, orderFlow, pft;


# Start Experiment Setup ----------------------------------------------
# Set default model setup parameters

# Total number of agents in the simulation
nAgents = 2000
# Initial time periods (burn-in phase)
Tinit = 10000
Tinit = 10000
# Maximum simulation time periods
Tmax =  200000
# Tmax =  400000
# Range for analyzing summary statistics, excluding initial burn-in
sumRange = range(50000,Tmax)

# Number of ticks per time period for data aggregation
deltaT = 1
deltaT = 50
deltaTList = [deltaT]


# Range parameters for trend-following rules
# Minimum lookback period for chartist strategies
Lmin = 10
# Maximum lookback period for chartist strategies
Lmax = 500

# Fundamental price of the asset
pf = 1000.
# Price tick size for order book discretization
deltaP = 0.025
# deltaP = 0.1


# use simple demands as in CI(2002)
# False yields more complex multi-share demands
# Flag to use simple demands as in Chiarella & Iori (2002)
# True: agents submit orders for fixed number of shares
# False: agents submit more complex multi-share demands
simpleDemands = True
# Flag to determine if agents adjust portfolios based on desired holdings
portfolioAdj = False

# Agent selection parameter:
# For lam = 1, agents are selected completely randomly
# For lam < 1, there is selection bias toward agents far from desired holdings
lam = 1.00 # can't used lambda for obvious reasons

# Simulation Run Number
######### DO NOT TOUCH 
numRuns = 0

# Number of simulations
# Touch
interation = 1

maxHoldList = [3, 5, 10, 25, 50, 100]
# List of maximum position sizes for agents
maxHoldList = [50.]
# Simple demands trade [0,maxTrade] shares

betaList = [3, 5, 10, 25, 50, 100]
# List of intensity of choice parameters for hyperbolic tangent demand function
# Note: not used in the paper, retained for future work.
betaList = [3.]

# forecasting parameters 
# noise forecaster 
# Noise forecaster standard deviation values to test
sigmaeList = [0.00016, .01, .25, .75]
# sigmaeList = [0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001]
sigmaeList = [0.0001]

# Strategy weight parameters (sum to 1.0)
# Weight for fundamental trading strategy
sigmaF = 0.45
sigmaF = 0.10
# Weight for trend-following/chartist trading strategy
sigmaM = 0.00

# really good:  sigmaF = 0.40, sigmaM = 0.40
# sigmaN = 0.6 
# Weight for noise trading strategy (calculated as remainder)
sigmaN = 1. - sigmaF - sigmaM


# number of periods before cancel
# List of order lifetimes (tau values) to test
tauList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
tauList = [50]
tauList = [15]
tauList = [10,25,50,250]
# Standard run tau
tauList = [25]
# Test tau
# tauList = [75]
# Alternative tau
# tauList = [15]



# bid or ask distance from expected future price
# Maximum bid/ask distance from expected future price (kappa parameter)
kMax = 0.15
kMax = 0.1
kMax = 0.2
kMax = 0.1



# Probability of submitting a market order rather than limit order
probMarketOrder = 0.05


# Fundamental forecast parameter 
#   Also price autocorrelation
# Mean reversion parameter for fundamental price process
# Also price autocorrelation parameter
# rhoBar = 0.9997
rhoBar = 0.9995
rhoBar = 0.9995
rhoBar = 0.999


# Noise on order (done to generate volume)
# Standard deviation of random noise added to order prices
orderSigma = 0.0
# Forecasts trim large moves for local variance estimates?
# Flag to trim large moves when estimating local variance
trimVol = False
# Order placement depends on local variance
# Flag to adjust order placement based on local volatility
adaptiveK = True
# lag length (high freq units) for recent vol estimate
# Lookback window for recent volatility estimation
volLag = 150
volLag = 150


# shares traded uniform [1,maxTrade]
# Maximum number of shares per trade (uniform between 1 and maxTrade)
maxTrade = 1
# number of agents arriving each period
# Number of agents arriving to market each period
agentsPerPeriod = 10
agentsPerPeriod = 20
agentsPerPeriod = 30

# set parameter sweeps
# Various parameter lists for sensitivity analysis


# volLagList = [25,50,100,150,250, 500,750]
volLagList = [150, 150, 150, 150, 150, 150, 150]
# agentsPerPeriodList = [20, 20, 20, 20,20, 20, 20]
agentsPerPeriodList = [30,30, 30, 30,30, 30, 30]
deltaTList = [1,1,1,1,1,1,1]
# sigmaFList = [0.05, 0.10, 0.15, 0.20,0.30,0.40,0.50]
sigmaFList = [0.20,0.20,0.2,0.2,0.2,0.2,0.2]
kMaxList =   [0.1, 0.1, 0.1, 0.1,0.1,0.1,0.1]
# kMaxList = [0.01,0.025, 0.05,0.1, 0.25, 0.5, 1.0]


# nListParams = len(kMaxList)
# Number of parameter combinations to test
nListParams = 1

# Optionally scale noise by sqrt(agentsPerPeriod)
# for i in range(len(sigmaeList)):
#     sigmaeList[i] /= np.sqrt(float(agentsPerPeriod))

# pr = cProfile.Profile()
# pr.enable()
# Performance profiling setup (commented out)
# End Experiment Setup ------------------------------------------------

# Start simulation execution
t0 = time()

# Initialize summary output file
summaryFile = open('runSummary.csv','w')
summaryWrite = csv.writer(summaryFile,delimiter=',',lineterminator='\n')
summaryWrite.writerow(['tau', 'sigmae', 'srret',   'acRret', 'acPrice', 'acRVMean', 'acabsRet', 'rretKurt', 'rVol', 'rSpread', 'rbidDepth', 'raskDepth'] )


# This is the random varaible portion of the code 
# If using a grid, set slurm array size
slurm_array=2 #This is the maximum number of array's we would ever set up in SLURM 
# Setup for random seed generation to ensure reproducibility
# Standard run seed
seed_of_seeds = 4242
# Used for homogeneuous fundamentals case
# seed_of_seeds = 42
seed_rng = np.random.RandomState(seed_of_seeds)
# seed_rng = np.random.RandomState(seed=None)
# Generate an array of random seeds for all iterations and parameter combinations
all_seeds = seed_rng.randint(0, 2**5-1, size=interation*slurm_array) #
parameters_and_seed_index = []


# Create parameter combinations with seed indices
# Used to select parameter combination when executing 
# many parameter combinations on a slurm grid
for ffff in range(len(tauList)):
    for fff in range(len(sigmaeList)):
        for i in range(len(all_seeds)):
            parameters_and_seed_index.append((ffff, fff, i))     

# Main loop for running simulations with different parameter combinations
for ffff in range(len(tauList)):
    for fff in range(len(sigmaeList)):
        j=0
        for ff in range(numRuns, numRuns + interation):
        
            # Select the appropriate seed for this parameter combination and iteration
            temp_seed=all_seeds[parameters_and_seed_index[(slurm_it-1)*interation+j][2]]
            #temp_seed=all_seeds[parameters_and_seed_index[(ffff,fff,i)]]
            j+=1
            np.random.seed(temp_seed)
            # Run the model with current parameter settings
            price, ret, rret, rRV, rRV2, totalV, rPrice, rSpread, rbidDepth, raskDepth, portDev, rportDev, holdings, dholdings, orders, rportDev_non_nan, Xsmooth_centered_non_nan, Xsmooth_trailing_non_nan, holdingsDiff, holdingsDiffABS, holdingsDist, holdingsDistABS, rVol, rOrders, marketBook, agentList, forecastSet, rdholdingsts, bidsdf, rbidSlope, asksdf, raskSlope, wealthByType, rpf, logPriceFund, rCRV, rtotalOrdersOnBook, orderFlow,pft = modelIteration(nAgents, Tinit, Tmax, Lmin, Lmax, pf, deltaP, float(sigmaeList[fff]), kMax, adaptiveK, simpleDemands, maxTrade, int(tauList[ffff]), sigmaF, sigmaM, sigmaN, portfolioAdj, deltaT, lam, int(maxHoldList[0]), betaList[0], volLag, agentsPerPeriod, rhoBar, orderSigma, trimVol, probMarketOrder) 
              
            t0 = time()
            # Output order book data (commented out to save space)
            #pd.DataFrame(bidsdf).to_csv("orderbookbids"+str(ff)+".csv", encoding='utf-8')
            #pd.DataFrame(asksdf).to_csv("orderbookasks"+str(ff)+".csv", encoding='utf-8')
            t1 = time()
            print("Just ran one iteration; cummulative time = ", (t1-t0)/60/60, "hr,\tor", (t1-t0)/60, "min,\tor", t1-t0, "sec")
            
            # Calculate statistics on second half of simulation to avoid initialization effects
            testRange = range(int( len(rret)/2),len(rret))
            acRret  = fastautocorr1(rret[testRange])
            acPrice = fastautocorr1(price)
            acRV = fastautocorr(rRV2[testRange],20)
            acabsRet = fastautocorr(abs(rret[testRange]),20)
            acRVMean = np.mean(acRV[1:])
            acabsRetMean = np.mean(acabsRet[1:])
            rretKurt = kurtosis(rret[testRange])
            # Write detailed time series output to CSV file
            shoklossset1 = open("dataOutputFile"+str(ff)+".csv", 'w')
            writer = csv.writer(shoklossset1, delimiter=',', lineterminator='\n')
            for r in range(0, int((Tmax - Tinit) / deltaT) - 3):
               writer.writerow((price[r], ret[r], rret[r], rRV[r], rRV2[r], rCRV[r], acRVMean, rVol[r], rPrice[r], rSpread[r], rtotalOrdersOnBook[r], rbidDepth[r], rbidSlope[r], raskDepth[r], raskSlope[r], tauList[ffff], sigmaeList[fff],temp_seed,pft[r] ))
            shoklossset1.close()
      
            numRuns = numRuns + 1

            # Write summary statistics for this parameter combination to summary file
            summaryWrite.writerow([tauList[ffff],sigmaeList[fff],np.std(rret[testRange])*np.sqrt(250.),acRret,acPrice,acRVMean, acabsRetMean,rretKurt,np.mean(rVol[testRange]),np.mean(rSpread[testRange]),np.mean(rbidDepth[testRange]),np.mean(raskDepth[testRange])])
            print(j)
            
# Close summary file
del summaryWrite
summaryFile.close()
 
# Calculate high-frequency realized volatility
rvhf = np.zeros(len(rret))  
j = 0;         
for t in range(Tinit+deltaT,Tmax-deltaT,deltaT):
    rvhf[j] = np.mean(abs(ret[t-deltaT:t]))
    j+=1


# Combine output files from all simulation runs
for mr in range(0, numRuns):
    # Read individual output file for this simulation run
    shockloss = csv.reader(open("dataOutputFile"+str(mr)+".csv","r"),delimiter=',')
    shockloss = list(shockloss)
    if mr == 0:
        # For the first run, create a new combined output file with headers
        shoklossset = open("dataOutputFileComplete.csv", 'w')
        writer = csv.writer(shoklossset, delimiter=',', lineterminator='\n')
        writer.writerow(("price", "ret", "rret", "rRV", "rRV2", "autocorrelation sum", "rvol", "rPrice", "spread", "bid depth", "ask depth", "tau", "sigmaE", "simulation run"))
        
        # Write all rows from the first simulation
        for firm in range(0,len(shockloss)):
            writer.writerow((float(shockloss[firm][0]), float(shockloss[firm][1]), float(shockloss[firm][2]), float(shockloss[firm][3]), float(shockloss[firm][4]), float(shockloss[firm][5]), float(shockloss[firm][6]), float(shockloss[firm][7]), float(shockloss[firm][8]), float(shockloss[firm][9]), float(shockloss[firm][10]), float(shockloss[firm][11]), float(shockloss[firm][12]), int(mr) ))
        shoklossset.close()
    else:
        # For subsequent runs, append to the combined file
        shoklossset = open("dataOutputFileComplete.csv", 'a')
        writer = csv.writer(shoklossset, delimiter=',', lineterminator='\n')
        for firm in range(0,len(shockloss)):
            writer.writerow((float(shockloss[firm][0]), float(shockloss[firm][1]), float(shockloss[firm][2]), float(shockloss[firm][3]), float(shockloss[firm][4]), float(shockloss[firm][5]), float(shockloss[firm][6]), float(shockloss[firm][7]), float(shockloss[firm][8]), float(shockloss[firm][9]), float(shockloss[firm][10]), float(shockloss[firm][11]), float(shockloss[firm][12]), int(mr) ))
        shoklossset.close()