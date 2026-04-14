
Replication Python code for:

Dynamic Order Dispersion and Volatility Persistence in a Simple Limit Order Book model

By:
Andrew Hawley; Blake LeBaron; Mark Paddrik; Nathan Palmer


This set of Python code will replicate most of the key figures in the paper with the simulated model.

It is built with two goals in mind:

1) The user can simply run the main.py program, and that should (given similar random number generators), 
give the exact replication for the figures in the paper.  They will automatically be placed in the figures 
subdirectory.
2) The code is set up in such a way that users should be able to modify most of the crucial parameters in the 
paper to do their own robustness checks of the model.


To run the code:
1) Download the 3 files, main.py, LOBModel.py, and make_plots4.py into a diretory.
2) Run the main.py program using whatever way you run Python on your machine.

Requirements:

Code requires the following standard Python libraries:

1) numpy
2) pandas
3) scipy
4) matplotlib
5) statsmodels
6) random

These libraries are all part of the standard Anaconda Python distribution which was
the base for our code.

We have tested all this code in both Win11, and Ubuntu Linux (using WSL on Windows).
Both used Python version 3.11.7.


Two omissions:
1.) The code does not generate some of the early robustness check figures (2 and 3).
2.) Also all figures involving actual financial data are removed due to licensing issues
with our data sets.  They are all publically available for a fee, and the vendors are as follows:

a.) Yahoo Finance (daily SPY)
b.) Tickdata (high frequency SPY))

