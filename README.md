# new-phone-who-dis
Speaker diarization on NPR Podcast data using Hidden Markov Models and Gaussian Mixture Models


https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts

## Data cleaning and descriptive statistics checkpoint:

For this assignment, you need to report your initial data exploration. Consider some of the following points as examples. Upload a Notebook with comments describing what you discovered. The emphasis here is not on perfect formatting, but on showing that you have done your due diligence in exploring the data and thinking through the consequences of what you see.

Before doing any exploration, consider when and how you plan on holding out data for model evaluation. The gold standard is to have totally independent data you have never seen before tucked away so you can evaluate model performance at the end, but there can be many reasons this may not work. Explain your choice. If you do want to hold out the same data every time, consider fixing a random seed when you make the test-train split.
Beware that a test set for time-series data is trickier to build than for independent data points.  If you have training data just before and just after the test data point, the correlation between them means the test depends a lot on the training data and hence is a bad test.  At least ensure your test set comes after your training data, and ideally in a separate data-collection session. 
Print out a few dozen rows of the data. Is there anything you didn't expect to see? What opportunities for data cleaning and feature engineering may be important? Take care of these things.
Plot a few individual time series and do a similar check. Is there anything unbelievable you see?
How much data is missing? Is the distribution of missing data likely different from the distribution of non-missing data? How might you do a meaningful imputation (if needed)? Are there variables that should be dropped? Implement some initial solutions.
Is there any hint that the data you have collected is differently distributed from the actual application of interest? If so, is there a strategy, such as reweighing samples, that might help?
Use a histogram or KDE to visualize the distribution of key variables. Consider log-scaling or other scaling of the axes. How should you think about outliers? Is there a natural scaling for certain variables?
Use 2D and/or 3D plot scatter plots, histograms, or heat maps to look for important relationships between variables. Consider using significance tests, linear model fits, or correlation matrices to clarify relationships.
Does what you see change any of your ideas for what models might be appropriate? Among other things, if your models rely on specific assumptions, is there a way you can check if these assumptions actually hold by looking at the data? If you are using linear models, do the relevant plots look linear? Is there some other scaling where the model assumptions might more nearly hold?
