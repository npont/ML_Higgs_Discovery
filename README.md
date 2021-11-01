# Machine Learning Project 1

In 2013, the Higgs boson elementary particle was discovered at CERN. In this project, our goal is to recreate the discovery process, 
by applying machine learning techniques to actual CERN particle accelerator data. 
Data regroup background signals (noise) and actual decay signals (representing collision processes). To determine if an event resulted in background noise or 
in real Higgs boson, we need a robust model, that we try to create here.

In this repository, there is a pre-processing file used on the raw data, a computation file that contains all the functions needed to implement our models, 
a cross-validation file to find best parameters for our models, a plot file to visualize the results of the cross-validation and finally a run file to call 
our models and create predictions. 
