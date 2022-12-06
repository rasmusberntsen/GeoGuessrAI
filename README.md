# GeoGuessrAI

### The project contains the following files:
- countries.csv contains path to the image and the country name for each image
- csvgen.py generated the csv file from the images in the images folder
- dataloader.py is the dataloader used to load the images and the country names
- heatmap.py contains the function used to output the heatmap shown in some (or all) of the reports
- ImageGetter.py contains the functions that scrapes the images from randomstreetview.com
- GeoNet.ipynb is the notebook used to load in the model and look at their performance on specific images from the test set
- RunGrid.py is the script used to run the grid search on all 32 combinations of hyperparameters and return a model for each
- RunModels.py is the script used to run the models on the test set and output the results print the performance of the models on the testset
- TrainingLoop.py is the script used to train the models and evaluate them on the validation set
