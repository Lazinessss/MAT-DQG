# MAT-DQG

**Accuracy Algorithm：**

1. General program:
       (1) ModelSelection.py：A variety of machine-learned regression models for modeling and prediction;
       (2) Printing.py：Creating folders and workbooks, and write 1D or 2D data to tables;
2. Model prediction:
       (1) Splitting.py：Dataset partition;
       (2) Training.py：The prediction of various machine learning models on specific dataset.
3. Outlier detection:
       (1) SingleDimensionalOutlierDetection.py：Single-Dimensional Outlier Detection (Quartile Method);
       (2) BoxPlotting.py：Plotting Box Plots (Schematic Diagram of Quartile Results);
       (3) MultiDimensionalOutlierDetection.py：Multidimensional outlier sample detection;
       (4) Clustering.py：Scatter plots are plotted based on clustering of all data, target performance data, and all feature data.

**Redundancy Algorithm**

1. General program:
   (1) ModelsCV.py：A variety of machine-learned regression models for modeling and prediction;
   (2) Printting.py：Creating folders and workbooks, and write 1D or 2D data to tables;
2. Feature selection and machine learning model prediction:
   (1) FeatureSelection/MBPSO.py：The main algorithm of the NCOR-FS method (including population initialization and evolution);
   (2) FeatureSelection/NondominatedSolutionCal.py：Non-dominant solution recognition;
   (3) FeatureSelection/ViolationDegreeCal.py：Calculation of NCOR Violation;
   (4) FeatureSelection/DncorCal：Highly correlated feature recognition.
