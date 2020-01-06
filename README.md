# 2018_ds_bowl

1. Data pre-processing : import the images and masks. [1]
2. Downsample both the training and test set.
3. Create dice metric [2] 
4. Build U-Net model [3]
5. Fit the data
6. Save the model
7. Load the model to make predictions
8. Encode the results as requested in competition requirements with run-length encoding [4]

Reference-links:
[1] https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
[2] https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
[3] https://github.com/jocicmarko/ultrasound-nerve-segmentation
[4] https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
