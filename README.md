# Assessing IoT Security with a Random Forest Predictor

I. EXECUTIVE SUMMARY  
The security of an IoT platform is dependent on the security of the individual layers in the IoT system. There are generally 3 layers in the architecture. First is the perception layer which consists of all IoT devices. Then comes the network layer which is responsible for transport of all the data and last the application layer which is responsible for the access of all the data stored in the databases through cloud or the other medium. Each layer has its own problems and a different variety of attacks. For perception layer we have addressed problems related to security of the device itself and assigned a score based on it. As for the network layer, we consider the IDS (Intrusion Detection System) efficiency and the network structure. Application layer security considers the different code injection and password security.  

II. REQUIREMENTS  
In this project, the instructor requires us to explore machine learning techniques, to approximate the input-output behavior of our Expert System (ES). The goal is to create a predictive model. We need to use our existing Expert Systems (ES) model to generate a dataset.  

III. IMPLEMENTATION  
This project utilizes a Random Forest Regressor to approximate the input-output surface of an Expert System (ES) by training on a dataset of security scores.   

A) Dataset Generation:
The dataset contains an almost equal distribution of all the scores on a scale of 0-10. This was done by generating all the combinations, of the possible inputs, however, this is not enough. All combinations produce a distribution of scores which resembles a bell curve, so we have additionally generated samples to even out this distrbution. This was neccesary for the model to produce the correct output for all scores or it would be undertrained for inputs that produce a score that was not close to 5. The data set produced has 10000 rows.   

B) Data Preprocessing:
The code first reads the csv file into a dataframe using
‘pandas’. It checks for null values and encodes categorical variables into numerical values using LabelEncoder. The columns ‘open_port_var’ and ‘password_var’ are dropped as they are not needed for the analysis. Features (X) and the target variable (y) are separated, followed by splitting the data into training and testing sets into a 70%-30% split respectively.  

C) Model Training:
The code makes use of the ‘RandomForestRegressor’ from the sklearn library. A ‘RandomForestRegressor’ is a learning method that works by constructing many decision trees during training. The output generated is the average of the predictions from the individual decision trees to improve prediction and control over-fitting.    

![image](https://github.com/akhilshetty97/IOTsecurityevaluation/assets/47709446/f2ff4995-095d-405b-8368-929d95c0913b)  
Figure Source: https://levelup.gitconnected.com/random-forest- regression-209c0f354c84  
The hyperparameters are tuned using ‘GridSearchCV’. The ‘RandomForestRegressor’ is then trained with the best parameters found with the ‘GridSearchCV’.  

D) User Input:
The user is prompted to enter values for several security- related attributes. These inputs are then converted to the appropriate numerical format using the previously fitted label encoders. A helper function ‘handle_unseen_labels’ ensures that any unseen labels in the user input are handled appropriately.  

E) Prediction and Visualization:
The trained model makes predictions based on the user's input. Feature importance is visualized using a bar plot, and the model's performance is depicted with a scatter plot comparing actual vs. predicted values. 

IV. EXPERIMENTS  
Initial Model Training:  
The dataset was similarly split into test and train data in the 70%-30% ratio using ‘train_test_split’ from ‘sklearn’. An initial Random Forest Regressor model was trained with default parameters to establish a baseline performance. An R2 score of 0.85 was achieved using this base model.
Hyperparameter Tuning:  
A parameter grid was defined to explore various hyperparameters for the Random Forest Regressor, ‘n_estimators’, ‘max_depth’, ‘min_samples_split’, ‘min_samples_leaf’, and ‘max_features’. ‘GridSearchCV’ was used to perform an exhaustive search over the defined parameter grid with cross-validation. The best hyperparameters obtained from Grid Search were used to configure a new Random Forest Regressor
Final Model Training:  
A Random Forest Regressor was trained with the best parameters found using Grid Search. These parameters are:  
a) max_depth: 20  
b) max_features: ‘sqrt’  
c) min_samples_leaf : 1  
d) min_samples_split: 2   
e) n_estimators: 300  
This model was then evaluated with the test set. The achieved an R2 score of 0.97 which is much higher than the base model which was trained.  

V. RESULTS  
The Random Forest Regressor was trained and evaluated on the dataset. Below are the key performance metrics obtained from the test set.  
a) R2 Score: 0.97  
b) Mean Absolute Error (MAE): 0.298  
c) Mean Squared Error (MSE): 0.152  
These metrics indicate a high level of accuracy for the model. An R2 score of 0.97 suggests that the model explains 97% of the variance in the security scores. The low values for MAE and MSE indicate that the predictions are close to the actual values. An MAE of 0.298 indicates that, on average, the model's predictions deviate from the actual security scores by 0.298 points. The MSE of 0.152 squares the differences between predicted and actual values and then averages them.  

Visualization:  
a) Feature Importance:  
To understand which features are most influential in predicting the security score, feature importances have been derived from the Random Forest Regressor.  
![image](https://github.com/akhilshetty97/IOTsecurityevaluation/assets/47709446/8c921383-6157-433d-9d53-65a027882ba9)  
This bar plot shows the importance of each feature in the model. Features are listed on the x-axis, and their importance scores are on the y-axis. The features are sorted in descending order of importance. From the plot, we observe that:  
ids_presence is most important feature, indicating its significant impact on the security score.  
secure_boot_var and interaction_data_var also have high importance, highlighting their critical roles in determining security performance.  
Features like input_data_var, and protocol_var have relatively lower importance but still contribute to the model.  

b) Predicted vs Actual Plot:  
To visualize the model's performance, we plotted the predicted security scores against the actual scores from the test set.  

![image](https://github.com/akhilshetty97/IOTsecurityevaluation/assets/47709446/79559aaa-dd18-4e57-bb72-e34c4aad995c)  
This scatter plot compares the predicted security scores (y- axis) against the actual scores (x-axis). The red line represents the ideal scenario where predicted values perfectly match the actual values.  

Blue dots: Each dot represents a prediction. The closer a dot is to the red line, the more accurate the prediction.  

The clustering of dots around the red line indicates that the model's predictions are generally accurate. Some dots deviate from the line, suggesting instances where the model's predictions are less accurate, but these are relatively few.  

c) User Input Prediction:  
The model also predicts the score based on a user input. The script encodes the user inputs using the label encoders.  
Input Example:  
Enter OS (e.g., Linux): Linux  
Enter OS version (e.g., 22.6): 22.6  
Is Secure Boot Enabled? (Yes/No): Yes    
Enter Protocol (e.g., HTTP): HTTP  
Is IDS Present? (Yes/No): Yes  
Enter Detection Rate (0-100): 100  
Enter False Negative Rate (0-100): 10  
Enter Response Time (0-100): 5  
Is Segmentation Present? (Yes/No): yes  
Enter Input Data Type: Normal input data Enter Interaction Data Type: Normal input data Enter RFID Data Type: Normal input data  

The predicted es_score is: 9.35  




