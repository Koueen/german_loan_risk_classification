The dataset corresponds to Credit German Bank, which includes 20 features +1 as the target (client
risk), in other words, if the client is going to pay off the loan at the end. Due to the high number
of features I started aiming the problem at looking for relevant features. For that, I used Sequential
Backward Selection considering accuracy metric. The process started with all 20 features and it
ended up with 12.

The next section is an initial exploratory analysis, in which I tried to represent graphically the most
impacting features to ’risk’ (our target). In order to explore them, I ploted a correlation heatmap,
where features like ’ACC status’ and ’duration in months’ were high correlated to ’risk’. Once I
selected the final features, I started to train Random Forest model. There are other ways to find best
models, i.e., using FLML Auto ML, which is a library in python that automates the search of the
best model. But for classification problems, Random Forest and XGBoost are normally the best
ones, therefore at the end of this section, they will be compared.
The training is based on grid search with a list of specific parameter configurations. I tried to avoid
overfitting the model, by specifying low estimator numbers (trees). The grid search is composed by
several scorers intended to be compared to the next trained model in the validation data. The cross-
validation with cv = 5 (200 samples in validation) is generated automatically. The chosen refit
metric is ’recall’. Then, once the best configuration is selected by this primary metric, is refitted
with the entire dataset. Why I selected recall metric? In my dataset, 1 denotes bad client (no pay
back) and 0 denotes good client being natural to risk term. Improving recall will mean, reducing
the number of times a bad candidate (1) is classified as good candidate (0), which is more critical
than classifying a good candidate as bad.

After getting the Random Forest trained and evaluated, I trained a XGBoost applying the same
method, a grid search and same metrics for evaluation. Once I get all metrics, I compared them in
order to pick the best one as our risk classifier. Despite the fact that Random Forest is performing
better than XGBoost, I selected XGBoost as my classifier because it seems that it can generalize
better in evaluation data. Observing recall, XGBoost performs 6% greater than Random Forest.
Random Forest training performs incredibly better than the other, but this is a sign of overfitting.