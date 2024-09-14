## Modeling

We used multiple models for comparison:
- **Random Forest Regressor**: Main model due to its strength in handling both categorical and continuous features.
- **Linear Regression**: Simple baseline model for comparison.

### Random Forest Regressor

This model works by building multiple decision trees during training and averaging their predictions to produce the final output. The main parameters tuned include:
- **n_estimators**: Number of trees.
- **max_depth**: Maximum depth of trees.
- **min_samples_split**: Minimum number of samples required to split an internal node.
- **min_samples_leaf**: Minimum number of samples required to be at a leaf node.
- **max_features**: Maximum number of features to consider for splits.

## Model Tuning

We used **RandomizedSearchCV** for hyperparameter tuning. The best hyperparameters found were:
- **n_estimators**: 500
- **max_depth**: 10
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **max_features**: log2

## Evaluation

The final model was evaluated using:
- **Mean Squared Error (MSE)**: 0.5396
- **R-squared**: 0.6272

## Conclusion

The Random Forest Regressor performed well in predicting house prices, explaining approximately 62.72% of the variability in the data. This model successfully identified key factors influencing house prices, such as the number of bedrooms, air conditioning, and property area. Future improvements could involve experimenting with more advanced models or incorporating additional data to enhance the model’s accuracy.

## Future Work

- Experiment with other machine learning models such as **Gradient Boosting** or **XGBoost**.
- Explore feature engineering techniques to improve model performance.
- Incorporate additional datasets to improve the robustness of the model.
