# def evaluate_predictions(predictions, actuals):
#     """
#     Evaluate the accuracy of stock predictions.

#     Parameters:
#     predictions (list of float): The predicted stock prices.
#     actuals (list of float): The actual stock prices.

#     Returns:
#     dict: A dictionary containing evaluation metrics.
#     """
#     if len(predictions) != len(actuals):
#         raise ValueError("The length of predictions and actuals must be the same.")

#     # Calculate Mean Absolute Error (MAE)
#     mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(predictions)

#     # Calculate Mean Squared Error (MSE)
#     mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)

#     # Calculate Root Mean Squared Error (RMSE)
#     rmse = mse ** 0.5

#     return {
#         "MAE": mae,
#         "MSE": mse,
#         "RMSE": rmse
#     }