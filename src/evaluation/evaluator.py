import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict
import mlflow
from src.training.data_provider import DataProvider


class ModelEvaluator:
    def __init__(self, settings: Dict):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.provider = DataProvider(settings)

        model_name = settings['model']['name']
        model_type = settings['model'].get('type', 'hybrid')
        self.model_path = os.path.join(settings['paths']['model_save'], f"{model_name}_{model_type}_best.keras")
        self.model_type = model_type

        # MLflow Setup
        mlflow_conf = settings.get('mlflow', {})
        self.experiment_name = mlflow_conf.get('experiment_name', 'Default_Experiment')
        self.tracking_uri = mlflow_conf.get('tracking_uri', 'mlruns')
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def run(self):
        self.logger.info("Calculating performance metrics...")

        try:
            _, _, X_test, y_test = self.provider.load_and_split(for_training=True)
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return

        if not os.path.exists(self.model_path):
            self.logger.error(f"Model not found at: {self.model_path}")
            return

        try:
            model = tf.keras.models.load_model(self.model_path)
            preds = model.predict([X_test['input_price'], X_test['input_macro']], verbose=0)
        except Exception as e:
            self.logger.error(f"Error running model: {e}")
            return

        pred_min = preds[0].flatten()
        pred_max = preds[1].flatten()
        pred_avg = (pred_min + pred_max) / 2

        actual_min = y_test['output_min']
        actual_max = y_test['output_max']
        actual_avg = (actual_min + actual_max) / 2

        self._calculate_metrics(actual_avg, pred_avg, pred_min, pred_max)

    def _calculate_metrics(self, actual, predicted, pred_min, pred_max):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

        correct_direction = np.sign(predicted) == np.sign(actual)
        direction_acc = np.mean(correct_direction) * 100

        STOP_LOSS_PCT = 0.02

        signals = np.sign(predicted)
        risk_managed_returns = []

        for i in range(len(actual)):
            # Raw return: Signal * Actual Change
            raw_return = signals[i] * actual[i]

            # Apply Stop Loss Logic
            if raw_return < -STOP_LOSS_PCT:
                risk_managed_returns.append(-STOP_LOSS_PCT)
            else:
                risk_managed_returns.append(raw_return)

        risk_managed_returns = np.array(risk_managed_returns)

        # Equity Curve Calculation
        cumulative_returns = np.cumprod(1 + risk_managed_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Sharpe Ratio (Annualized)
        mean_ret = np.mean(risk_managed_returns)
        std_ret = np.std(risk_managed_returns)

        if std_ret == 0:
            sharpe = 0
        else:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)

        # Win Rate (Trades with positive return)
        win_rate = np.mean(risk_managed_returns > 0) * 100

        # Print Report
        print("\n" + "=" * 50)
        print("MODEL EVALUATION REPORT (RISK MANAGED: SL 2%)")
        print("=" * 50)
        print("REGRESSION METRICS:")
        print(f" - MAE:  {mae:.4f}")
        print(f" - RMSE: {rmse:.4f}")
        print(f" - MAPE: {mape:.2f}%")
        print(f" - R2:   {r2:.4f}")

        print("\nDIRECTION METRICS:")
        print(f" - Accuracy: {direction_acc:.2f}%")

        print("\nFINANCIAL METRICS:")
        print(f" - Win Rate:     {win_rate:.2f}%")
        print(f" - Max Drawdown: {max_drawdown:.2f}%")
        print(f" - Sharpe Ratio: {sharpe:.2f}")
        print("=" * 50 + "\n")

        # Log to MLflow
        with mlflow.start_run(run_name=f"Eval_{self.model_type.upper()}", nested=True):
            mlflow.log_metric("eval_mae", mae)
            mlflow.log_metric("eval_rmse", rmse)
            mlflow.log_metric("eval_mape", mape)
            mlflow.log_metric("eval_r2", r2)
            mlflow.log_metric("eval_accuracy", direction_acc)
            mlflow.log_metric("eval_win_rate", win_rate)
            mlflow.log_metric("eval_max_drawdown", max_drawdown)
            mlflow.log_metric("eval_sharpe", sharpe)