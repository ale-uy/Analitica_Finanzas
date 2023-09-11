from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.serialize import model_to_json, model_from_json


class Profeta:

    @classmethod
    def cargar_modelo_prophet(cls):
        with open('prophet_model.json', 'r') as fin:
            model = model_from_json(fin.read())  # Load model
        return model

    @classmethod
    def entrenar_modelo(cls, 
            df: pd.DataFrame, 
            target: str, 
            dates: str, 
            horizon: str = '30 days',
            grid: bool = False, 
            parallel = None,
            save_model: bool = False):
        """
        Entrena y ajusta un modelo Prophet para pronóstico de series temporales.

        Parámetros:
            df (pd.DataFrame): El DataFrame que contiene los datos de la serie temporal.
            target (str): El nombre de la columna que contiene los valores objetivo.
            dates (str): El nombre de la columna que contiene las fechas correspondientes.
            horizon (str): La ventana de tiempo para la predicción futura. Por defecto es '30 days'.
            grid (bool): Indica si se debe realizar una búsqueda de cuadrícula de hiperparámetros.
            parallel: Opciones de paralelización para cross_validation. Opciones: 'processes', 'threads'.
            save_model (bool): Indica si se debe guardar el modelo ajustado en formato JSON.

        Retorna:
            Prophet: El modelo Prophet ajustado.

        Ejemplo:
            # Crear un DataFrame de ejemplo
            data = {
                'fecha': pd.date_range(start='2023-01-01', periods=50, freq='D'),
                'valor': range(50)
            }
            df = pd.DataFrame(data)

            # Crear una instancia de la clase DL y entrenar el modelo Prophet
            best_model = Profeta.entrenar_modelo(df, 'valor', 'fecha', grid=False, save_model=False)

            # Hacer predicciones con el modelo
            future = best_model.make_future_dataframe(periods=10)
            forecast = best_model.predict(future)

            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

        """
        # Preparar los datos en el formato requerido por Prophet
        df_prophet = df.rename(columns={target: 'y', dates: 'ds'})

        # Definir la cuadrícula de parámetros para la búsqueda
        if grid:
            param_grid = {
                # 'changepoint_prior_scale': [0.01, 0.05, 0.2],
                # 'seasonality_prior_scale': [1.0, 10]
                'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                #'holidays_prior_scale': [0.01, 10],
                #'seasonality_mode': ['additive', 'multiplicative'],
                #'changepoint_range': [0.8, 0.95]
            }
        else:
            param_grid = {
                'changepoint_prior_scale': [0.05],
                'seasonality_prior_scale': [10]
            }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the MSEs for each params here

        # Use cross validation to evaluate all parameters
        best_model = None
        for params in all_params:
            model = Prophet(**params).fit(df_prophet)  # Fit model with given params
            df_cv = cross_validation(model, horizon=horizon, parallel=parallel)
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0]) # type: ignore
            if df_p['rmse'].values[0] <= min(rmses): # type: ignore
                best_model = model

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results.sort_values(by=['rmse']))

        if save_model:
            with open('prophet_model.json', 'w') as fout:
                fout.write(model_to_json(best_model))  # Save model

        return best_model
