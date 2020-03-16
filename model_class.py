"""
Created: 14 August 2019
Author: Sander van den Oord

Create an ElasticModel or LGBM Model that right away calculates all
the necessary predictions, metrics and plots.

To do:
    - add docstrings to all classes and functions
"""


from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from multiprocessing import cpu_count

import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# good plotting settings
sns.set(style='white', font_scale=1.5)

# some good default params for elasticnet and lightgbm
# tol=0.01 fit models fast, but should probably be set smaller for better models
ELASTIC_PARAMS_DEFAULT = {'tol': 0.01, 'alpha': 0.1, 'l1_ratio': 0.5}
LGBM_PARAMS_DEFAULT = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 15,
    "max_bin": 256,
    "feature_fraction": 0.6,
    "verbosity": 0,
    "drop_rate": 0.1,
    "is_unbalance": False,
    "max_drop": 50,
    "min_child_samples": 10,
    "min_child_weight": 150,
    "min_split_gain": 0,
    "subsample": 0.9,
}

cores_available = cpu_count() - 2
RANDOM_FOREST_PARAMS_DEFAULT = {
    'n_jobs': cores_available,
    'n_estimators': 25,
    'max_depth': 10,
    'max_features': 0.8,
    'random_state': 42,
}

BANDPASS_LOWER_LIMIT = 5.
BANDPASS_UPPER_LIMIT = 95.


class Model(ABC):
    def __init__(
            self,
            X, y,
            train_start, train_end,
            test_start, test_end,
            apply_bandpass=True,
            calculate_cusum=False
    ):

        self.pred_target = y.name

        self.X = X
        self.y_raw = y

        self.train_start = train_start
        self.train_end = train_end

        self.test_start = test_start
        self.test_end = test_end

        self.params = None

        self.apply_bandpass = apply_bandpass
        self.calculate_cusum = calculate_cusum

        self.fitted_model = None
        self.predictions = None
        self.errors = None
        self.metrics_train = None
        self.metrics_test = None

        self.plot_true_vs_predictions = {}
        self.plot_over_time_true_vs_predictions = {}
        self.plot_over_time_error = {}
        self.plot_hist_error = {}

    @property
    @abstractmethod
    def model_type(self):
        pass

    # @abstractmethod makes sure that Model itself can't be instantiated,
    # only a subclass of Model, such as ElasticnetModel or LgbmModel
    @abstractmethod
    def _fit_model(self):
        pass

    def run_model(self):
        self._apply_bandpass_filter(BANDPASS_LOWER_LIMIT, BANDPASS_UPPER_LIMIT)

        self.fitted_model = self._fit_model()

        self.predictions = self._calculate_predictions()

        self.errors = self._calculate_errors()
        self.metrics_train = self._calculate_metrics('train')
        self.metrics_test = self._calculate_metrics('test')

    def create_plots(self):
        self._create_plot_over_time_y_raw()

        periods = {
            'train': self.train_index,
            'test': self.test_index,
            'train_and_test': self.train_and_test_index}

        for period_name, period_index in periods.items():
            self._create_plot_over_time_true_vs_predictions(period_name, period_index)
            self._create_plot_over_time_error(period_name, period_index)
            self._create_plot_true_vs_predictions(period_name, period_index)
            self._create_plot_hist_error(period_name, period_index)

    def _apply_bandpass_filter(self, lower_limit, upper_limit):
        """
        Remove data points from y that have a value above or below a certain
        limit. Also removes those same time points from the features in X, so
        those time points are not used in training.

        :return: X_bandpass, y_bandpass
        """
        self.y = self.y_raw[(self.y_raw >= lower_limit) & (self.y_raw <= upper_limit)].copy()
        self.X = self.X.loc[self.y.index]

        self.train_index = self.y.loc[self.train_start:self.train_end].index
        self.test_index = self.y.loc[self.test_start:self.test_end].index
        self.train_and_test_index = self.y.loc[self.train_start:self.test_end].index

    def _calculate_predictions(self):
        predictions_array = self.fitted_model.predict(self.X)
        predictions_timeseries = pd.Series(
            predictions_array, index=self.y.index, name='predictions')
        return predictions_timeseries

    def _calculate_errors(self):
        errors = (self.y - self.predictions).rename('prediction_errors')
        return errors

    def _calculate_metrics(self, metric_period):

        def calculate_perc_large_error(actuals, predictions, upper_limit=5.):
            """Calculates which percentage of predictions errors can be
            categorized as a large error.
            :return: perc_large_error"""
            large_error_count = ((actuals - predictions).abs() > upper_limit).sum()
            perc_large_error = large_error_count * 1. / actuals.size * 100.
            return perc_large_error

        def run_metrics(actuals, predictions):
            rmse = np.sqrt(mean_squared_error(actuals, predictions)).round(2)
            mae = mean_absolute_error(actuals, predictions).round(2)
            r2 = r2_score(actuals, predictions).round(2)
            perc_large_error = calculate_perc_large_error(actuals, predictions,
                                                          upper_limit=5.).round(2)
            return {'rmse': rmse, 'mae': mae, 'r2': r2,
                    'perc_large_error': perc_large_error}

        if metric_period == 'train':
            return run_metrics(self.y.loc[self.train_index],
                               self.predictions.loc[self.train_index])
        elif metric_period == 'test':
            return run_metrics(self.y.loc[self.test_index],
                               self.predictions.loc[self.test_index])
        else:
            return None

    @staticmethod
    def _make_plot_nice():
        # remove right and top border
        sns.despine()

        # standard function that makes plot look better
        plt.tight_layout()

    def _create_plot_over_time_y_raw(self):
        # a standard setup for my plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))

        # use dots (.) instead of lines to get a better insight because there are so many datapoints
        ax = self.y_raw.plot(linestyle='', marker='.', alpha=1.0, ms=0.5)

        # valves should have values between 0 and 100, to be sure i set those boundaries at -20 and 120
        ax.set(ylim=(-20, 120), title=f'{self.y_raw.name}')

        # add lines where bandpass filter is normally applied
        ax.axhline(y=5., linestyle='--', color='black', lw=0.75)
        ax.axhline(y=95., linestyle='--', color='black', lw=0.75)

        self._make_plot_nice()
        self.plot_over_time_y_raw = fig
        plt.close(fig)

    def _create_plot_true_vs_predictions(self, period_name, period_index):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        if period_name in ['train', 'test']:
            ax.scatter(
                x=self.y.loc[period_index],
                y=self.predictions.loc[period_index],
                s=0.5, alpha=0.4)
        else:
            for index_ in [self.train_index, self.test_index]:
                ax.scatter(
                    x=self.y.loc[index_],
                    y=self.predictions.loc[index_],
                    s=0.5, alpha=0.4,
                )

        ax.plot((0, 100), (0, 100), ls="--")
        ax.set(xlim=(0, 100), ylim=(0, 100),
               title=f'{self.y.name} {self.model_type}\n'
                     f'{period_name}: predictions vs true')

        self._make_plot_nice()
        self.plot_true_vs_predictions[period_name] = fig
        plt.close(fig)

    def _create_plot_over_time_true_vs_predictions(self, period_name, period_index):
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))

        plot_settings = {'linestyle': '', 'marker': '.', 'alpha': 0.5, 'ms': 0.3}

        ax.plot(self.y.loc[period_index],
                label='true', **plot_settings)
        ax.plot(self.predictions.loc[period_index],
                label='predictions', **plot_settings)

        if period_name == 'train_and_test':
            ax.axvline(self.test_start, linestyle='--', c='black')

        title = f'{self.y.name}\n {self.model_type}'
        if self.model_type == 'ELASTIC':
            for key, value in self.params.items():
                title += f' {key}:{value}'
        title += '\n'

        if period_name == 'train':
            title += f'{period_name.upper()} metrics: '
            for key, value in self.metrics_train.items():
                title += f' {key}:{value}'
        else:
            title += 'TEST metrics: '
            for key, value in self.metrics_test.items():
                title += f' {key}:{value}'

        ax.set(ylim=(0, 100), title=title)

        ax.legend(markerscale=50, loc='upper right')

        self._make_plot_nice()
        self.plot_over_time_true_vs_predictions[period_name] = fig
        plt.close(fig)

    def _create_plot_over_time_error(self, period_name, period_index):
        fig, ax = plt.subplots(1, 1, figsize=(15, 7))

        ax.plot(self.errors.loc[period_index], linestyle='', marker='.', alpha=1.0, ms=1.0)
        ax.set(
            title=f'{self.y.name} {self.model_type} Error '
                  f'{period_name.upper()} period: true minus predictions',
            ylim=(-10, 10))
        ax.axhline(y=0., linestyle='--')
        ax.axhline(y=5., linestyle='--', color='red', lw=0.6)
        ax.axhline(y=-5., linestyle='--', color='red', lw=0.6)

        if period_name == 'train_and_test':
            ax.axvline(self.test_start, linestyle='--', c='black')

        self._make_plot_nice()
        self.plot_over_time_error[period_name] = fig
        plt.close(fig)

    def _create_plot_hist_error(self, period_name, period_index):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        if period_name in ['train', 'test']:
            ax.hist(self.errors.loc[period_index], bins=100)
        else:
            for index_ in [self.train_index, self.test_index]:
                ax.hist(self.errors.loc[index_], bins=100)

        ax.set(xlim=(-6, 6),
               title=f'{self.y.name} {self.model_type} '
                     f'Distribution of errors on {period_name}')
        ax.axvline(x=0, linestyle='--', color='black')
        ax.axvline(x=-5, linestyle='-.', color='red')
        ax.axvline(x=5, linestyle='-.', color='red')

        self._make_plot_nice()
        self.plot_hist_error[period_name] = fig
        plt.close(fig)


class ElasticnetModel(Model):
    model_type = 'ELASTIC'

    def __init__(self, X, y, train_start, train_end, test_start, test_end, params=None):
        super().__init__(X, y, train_start, train_end, test_start, test_end)
        if params:
            self.params = params
        else:
            self.params = ELASTIC_PARAMS_DEFAULT.copy()

    def _fit_model(self):
        elasticnet_pipeline = self._get_elasticnet_pipeline()

        elasticnet_pipeline.fit(
            self.X.loc[self.train_index],
            self.y.loc[self.train_index])

        return elasticnet_pipeline

    def _get_elasticnet_pipeline(self):
        """Create a pipeline with standard_scaler and ElasticNet model.
        :return: sklearn pipeline"""
        standard_scaler = StandardScaler()
        elasticnet = ElasticNet(**self.params)
        elasticnet_pipeline = make_pipeline(standard_scaler, elasticnet)
        return elasticnet_pipeline

    def run_model(self):
        super().run_model()
        self.elastic_coefficients = self._get_elastic_coefficients()

    def create_plots(self):
        super().create_plots()
        self._create_plot_elastic_coefficients()

    def _get_elastic_coefficients(self):
        model_coef = pd.DataFrame(
            self.fitted_model.named_steps['elasticnet'].coef_,
            columns=['coef'],
            index=self.X.columns
        )

        model_coef_nonzero = model_coef[model_coef['coef'] != 0.0].copy()

        model_coef_nonzero['coef_abs'] = model_coef_nonzero['coef'].abs()
        model_coef_nonzero = model_coef_nonzero.sort_values(by='coef_abs', ascending=False)

        return model_coef_nonzero['coef'].rename('elastic_coefficients')

    def _create_plot_elastic_coefficients(self):
        model_coef = self.elastic_coefficients

        fig, ax = plt.subplots(1, 1, figsize=(15, model_coef.shape[0] * 0.4))
        model_coef[::-1].plot(kind='barh', ax=ax)

        title = f'{self.y.name} {self.model_type} Non-zero coefficients'
        for key, value in self.params.items():
            title += f' {key}:{value}'

        ax.set(title=title)

        self._make_plot_nice()
        self.plot_elastic_coefficients = fig
        plt.close(fig)


class LgbmModel(Model):
    model_type = 'LGBM'

    def __init__(self, X, y, train_start, train_end, test_start, test_end, params=None):
        super().__init__(X, y, train_start, train_end, test_start, test_end)
        if params:
            self.params = params
        else:
            self.params = LGBM_PARAMS_DEFAULT.copy()

    def _fit_model(self):
        lgb_train = lgb.Dataset(
            self.X.loc[self.train_index],
            self.y.loc[self.train_index])

        lgbm_fitted = lgb.train(
            self.params,
            lgb_train)

        return lgbm_fitted

    def run_model(self):
        super().run_model()
        self._calculate_feature_importances()

    def create_plots(self):
        super().create_plots()
        self._create_plot_feature_importance()

    def _calculate_feature_importances(self):
        df_feat_imp = pd.DataFrame({
            'gain': self.fitted_model.feature_importance(importance_type='gain'),
            'split': self.fitted_model.feature_importance(importance_type='split')
        }, index=self.X.columns)

        self.feature_importance_gain = df_feat_imp['gain'].sort_values(ascending=False)
        self.feature_importance_split = df_feat_imp['split'].sort_values(ascending=False)
        self.df_feat_imp = df_feat_imp

    def _create_plot_feature_importance(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 12))

        for nr, ax in enumerate(axes):
            plt.sca(ax)
            series_top = self.df_feat_imp.iloc[:, nr].sort_values(
                ascending=False).iloc[:20][::-1]
            ax = series_top.plot(kind='barh', )
            ax.set(
                title=f'{self.y.name} {self.model_type}\n'
                      f'lgbm feature importance by {series_top.name.upper()}')

        self._make_plot_nice()
        self.plot_feature_importance = fig
        plt.close(fig)


class RandomForestModel(Model):
    model_type = 'RANDOM_FOREST'

    def __init__(self, X, y, train_start, train_end, test_start, test_end, params=None):
        super().__init__(X, y, train_start, train_end, test_start, test_end)
        if params:
            self.params = params
        else:
            self.params = RANDOM_FOREST_PARAMS_DEFAULT.copy()

    def _fit_model(self):
        random_forest = RandomForestRegressor(**self.params)

        random_forest.fit(
            self.X.loc[self.train_index],
            self.y.loc[self.train_index])

        return random_forest

    def run_model(self):
        super().run_model()
        self._calculate_feature_importances()

    def _calculate_feature_importances(self):
        self.feature_importance = pd.Series(
            self.fitted_model.feature_importances_,
            index=self.X.columns,
        ).sort_values(ascending=False)
