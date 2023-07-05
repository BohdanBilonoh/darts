"""
TFT Explainer for Temporal Fusion Transformer models.
------------------------------------

This module contains the implementation of the TFT Explainer class. The TFTExplainer uses a trained TFT model
and extracts the explainability information from the model.

The .get_variable_selection_weights() method returns the variable selection weights for each of the input features.
This reflects the feature importance for the model. The weights of the encoder and decoder matrix are returned.
An optional plot parameter can be used to plot the variable selection weights.

The .plot_attention_heads() method shows the transformer attention heads learned by the TFT model.
The attention heads reflect the effect of past values of the dependent variable onto the prediction and
what autoregressive pattern the model has learned.

The values of the attention heads can also be extracted using the .get_attention_heads() method.
explain_result = .explain()
res_attention_heads = explain_result.get_explanation(component="attention_heads", horizon=0)

For an examples on how to use the TFT explainer, please have a look at the TFT notebook in the /examples directory
 <https://github.com/unit8co/darts/blob/master/examples/13-TFT-examples.ipynb>`_.

"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.explainability.explainability import (
    ExplainabilityResult,
    ForecastingModelExplainer,
)
from darts.models import TFTModel, TFTOutputsWithInterpretations
from darts.logging import get_logger, raise_if, raise_log
from darts.utils.utils import series2seq

logger = get_logger(__name__)


class TFTExplainer(ForecastingModelExplainer):

    def __init__(
        self,
        model: TFTModel,
        background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        background_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        background_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
    ):
        """
        The base class for forecasting model explainers. It defines the *minimal* behavior that all
        forecasting model explainers support.

        Naming:

        - A background series is a `TimeSeries` with which to 'train' the `Explainer` model.
        - A foreground series is the `TimeSeries` to explain using the fitted `Explainer` model.

        Parameters
        ----------
        model
            A `ForecastingModel` to be explained. It must be fitted first.
        background_series
            A series or list of series to *train* the `ForecastingModelExplainer` along with any foreground series.
            Consider using a reduced well-chosen background to reduce computation time.

                - optional if `model` was fit on a single target series. By default, it is the `series` used
                at fitting time.
                - mandatory if `model` was fit on multiple (sequence of) target series.

        background_past_covariates
            A past covariates series or list of series that the model needs once fitted.
        background_future_covariates
            A future covariates series or list of series that the model needs once fitted.
        """
        if not model._fit_called:
            raise_log(
                ValueError(
                    "The model must be fitted before instantiating a ForecastingModelExplainer."
                ),
                logger,
            )

        if model._is_probabilistic():
            logger.warning(
                "The model is probabilistic, but num_samples=1 will be used for explainability."
            )

        self.model = model

        # if `background_series` was not passed, use `training_series` saved in fitted forecasting model.
        if background_series is None:

            raise_if(
                (background_past_covariates is not None)
                or (background_future_covariates is not None),
                "Supplied background past or future covariates but no background series. Please provide "
                "`background_series`.",
            )

            raise_if(
                self.model.training_series is None,
                "`background_series` must be provided if `model` was fit on multiple time series.",
            )

            background_series = self.model.training_series
            background_past_covariates = self.model.past_covariate_series
            background_future_covariates = self.model.future_covariate_series

        else:
            if self.model.encoders.encoding_available:
                (
                    background_past_covariates,
                    background_future_covariates,
                ) = self.model.generate_fit_encodings(
                    series=background_series,
                    past_covariates=background_past_covariates,
                    future_covariates=background_future_covariates,
                )

        self.background_series = series2seq(background_series)
        self.background_past_covariates = series2seq(background_past_covariates)
        self.background_future_covariates = series2seq(background_future_covariates)

        if self.model.supports_past_covariates:
            raise_if(
                self.model.uses_past_covariates
                and self.background_past_covariates is None,
                "A background past covariates is not provided, but the model needs past covariates.",
            )

        if self.model.supports_future_covariates:
            raise_if(
                self.model.uses_future_covariates
                and self.background_future_covariates is None,
                "A background future covariates is not provided, but the model needs future covariates.",
            )

        self.target_components = self.background_series[0].columns.to_list()
        self.past_covariates_components = None
        if self.background_past_covariates is not None:
            self.past_covariates_components = self.background_past_covariates[
                0
            ].columns.to_list()
        self.future_covariates_components = None
        if self.background_future_covariates is not None:
            self.future_covariates_components = self.background_future_covariates[
                0
            ].columns.to_list()

        self._check_background_covariates(
            self.background_series,
            self.background_past_covariates,
            self.background_future_covariates,
            self.target_components,
            self.past_covariates_components,
            self.future_covariates_components,
        )

        if not self._test_stationarity():
            logger.warning(
                "At least one time series component of the background time series is not stationary."
                " Beware of wrong interpretation with chosen explainability."
            )

    def explain(
        self,
        foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        foreground_past_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        foreground_future_covariates: Optional[
            Union[TimeSeries, Sequence[TimeSeries]]
        ] = None,
        **kwargs
    ) -> Dict[str, List[pd.DataFrame]]:
        """
        Explains a foreground time series, returns an :class:`ExplainabilityResult`.

        Results can be retrieved via the method
        :func:`ExplainabilityResult.get_explanation(horizon, target_component)`.
        The result is a multivariate `TimeSeries` instance containing the 'explanation'
        for the (horizon, target_component) forecast at any timestamp forecastable corresponding to
        the foreground `TimeSeries` input.

        The component name convention of this multivariate `TimeSeries` is:
        ``"{name}_{type_of_cov}_lag_{idx}"``, where:

        - ``{name}`` is the component name from the original foreground series (target, past, or future).
        - ``{type_of_cov}`` is the covariates type. It can take 3 different values:
          ``"target"``, ``"past_cov"`` or ``"future_cov"``.
        - ``{idx}`` is the lag index.

        **Example:**

        Say we have a model with 2 target components named ``"T_0"`` and ``"T_1"``,
        3 past covariates with default component names ``"0"``, ``"1"``, and ``"2"``,
        and one future covariate with default component name ``"0"``.
        Also, ``horizons = [1, 2]``.
        The model is a regression model, with ``lags = 3``, ``lags_past_covariates=[-1, -3]``,
        ``lags_future_covariates = [0]``.

        We provide `foreground_series`, `foreground_past_covariates`, `foreground_future_covariates` each of length 5.


        >>> explain_results = explainer.explain(
        >>>     foreground_series=foreground_series,
        >>>     foreground_past_covariates=foreground_past_covariates,
        >>>     foreground_future_covariates=foreground_future_covariates
        >>> )
        >>> output = explain_results.get_explanation(horizon=1, target="T_1")

        Then the method returns a multivariate TimeSeries containing the *explanations* of
        the corresponding `ForecastingModelExplainer`, with the following component names:

             - T_0_target_lag-1
             - T_0_target_lag-2
             - T_0_target_lag-3
             - T_1_target_lag-1
             - T_1_target_lag-2
             - T_1_target_lag-3
             - 0_past_cov_lag-1
             - 0_past_cov_lag-3
             - 1_past_cov_lag-1
             - 1_past_cov_lag-3
             - 2_past_cov_lag-1
             - 2_past_cov_lag-3
             - 0_fut_cov_lag_0

        This series has length 3, as the model can explain 5-3+1 forecasts
        (timestamp indexes 4, 5, and 6)

        Parameters
        ----------
        foreground_series
            Optionally, the target `TimeSeries` to be explained. Can be multivariate.
            If not provided, the background `TimeSeries` will be explained instead.
        foreground_past_covariates
            Optionally, past covariate timeseries if needed by the ForecastingModel.
        foreground_future_covariates
            Optionally, future covariate timeseries if needed by the ForecastingModel.

         Returns
         -------
         ExplainabilityResult
             The forecast explanations.

        """

        if "n" not in kwargs:
            kwargs["n"] = self.model.output_chunk_length

        if foreground_series is not None:
            kwargs["series"] = foreground_series

        if foreground_past_covariates is not None:
            kwargs["past_covariates"] = foreground_past_covariates

        if foreground_future_covariates is not None:
            kwargs["future_covariates"] = foreground_future_covariates

        outputs: TFTOutputsWithInterpretations = self.model.predict(**kwargs)

        if isinstance(outputs.attention, list):
            attention = [
                attn[
                    0, :, :self.model.input_chunk_length
                ].sum(axis=0)
                for attn in outputs.attention
            ]
        else:
            attention = [
                outputs.attention[
                    0, :, :self.model.input_chunk_length
                ].sum(axis=0)
            ]

        if isinstance(outputs.encoder_variables, list):
            encoder_variables = [
                encoder_variable.mean(axis=0)
                for encoder_variable in outputs.encoder_variables
            ]
        else:
            encoder_variables = [
                outputs.encoder_variables.mean(axis=0)
            ]

        if isinstance(outputs.decoder_variables, list):
            decoder_variables = [
                decoder_variable.mean(axis=0)
                for decoder_variable in outputs.decoder_variables
            ]
        else:
            decoder_variables = [
                outputs.decoder_variables.mean(axis=0)
            ]

        static_variables = outputs.static_variables

        return {
            "attention": attention,
            "encoder_variables": self._get_importance(
                encoder_variables,
                self.model.model.encoder_variables
            ),
            "decoder_variables": self._get_importance(
                decoder_variables,
                self.model.model.decoder_variables
            ),
            "static_variables": self._get_importance(
                static_variables,
                self.model.model.static_variables
            ) if static_variables is not None else None
        }

    def _get_importance(
        self,
        weights: List[np.ndarray],
        names: List[str],
        n_decimals: int = 3,
    ) -> List[pd.DataFrame]:
        """Returns the encoder or decoder variable of the TFT model.

        Parameters
        ----------
        weights
            The weights of the encoder or decoder of the trained TFT model.
        names
            The encoder or decoder names saved in the TFT model class.
        n_decimals
            The number of decimals to round the importance to.

        Returns
        -------
        List of pd.DataFrame
            The importance of the variables.
        """

        # create a dataframe with the variable names and the weights
        name_mapping = self._name_mapping

        importances = []
        for weights_ in weights:
            weights_percentage = weights_.round(n_decimals) * 100
            importance = pd.DataFrame(
                weights_percentage,
                columns=[name_mapping[name] for name in names],
            )
            importances.append(importance)

        return importances

    @property
    def _name_mapping(self) -> Dict[str, str]:
        """Returns the feature name mapping of the TFT model.

        Returns
        -------
        Dict[str, str]
            The feature name mapping. For example
            {
                'past_covariate_0': 'heater',
                'past_covariate_1': 'year',
                'past_covariate_2': 'month',
                'future_covariate_0': 'darts_enc_fc_cyc_month_sin',
                'future_covariate_1': 'darts_enc_fc_cyc_month_cos',
                'target_0': 'ice cream',
             }

        """
        mapping = {}

        if self.background_past_covariates:
            _mapping = {
                f"past_covariate_{i}": colname
                for i, colname in enumerate(self.background_past_covariates[0].components)
            }
            mapping |= _mapping
        if self.background_future_covariates:
            _mapping = {
                f"future_covariate_{i}": colname
                for i, colname in enumerate(self.background_future_covariates[0].components)
            }
            mapping |= _mapping
        if hasattr(self.model, "static_covariates") and self.model.static_covariates is not None:
            _mapping = {
                f"static_covariate_{i}": colname for i, colname in enumerate(self.model.static_covariates.columns)
            }
            mapping |= _mapping
        if self.background_series:
            _mapping = {
                f"target_{i}": colname for i, colname in enumerate(self.background_series[0].components)
            }
            mapping |= _mapping
        return mapping
