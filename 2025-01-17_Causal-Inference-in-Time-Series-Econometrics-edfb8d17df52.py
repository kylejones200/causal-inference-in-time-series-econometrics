# Description: Short example for Causal Inference in Time Series Econometrics.




from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import statsmodels.api as sm

class GrangerAnalysis:
    def __init__(self, data):
        self.data = data
    def test_granger_causality(self, variable1, variable2, max_lags=12):
        """Test for Granger causality between two variables"""
        data = pd.concat([self.data[variable1], self.data[variable2]], axis=1)
        results = grangercausalitytests(data, maxlag=max_lags)
        causality_results = pd.DataFrame(
            index=range(1, max_lags + 1),
            columns=['F-statistic', 'p-value']
        )
        for lag in range(1, max_lags + 1):
            causality_results.loc[lag] = [
                results[lag][0]['ssr_ftest'][0],
                results[lag][0]['ssr_ftest'][1]
            ]
        return causality_results
    def plot_causality_results(self, results):
        """Plot p-values for different lag orders"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(results.index, results['p-value'], marker='o')
        plt.axhline(y=0.05, color='r', linestyle='--', label='5% significance')
        plt.xlabel('Lag Order')
        plt.ylabel('p-value')
        plt.title('Granger Causality Test Results')
        plt.legend()
        plt.show()


class SVARModel:
    def __init__(self, data):
        self.data = data
        self.var_model = None
        self.svar_results = None
    def fit(self, lags=1, A=None, B=None):
        """Fit SVAR model with short-run (A) and long-run (B) restrictions"""
        self.var_model = VAR(self.data)
        var_results = self.var_model.fit(lags)
        if A is None:
            A = np.eye(len(self.data.columns))
        if B is None:
            B = np.eye(len(self.data.columns))
        self.svar_results = var_results.svar(A=A, B=B)
        return self.svar_results
    def impulse_response(self, periods=20):
        """Calculate impulse response functions"""
        return self.svar_results.irf(periods=periods)
    def forecast_error_variance_decomposition(self, periods=20):
        """Compute forecast error variance decomposition"""
        return self.svar_results.fevd(periods=periods)


class LocalProjections:
    def __init__(self, data):
        self.data = data
    def estimate_impulse_response(self, dependent_var, shock_var, controls=None, horizons=20):
        """Estimate impulse responses using local projections"""
        responses = []
        confidence_intervals = []
        for h in range(horizons + 1):
            y = self.data[dependent_var].shift(-h)
            X = self.data[[shock_var]]
            if controls is not None:
                X = pd.concat([X, self.data[controls]], axis=1)
            X = sm.add_constant(X)
            valid_idx = y.notna()
            y = y[valid_idx]
            X = X[valid_idx]
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': h})
            responses.append(results.params[shock_var])
            confidence_intervals.append(results.conf_int().loc[shock_var])
        return np.array(responses), np.array(confidence_intervals)


class SyntheticControl:
    def __init__(self, data, treatment_unit, control_units, treatment_period, outcome_var):
        self.data = data
        self.treatment_unit = treatment_unit
        self.control_units = control_units
        self.treatment_period = treatment_period
        self.outcome_var = outcome_var
    def construct_synthetic_control(self):
        """Construct synthetic control unit"""
        pre_treatment = self.data[self.data.index < self.treatment_period]
        def objective(weights):
            synthetic = np.sum([
                w * pre_treatment[self.outcome_var][pre_treatment.unit == u]
                for w, u in zip(weights, self.control_units)
            ], axis=0)
            treated = pre_treatment[self.outcome_var][
                pre_treatment.unit == self.treatment_unit
            ]
            return np.mean((treated - synthetic) ** 2)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        result = minimize(
            objective,
            x0=np.ones(len(self.control_units)) / len(self.control_units),
            constraints=constraints
        )
        return result.x


class DynamicDiD:
    def __init__(self, data):
        self.data = data
    def estimate_dynamic_effects(self, outcome_var, treatment_var, unit_fe=True, time_fe=True):
        """Estimate dynamic treatment effects"""
        leads_lags = range(-4, 5)
        for t in leads_lags:
            self.data[f'treat_t{t}'] = self.data[treatment_var].shift(-t)
        formula = f"{outcome_var} ~ " + " + ".join([f"treat_t{t}" for t in leads_lags])
        if unit_fe:
            formula += " + EntityEffects"
        if time_fe:
            formula += " + TimeEffects"
        model = sm.PanelOLS.from_formula(formula, data=self.data)
        return model.fit(cov_type='clustered', cluster_entity=True)

class TSInstrumentalVariables:
    def __init__(self, data):
        self.data = data

def estimate_iv(self, dependent_var, endogenous_var, instrument_var, controls=None):
        """Estimate instrumental variables regression"""
        X_first = sm.add_constant(self.data[instrument_var])
        if controls is not None:
            X_first = pd.concat([X_first, self.data[controls]], axis=1)
        first_stage = sm.OLS(self.data[endogenous_var], X_first).fit()
        fitted_values = first_stage.predict()
        X_second = sm.add_constant(fitted_values)
        if controls is not None:
            X_second = pd.concat([X_second, self.data[controls]], axis=1)
        second_stage = sm.OLS(self.data[dependent_var], X_second).fit()
        return first_stage, second_stage


class ModernCausalInference:
    def __init__(self, data):
        self.data = data
    def double_machine_learning(self, y, d, x, ml_model=None):
        """Implement double machine learning for causal inference"""
        if ml_model is None:
            ml_model = RandomForestRegressor(n_estimators=100)
        ml_model.fit(self.data[x], self.data[d])
        d_hat = ml_model.predict(self.data[x])
        ml_model.fit(self.data[x], self.data[y])
        y_hat = ml_model.predict(self.data[x])
        treatment_effect = sm.OLS(
            self.data[y] - y_hat,
            self.data[d] - d_hat
        ).fit()
        return treatment_effect
