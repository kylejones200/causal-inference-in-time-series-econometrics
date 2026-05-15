# Causal Inference in Time Series Econometrics

*Looking at methods to move from correlation to causation using economics*

---

Econometric data is full of correlations that mean nothing. Ice cream sales and drowning rates move together — because both respond to summer. Interest rates and bond prices move together — because one mechanically determines the other. The challenge is not finding relationships in time series. It is finding ones that reflect a real causal mechanism.

Causal inference in econometrics is a toolkit of methods for making that distinction. Each method makes different assumptions about what you know and does not know about the data-generating process.

## Granger Causality: The Foundation

Granger causality asks whether past values of X help forecast Y, over and above Y's own history. It is not philosophical causation — it is predictive precedence — but it is a useful first screen.

```python
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

# Both series must be stationary before testing
data = pd.DataFrame({'gdp_growth': gdp, 'money_supply': m2})

results = grangercausalitytests(data[['gdp_growth', 'money_supply']], maxlag=8)
```

The test compares a restricted VAR (Y's own lags only) to an unrestricted VAR (Y's lags + X's lags). Rejection of the restriction means X has predictive content for Y.

Important limits: Granger causality is silent on confounders. If a third variable drives both X and Y, X will appear to Granger-cause Y spuriously. Always test both directions and interpret results alongside domain knowledge.

## Structural Vector Autoregression (SVAR)

VAR models treat every variable as a function of lagged values of all variables in the system. SVARs add economic structure — restrictions on which contemporaneous relationships are allowed — so that shocks can be given causal interpretations.

```python
from statsmodels.tsa.api import VAR

model = VAR(data)
result = model.fit(maxlags=4, ic='aic')

# Impulse response: how does a shock to money supply propagate to GDP?
irf = result.irf(periods=20)
irf.plot(orth=True)
```

The orthogonalized IRF uses Cholesky decomposition, which assumes a causal ordering: shocks to variables listed first affect all others contemporaneously, but shocks to later variables do not feed back to earlier ones in the same period. Choosing this ordering is a structural assumption — it must be grounded in economic theory, not chosen to get a desired result.

Forecast error variance decomposition (FEVD) answers a complementary question: what fraction of the forecast uncertainty of GDP is attributable to money supply shocks vs. GDP shocks themselves? This is useful for understanding which relationships dominate at short vs. long horizons.

## Local Projections for Causal Analysis

Local projections estimate how an outcome variable responds to a shock over multiple horizons by running a separate regression at each horizon h:

```
Y_{t+h} = α_h + β_h · shock_t + γ_h · controls_t + ε_{t+h}
```

```python
import statsmodels.api as sm
import numpy as np

def local_projections(data, dep_var, shock_var, horizons=16):
    responses, cis = [], []
    for h in range(horizons + 1):
        y = data[dep_var].shift(-h)
        X = sm.add_constant(data[[shock_var]])
        valid = y.notna()
        res = sm.OLS(y[valid], X[valid]).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})
        responses.append(res.params[shock_var])
        cis.append(res.conf_int().loc[shock_var].values)
    return np.array(responses), np.array(cis)
```

The advantage over VAR-based IRFs is robustness: local projections do not impose the VAR's parametric structure on the shape of the response. At long horizons — beyond 8 to 12 periods — local projections are generally preferred. The tradeoff is wider confidence intervals.

## Synthetic Control Method

The synthetic control method handles a specific problem: you want to know the causal effect of an intervention on one unit (a country, state, or firm), but you cannot randomize. You have pre-intervention data for the treated unit and a pool of untreated comparators.

The method finds a weighted combination of control units that best matches the treated unit before the intervention, then uses that synthetic control as the post-intervention counterfactual:

```python
from scipy.optimize import minimize
import numpy as np

def synthetic_control(pre_treated, pre_controls):
    n = pre_controls.shape[1]

    def loss(w):
        return np.mean((pre_treated - pre_controls @ w) ** 2)

    result = minimize(
        loss,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    return result.x
```

The treatment effect at each post-treatment period is the gap between the treated unit and the synthetic control. Inference uses placebo tests: apply the same procedure to each control unit as if it had been treated, and compare the real gap to the distribution of placebo gaps.

The method requires good pre-treatment fit. A synthetic control that poorly matches the treated unit pre-intervention is not a credible counterfactual.

## Difference-in-Differences with Multiple Time Periods

Difference-in-differences (DiD) identifies causal effects by comparing the change over time in a treated group against the change in a control group. The key assumption — parallel trends — says the two groups would have evolved similarly without the treatment.

With multiple treatment periods and staggered adoption (units treated at different times), simple two-way fixed effects DiD is biased. The Callaway-Sant'Anna estimator handles this correctly:

```python
# Using the did package (R) or csdid (Python implementation)
# Group-time average treatment effects
att_gt = did.att_gt(
    yname='outcome',
    tname='time',
    idname='entity_id',
    gname='treatment_year',  # 0 if never treated
    data=df
)
```

The key diagnostic is the pre-treatment event study: plot the treatment effect estimates for periods before the treatment date. They should be near zero. If pre-trends are present, parallel trends fails and the DiD estimate is invalid.

## Instrumental Variables in Time Series

Instrumental variables (IV) address endogeneity: X affects Y, but something also affects X at the same time as Y, creating a spurious correlation. An instrument Z is a variable that:
1. Affects X (relevance)
2. Does not affect Y directly, only through X (exclusion restriction)
3. Is not correlated with the error term (exogeneity)

```python
from linearmodels.iv import IV2SLS

result = IV2SLS(
    dependent=df['gdp_growth'],
    exog=df[['constant', 'controls']],
    endog=df['money_supply'],
    instruments=df[['rainfall', 'commodity_index']]  # example instruments
).fit(cov_type='robust')

print(result.summary)
```

Finding valid instruments is the hardest part. The exclusion restriction — that Z affects Y only through X — cannot be tested. It must be defended on economic grounds. Weak instruments (low first-stage F-statistic, roughly below 10) produce biased IV estimates that can be worse than OLS.

## Modern Approaches to Causal Inference

Machine learning has entered the causal inference toolkit through methods like Double/Debiased Machine Learning (DML), which uses ML models to partial out the effect of high-dimensional controls:

```python
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor

dml = LinearDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    cv=5
)
dml.fit(Y=df['outcome'], T=df['treatment'], X=df[controls], W=df[confounders])

print(dml.effect(df[controls]))
```

DML fits ML models for both the outcome and treatment as functions of controls, takes the residuals, and regresses one set of residuals on the other. This removes the influence of high-dimensional confounders without the bias of standard regression when controls are many relative to sample size.

## Conclusion

Causal inference in time series econometrics is about assembling credible evidence, not just running tests. The statistical machinery — Granger tests, SVARs, local projections, DiD, IV — all produce numbers. The hard work is constructing an argument that those numbers mean something causal.

The key questions for any causal claim:
- What is the identification assumption, and is it plausible?
- What would have to be true for this estimate to be wrong?
- Does the direction of the effect match economic theory?
- Do placebo tests or robustness checks support the main result?

Good causal analysis answers these questions explicitly. Statistical significance alone is not an argument.
