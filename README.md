<h1 align="center">FPL-Broward-College-ML-Data-Science-Project</h1>
<h2 align="center">A Comparative Analysis of ARIMA(X) Models and AR-X GARCH Models Used to Predict Historical Energy Data in South Florida</h2>
<h3 align="center">Research Statement</h3> 

 

<p align="center">Broward College and Florida Power & Light (FPL) company have invested time and resources into researching time-series models to analyze FPL's influence on the Consumer Price Index for all urban consumers of energy services in South Florida. Insights gained from this exploration can lead to more informed decisions regarding energy pricing and the CPI’s response to price changes.</p> 

<p align="center">The data used for this research was sourced from the U.S. Bureau of Labor Statistics (BLS) and was made accessible through the Federal Reserve Bank of St. Louis Economic Data’s (FRED) online database. The CPI comprises the prices paid by consumers for a basket of energy services, which includes electricity, natural gas, fuel oil, and other forms of energy that households might use such as propane, kerosene, and coal.</p> 

<p align="center">For this research, we focused on comparing the predictive ability of ARIMA (AutoRegressive Integrated Moving Average) and ARIMAX (with eXogenous variables) models when used to predict the Monthly CPI in the tri-county area. ARIMA’s use of it’s own lags (past values) and lagged prediction errors can help model the underlying trends, seasonal effects, and any cyclical patterns that might be present in the historical data.</p> 

<p align="center">We also included eXogenous variables (external factors) into the ARIMAX model and compared results against the ARIMA model. For this, we incorporated historical hurricane events (e.g., Katrina, Wilma, etc.) and economic events (e.g., FPL rate changes, COVID-19 pandemic) to analyze their influence on the model’s predictions.</p>  

<p align="center">We conducted separate statistical tests to see the influence these events have on the CPI and their significance. First, we conducted Granger Causality Tests to see if hurricane events and/or economic changes like FPL rate increase/decreases and the COVID-19 pandemic can be used to predict fluctuations in the CPI. Then we conducted Independent T-Tests to compare the average CPI before and after each significant event. We partitioned the data into 'pre-event' and 'post-event' periods for each hurricane and economic event so that the independent t-test can statistically evaluate whether there is a significant difference in the CPI caused by these events.</p> 

<p align="center">For further testing, we also conducted an Event Study & Difference-in-Differences (DiD) Analysis Using Generalized Least Squares with Autoregressive Errors (GLSAR) Linear Regression Model for each event. Conducting an event study involves measuring the impact of each singular event on the time series. By combining this approach with a DiD analysis, it is possible to compare the changes in CPI of energy services before and after the hurricane events and economic shocks. The DiD approach controls for other unobserved factors that could be simultaneously affecting the outcome, thus isolating the true impact of these events. Using a GLSAR model further refines the analysis by adjusting for potential autocorrelation within the data, ensuring that the estimates are not biased due to the data's nature.</p> 

 <p align="center">Next, before we tested the models, we performed an Exploratory Data Analysis (EDA) and data pre-processing of the time-series data. First, we used Seasonal Decomposition of Time Series by Loess (STL) to extract and analyze the underlying seasonal, trend, and residual components separately. We then plotted ACF (Autocorrelation Function) & PACF (Partial Autocorrelation Function) charts to visually identify the appropriate lags for the AR (AutoRegressive) and MA (Moving Average) components in the ARIMA model. Since stationarity is a prerequisite for ARIMA modeling, we conducted ADF (Augmented Dickey-Fuller) tests to determine the stationarity of our time series data. To achieve stationarity, we performed first order difference transformation. This eliminates the unit root, stabilizing the mean of the time-series in preparation for modeling.</p> 

<p align="center">After EDA and data pre-processing, we used two methods for selecting the appropriate model parameters (p, d, q). First, we used Auto-ARIMA. which is an automated approach to find the best parameters. Then, we used Grid-Search, which is a more exhaustive search used to find the best parameters. Auto-ARIMA produced (1, 0, 0) as the appropriate parameters for both models. However, Grid-Search produced the best parameters (2, 0, 2) for both models, according to the information criteria (Log Likelihood, AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion)). This means that the current value of the series is regressed on the previous two values of the series and the forecast errors from the previous two periods.</p> 

<p align="center">To validate our model’s performance, we split the data into three training (seen data) and testing (unseen data) subsets, using 70/30, 80/20, and 90/10 ratios. The results for each split are compared for each model. We found that the ARIMAX (2, 0, 2) model performs better than the ARIMA (2, 0, 2) model at capturing changes in the CPI that correspond with the events that were included. To add confidence to our findings, we also implemented K-Fold Cross-Validation with TimeSeriesSplit and Blocked Cross-Validation. These methods provided more robust validation due to their evaluation of the model’s performance over multiple time periods rather than a single fixed period.</p> 

<p align="center">However, the result from both models shows heteroscedasticity (volatility) in the residuals (difference between actual values & predicted values) that the models are not able to capture. This is proven by the results of McCleod-Li tests performed on the residuals of both ARIMA (2, 0, 2) and ARIMAX (2, 0, 2) models. The McCleod-Li test is used to detect nonlinearity and conditional heteroscedasticity (changes in variance) in a time series. To address this, we extended our models to include GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) & GARCH AR-X (AutoRegressive with eXogenous variables) variants to model the predicted volatility from the actual volatility in the time series and compare the results. Grid-Search provided the following parameters: GARCH (2, 1), EGARCH (2, 2), TGARCH (1, 2, 2), GARCH AR-X (2, 1), EGARCH AR-X (2, 0), and TGARCH AR-X (2, 2, 2). We found that GARCH and its variants perform better at modeling volatility when we incorporate AR-X components into the mean equation of the model based on the evaluated metrics. However, the coefficients of the EGARCH (2, 2) show to be most beneficial in understanding the dynamics of the data due to their statistical significance. This means that volatility of the time series is best modeled as a function of both its own lagged values and the lagged squared errors up to two periods ago. To further validate the performance of the GARCH and GARCH AR-X models, we conducted Expanding Window Cross-Validation and Purged Cross-Validation. These methods evaluate volatility under different market conditions over time.</p>

<p align="center">The result is a hybrid, integrated ARIMA (2, 0, 2) - EGARCH (2, 2) model that works to predict the average of the time series along with its volatility over time. Ultimately, providing a framework that captures the data's trajectory and extent of its variability.</p>

<h3 align="center">ARIMA (Autoregressive Integrated Moving Average) Model</h3> 

<p align="center">$(1 - \phi_1 L - \phi_2 L^2 - \ldots - \phi_p L^p)(1 - L)^d Y_t = (1 + \theta_1 L + \theta_2 L^2 + \ldots + \theta_q L^q) \epsilon_t$</p> 

<p align="center">$\text{where:}$</p> 

<p align="center">$Y_t$: The value of the series at time $t$, which is the observation we aim to model or forecast.</p> 

<p align="center">$\phi_1, \phi_2, \ldots, \phi_p$: Autoregressive coefficients that quantify the influence of the past $p$ values of the series on its current value.</p> 

<p align="center">$L$: Lag operator, such that applying $L$ to $Y_t$ yields $L Y_t = Y_{t-1}$, the previous value of the series.</p> 

<p align="center">$p$: Order of the autoregressive part, indicating the number of lagged observations of the series included in the model.</p> 

<p align="center">$(1 - L)^d$: Differencing operator applied $d$ times to the series to achieve stationarity, which is often necessary for ARIMA models.</p> 

<p align="center">$d$: Order of differencing, representing how many times the data has been lagged or differenced to remove non-stationarity.</p> 

<p align="center">$\theta_1, \theta_2, \ldots, \theta_q$: Moving average coefficients that quantify the influence of past $q$ forecast errors on the current value of the series.</p> 

<p align="center">$q$: Order of the moving average part, indicating the number of lagged forecast errors included in the model.</p> 

<p align="center">$\epsilon_t$: Error term at time $t$, which is a stochastic term that accounts for the randomness or unpredictability in the series at time $t$.</p> 

<h3 align="center">ARIMAX (Autoregressive Integrated Moving Average with eXogenous variables) Model</h3> 

<p align="center">$Y_t = c + \sum_ {i=1}^{p} \phi_i Y_{t-i} + \sum_ {j=1}^{q} \theta_j \varepsilon_{t-j} + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \varepsilon_t$</p> 

<p align="center">where:</p> 

<p align="center">- $Y_t$: The dependent variable representing the value of the time series at time $t$. It is the variable that the model aims to forecast or explain.</p> 

<p align="center">- $c$: A constant term or intercept in the equation, representing the baseline value of $Y_t$ when all other variables are equal to zero.</p> 

<p align="center">- $\phi_i$: Coefficients for the autoregressive (AR) terms. These coefficients measure the impact of the past values of the series on its current value.</p> 
<p align="center">Specifically, $\phi_i$ is the coefficient of the $i$-th lagged value of $Y_t$, indicating how much a change in $Y_{t-i}$ affects $Y_t$.</p> 

<p align="center">- $Y_{t-i}$: The $i$-th lagged value of the time series, meaning the value of $Y$ at time $t-i$. These lagged values are used to capture the influence of past values of the time series on its current value.</p> 

<p align="center">- $\theta_j$: Coefficients for the moving average (MA) terms. These coefficients represent the impact of past forecast errors on the current value of the series. Each $\theta_j$ is associated with the $j$-th lagged forecast error.</p> 

<p align="center">- $\varepsilon_{t-j}$: The forecast error at time $t-j$, which is the difference between the observed value and the predicted value at that lag. It represents the unpredicted part of $Y$ at time $t-j$.</p> 

<p align="center">- $\beta_1, \beta_2$: Coefficients for the exogenous variables $X_{1,t}$ and $X_{2,t}$. These coefficients measure the impact of the external variables on the dependent variable $Y_t$. They show how much a unit change in each exogenous variable affects $Y_t$.</p> 

<p align="center">- $X_{1,t}, X_{2,t}$: The exogenous variables at time $t$. In this model, $X_{1,t}$ represent hurricane events, capturing their impact on $Y_t$, while $X_{2,t}$ represent economic events, indicating how economic factors at time $t$ affect $Y_t$.</p> 

<p align="center">- $\varepsilon_t$: The error term at time $t$, accounting for random fluctuations or noise in $Y_t$ that are not explained by the autoregressive, moving average, or exogenous components of the model. This term ensures that the model accounts for stochastic variations in the time series.</p> 

