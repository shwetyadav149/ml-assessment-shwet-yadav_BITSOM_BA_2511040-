# Part B: Business Case Analysis
Promotion Effectiveness at a Fashion Retail Chain
# B1. Problem Formulation
# (a) Machine Learning Formulation
# Problem Definition

The objective is to select the best promotion for each store each month to maximize demand. In my implementation, I modeled this by predicting items_sold and then choosing the promotion with the highest predicted value.

# Target Variable
items_sold (monthly units sold per store)
# Input Features (as used in my solution)
Promotion: one-hot encoded (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points)
Store attributes: store_size, location_type
Competition: competition_density
Time features (engineered): year, month, day_of_week, is_month_end
(Optional extensions consistent with my approach): lagged sales, moving averages
# Problem Type
Supervised Regression
# Justification
The output is continuous (volume) → regression.
I trained multiple regressors (Linear, Ridge, Lasso, Random Forest) and evaluated via RMSE/MAE/R².
The decision policy is argmax over predicted items_sold across promotion options per store-month.
# (b) Why items_sold is Better than Revenue
# Observations from My EDA
Promotions significantly change quantity sold (boxplots by promotion show different medians/spreads).
Price/discounts distort revenue; two promotions can yield similar revenue but very different volumes.
# Why Volume is More Reliable
Directly measures demand response to promotions.
Less sensitive to pricing changes and product mix.
Aligns with operational goals: inventory planning, footfall conversion, and stock turnover.
# Broader Principle

Choose a target that directly reflects the decision you want to optimize and is minimally confounded by external factors.
Here, volume is the clean signal for “promotion effectiveness.”

# (c) Alternative Modelling Strategy (beyond a single global model)
# Evidence from My Work
EDA showed heterogeneity: different store sizes and locations respond differently.
Time features showed seasonality (monthly patterns) affecting outcomes.
# Recommended Strategy
Segmented Models
Separate models for urban/semi-urban/rural and/or store_size buckets.
Hierarchical / Multi-level Model
Global model + store-level adjustments (store intercepts or embeddings).
Hybrid
Global Random Forest with interaction features (promotion × location, promotion × store_size).
# Why This Works
Captures context-specific promotion effects.
Reduces bias from averaging across dissimilar stores.
Improves personalization of recommendations.
# B2. Data and EDA Strategy
# (a) Data Joining Strategy
# Source Tables
Transactions (sales)
Store attributes
Promotion details
Calendar (weekend/festival flags)
# Join Keys
store_id
transaction_date (aligned to month)
# Final Dataset Grain

One row = Store × Month × Promotion

(For training, use the observed promotion; for recommendation, score all candidate promotions for the same row.)

# Aggregations (aligned with my implementation)
items_sold = sum per store-month
competition_density = monthly average
Promotion = categorical (one-hot)
Calendar → derive month, day_of_week, is_month_end, festival flags
# Why This Grain
Matches the decision cadence (monthly).
Reduces noise vs. transaction-level data.
Compatible with time-based splitting.
# (b) EDA Performed (with Insights → Decisions)
1) Target Distribution (Histogram/KDE)
Insight: Slight skew and presence of extreme values.
Action: Applied IQR-based outlier handling with median replacement (not deletion) to stabilize training without losing data.
2) Promotion vs Sales (Boxplots)
Insight: Clear separation in medians/variance across promotions → promotions matter.
Action: Keep promotion as strong categorical feature; consider interactions with store/time.
3) Competition vs Sales (Scatter)
Insight: Negative relationship (higher competition → lower sales).
Action: Include competition_density; tree models can capture non-linear effects.
4) Time Trends (Monthly/Weekly Plots)
Insight: Seasonality (peaks during certain months; weekday effects).
Action: Engineer month, day_of_week, is_month_end; these improved model performance.
5) Correlation Heatmap
Insight: Identified relationships and potential redundancies.
Action: Retained informative features; used Ridge/Lasso to manage multicollinearity.
6) Store/Location Effects (Box/Violin Plots)
Insight: Distribution differs by store_size/location_type.
Action: Motivates segmentation or interaction features.
# (c) Promotion Imbalance (80% No-Promotion)
# Risk
Model may learn a bias toward “no promotion”, underestimating promotion lift.
# Mitigation (consistent with my approach)
Explicit promotion encoding (one-hot).
Prefer tree-based models (Random Forest) → robust to imbalance and non-linearities.
If needed:
Reweighting samples or stratified batching
Evaluate per-promotion performance, not just overall RMSE
# B3. Model Evaluation and Deployment
# (a) Train-Test Split & Metrics
# What I Implemented
Time-based 80–20 split:
Train = earliest 80%
Test = most recent 20%
Verified with date ranges (train precedes test).
# Why Random Split is Wrong
Causes data leakage (future patterns leak into training).
Overestimates performance; not realistic for forecasting.
# Metrics Used (and Why)
RMSE
Penalizes large errors → important for avoiding big stocking mistakes.
MAE
Interpretable average error (e.g., “±X items”).
R²
Variance explained; sanity check for model fit.
# Interpretation in My Results
Selected model with lowest Test RMSE, reasonable MAE, and good R².
Also checked Train vs Test RMSE gap for overfitting.
# (b) Explaining Different Recommendations (Feature Importance)
# Scenario
Store 12:
December → Loyalty Points
March → Flat Discount
# How I Investigated (based on my solution)
Used Random Forest feature importance.
Examined top drivers: month, promotion_type, competition_density, store attributes.
# Insights
December (festive/high demand):
Loyalty incentives increase repeat purchase and basket expansion.
March (post-season/low demand):
Price sensitivity higher → discounts more effective.
# Communication to Marketing
Show feature importance ranking.
Provide simple narrative:
“In high-demand months, rewards maximize value; in low-demand months, discounts stimulate demand.”
Optionally add scenario charts (predicted sales by promotion for that store-month).
# (c) Deployment Strategy (End-to-End)
1) Model Packaging
Save trained pipeline (preprocessing + model):
joblib.dump(best_model, "promotion_model.pkl")
2) Monthly Scoring Pipeline
Ingest new month’s data (store attributes + calendar + competition).
Apply same preprocessing (ColumnTransformer with scaling + one-hot).
For each store-month, simulate all 5 promotions:
Create 5 rows (one per promotion)
Predict items_sold for each
3) Recommendation Logic
For each store:
Select promotion with highest predicted items_sold
Output: recommendation table (store_id, month, best_promotion, expected_sales)
4) Monitoring (essential)
Performance: RMSE/MAE on recent actuals
Data drift: shifts in key features (e.g., competition, footfall proxies)
Prediction stability: sudden swings in recommended promotions
5) Retraining Policy
Trigger retraining when:
RMSE degrades beyond threshold
Seasonal pattern shifts
New promotions introduced
Suggested cadence: quarterly or bi-annual, plus event-based retraining
# Final Conclusion (Aligned with My Solution)
The problem is best solved via supervised regression with items_sold as target.
EDA-driven feature engineering (time features, promotion encoding, competition) significantly improved performance.
Median-based outlier replacement stabilized the model without losing data.
Time-based split ensured realistic evaluation.
Among tested models (Linear, Ridge, Lasso, Random Forest with alpha tuning for regularization),
# Random Forest performed best due to its ability to capture non-linear interactions and robustness to outliers/imbalance.
The deployed system can score all promotions per store-month and recommend the one that maximizes expected sales, with ongoing monitoring and retraining for reliability.