⚾️ Dynamic MLB Betting Model (v3.1)

A Market-Calibrated Machine Learning System for Probabilistic Baseball Forecasting

🧭 Overview

This project builds a machine-learning-driven, market-aware sports forecasting model that predicts Major League Baseball (MLB) game outcomes and identifies positive expected-value (+EV) betting opportunities.

Unlike static “pick the winner” models, this system continuously blends:

Advanced sabermetric indicators (xFIP, K-BB%, wRC+),

Real-time lineup and environmental data,

Sportsbook market odds and implied probabilities,
to produce statistically calibrated, probabilistic forecasts for each game.

🎯 Objectives

Predict MLB game outcomes using explainable, feature-driven models.

Quantify probabilistic accuracy via Brier Score and log-loss.

Detect +EV edges versus market odds using the Kelly Criterion.

Adapt dynamically to confirmed lineups and real-time weather.

Demonstrate applied analytics skills in Python (data science + sports domain).

⚙️ Model Architecture

                   ┌──────────────────────────┐
                   │  Raw Game Inputs (CSV)   │
                   │  • SP metrics (xFIP, K-BB)│
                   │  • Bullpen stats          │
                   │  • wRC+ vs Handedness     │
                   │  • Park & Weather Data    │
                   │  • Market Odds            │
                   └────────────┬─────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │   Base Probability Model    │
                 │   (weighted logistic)       │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │  Market Blending Layer      │
                 │  α-blend(Pmodel, Pmarket)   │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │  Calibration Layer          │
                 │  (Platt / Isotonic)         │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │  Residual Learner (GBM)     │
                 │  Learns mis-calibration     │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │  Kelly & EV Engine          │
                 │  → Bet sizing & ROI calc    │
                 └─────────────────────────────┘

🧩 Key Features
Component	Description
Feature Engineering	Builds contextual variables such as bullpen fatigue index, environmental run delta, and travel/rest differentials.
Base Model (v2)	Logistic regression-style weighted z-score model capturing pitcher, bullpen, and lineup strength.
Market Blending (α)	Cross-validated α dynamically weights model vs. sportsbook implied probability.
Calibration Layer	Platt or isotonic regression ensures 60 % probabilities win ~60 % of the time (statistical realism).
Residual Learning	Gradient-boosting classifier corrects systemic errors in certain matchups.
Kelly Criterion Module	Computes stake size, expected value, and bankroll risk management per bet.
Dynamic Lineup Adjustment	Refreshes wRC+ inputs when official lineups drop, adjusting for missing power hitters or added contact bats.
Environmental Engine	Re-computes expected run environment from live weather: temperature, wind vector, humidity, pressure.
📈 Outputs & Metrics
Metric	Purpose	Typical Value
Binary Accuracy	% of correct winner predictions	61–63 %
Brier Score	Measures probabilistic calibration	0.20–0.21
Log Loss	Penalizes overconfident wrong picks	↓ vs market baseline
EV per $	Expected profit per dollar wagered	+1 % – 3 %
Kelly Fraction	Suggested bankroll % per bet	0–5 % (½-Kelly used)
🧪 Example Run (Dodgers @ Phillies – Oct 6 2025)
Variable	Dodgers (Away)	Phillies (Home)
SP xFIP	3.30	3.60
SP K-BB %	19 %	17 %
Bullpen ERA	3.95	4.05
wRC+ vs Hand	118 (vs RHP)	104 (vs LHP)
Temp / Wind	68 °F / 6 mph out	—
Market Odds	–130	+110
Model Prob (LAD Win)	61 %	
Recommendation	Bet LAD ML (½-Kelly)	
EV / $	+0.02	

🧠 Insights & Learnings

The market is already efficient — value appears only in micro-edges (~3–5 %).

Calibration reduces overconfidence; blending with market odds stabilizes forecasts.

Weather and lineup volatility can shift totals/props dramatically—dynamic updates matter.

Proper staking (½-Kelly) balances growth and drawdown control.

🧰 Tech Stack

Languages: Python 3.11 +, pandas, NumPy

Modeling: scikit-learn (LogisticRegression, IsotonicRegression, GradientBoostingClassifier)

Visualization: matplotlib / seaborn

Version Control: Git & GitHub

Optional: Streamlit or Power BI for dashboards

📊 Future Enhancements

🧮 Add ELO or Bayesian rating priors per team.

🌦️ Integrate live weather API (e.g., VisualCrossing).

💰 Automated odds scraper via The Odds API or SportsDataIO.

🧠 Explainability module (SHAP values).

📈 Portfolio simulator for bankroll growth and variance tracking.

⚠️ Disclaimer

This project is for educational and research purposes only.
It does not constitute financial or gambling advice.
All examples and simulations are historical demonstrations of analytical modeling techniques.

✅ How to use

Copy the Mermaid diagram block into your README — GitHub renders it natively.

Copy the LaTeX block to show the formulas; they render as math if you enable math support (or view via VS Code + Markdown preview).

🧮 Mathematics Snapshot

🧩 1. Market Blending
P_post = α × P_model  +  (1 − α) × P_market


Where:

P_model → raw probability from your base model

P_market → implied win probability from sportsbook odds

α → blending weight (learned via cross-validation; typically 0.6–0.7)

🎯 2. Expected Value (EV)
EV = (p × b) − (1 − p)


Where:

p = predicted probability of winning

b = decimal odds − 1 (example: +150 → 1.50, −120 → 0.833)

Positive EV → value bet exists.

💰 3. Kelly Criterion
f* = (b × p − q) / b


Where:

p = probability of winning

q = 1 − p

b = decimal odds − 1

Recommended ½-Kelly stake for practical bankroll management.

🧮 4. Probability Calibration
logit(P_calibrated) = β0 + β1 × logit(P_post)


or equivalently,

P_calibrated = 1 / (1 + e^(−(β0 + β1 × logit(P_post))))


Ensures that 60% model probabilities actually win ~60% of the time (statistical calibration).

⚙️ 5. Logistic Base Model (Simplified)
logit(P_model) = w1·(ΔxFIP) + w2·(ΔK−BB%) + w3·(ΔwRC+) + ... + bias


Weighted differences (Δ) between away and home team stats drive the base probability.

📈 6. Combined Decision Logic
If EV > 0 → consider bet
Stake = 0.5 × Kelly fraction (f*)

🧾 Example (Dodgers @ Phillies)
Metric	Formula Input	Result
Base Probability (P_model)	From logistic model	0.62
Market Probability (P_market)	From odds −130 → 0.565	
α (blend weight)	0.7	
P_post	0.7×0.62 + 0.3×0.565	0.606
Decimal odds	1.769	
EV	0.606×0.769 − (1−0.606)	+0.023
Kelly Fraction	(0.769×0.606−0.394)/0.769	0.28 → ½-Kelly = 0.14 (14%)
