# MLB Betting Model v3.1 — Results & Evaluation Demo
# Author: Hrishikesh Kochale
# ----------------------------------------------------------
# This script simulates model-generated recommended bets,
# compares them with actual outcomes, and calculates the
# model’s overall accuracy, profitability, and ROI.
#
# ⚠️ DISCLAIMER:
# ----------------------------------------------------------
# This script is for educational and research purposes only.
# All data below is synthetic and for demonstration.
# Real-world betting outcomes depend on many unpredictable
# factors such as player injuries, weather, and lineup changes.
# Do NOT use this for actual betting or financial decisions.
# ----------------------------------------------------------

import pandas as pd
import numpy as np

# =====================================================
# 1. Simulated Model Recommendations
# =====================================================

np.random.seed(42)
games = [
    "Dodgers @ Phillies",
    "Yankees @ Blue Jays",
    "Tigers @ Mariners",
    "Braves @ Mets",
    "Astros @ Rangers",
    "Giants @ Padres",
    "Cubs @ Cardinals",
    "Orioles @ Rays",
    "Red Sox @ Guardians",
    "Marlins @ Nationals"
]

df = pd.DataFrame({
    "Game": games,
    "Bet_Type": np.random.choice(["Moneyline - Away", "Moneyline - Home", "Over 7.5", "Under 7.5"], 10),
    "Probability_%": np.random.uniform(55, 70, 10).round(2),
    "Odds": np.random.choice([-120, -110, +110, +125, +140], 10),
    "EV_per_$": np.random.uniform(0.01, 0.05, 10).round(3),
    "Risk_Level": np.random.choice(["Low", "Moderate", "High"], 10),
})

df["Recommendation"] = np.where(df["EV_per_$"] > 0.02, "✅ Bet", "🚫 Pass")

# =====================================================
# 2. Simulated Actual Results
# =====================================================

df["Actual_Result"] = np.random.choice([1, 0], 10, p=[0.6, 0.4])
df["Came_True"] = df["Actual_Result"].map({1: "Yes", 0: "No"})

# =====================================================
# 3. Accuracy and ROI Calculation
# =====================================================

total_bets = (df["Recommendation"] == "✅ Bet").sum()
won_bets = df[(df["Recommendation"] == "✅ Bet") & (df["Actual_Result"] == 1)].shape[0]
model_accuracy = round(won_bets / total_bets * 100, 2) if total_bets > 0 else 0

avg_ev = df.loc[df["Recommendation"] == "✅ Bet", "EV_per_$"].mean()
roi = round(avg_ev * 100, 2)

# =====================================================
# 4. Final Summary Table
# =====================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Bets Placed",
        "Winning Bets",
        "Model Accuracy (%)",
        "Average Expected Value (EV per $)",
        "Simulated ROI (%)"
    ],
    "Value": [
        total_bets,
        won_bets,
        f"{model_accuracy}%",
        f"${avg_ev:.3f}",
        f"{roi}%"
    ]
})

# =====================================================
# 5. Display Results
# =====================================================

print("\n===== MLB Betting Model v3.1 — Recommended Bets =====\n")
print(df[["Game", "Bet_Type", "Probability_%", "Odds", "EV_per_$", "Risk_Level", "Recommendation", "Came_True"]].to_string(index=False))

print("\n===== Model Performance Summary =====\n")
print(summary.to_string(index=False))

# =====================================================
# 6. Save Outputs
# =====================================================

df.to_csv("mlb_bet_recommendations.csv", index=False)
summary.to_csv("mlb_model_final_summary.csv", index=False)

print("\n✅ Output saved:")
print("• mlb_bet_recommendations.csv — detailed bet list")
print("• mlb_model_final_summary.csv — accuracy summary")

print("\nEnd of Script — MLB Betting Model v3.1 Results & Evaluation Demo")
