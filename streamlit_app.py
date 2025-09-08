import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Load data ---
hist = pd.read_csv("data/Historical_Data.csv")
couriers = pd.read_csv("data/Courier_Performance.csv")
orders = pd.read_csv("data/Orders_Input.csv")

# --- Train AI Model ---
X = hist[["Destination_City", "Order_Value_SAR", "Is_Fragile", "Weight_KG", "Customer_Type", "Courier_Used"]]
X = pd.get_dummies(X)
y = hist["Actual_Success"]

model = RandomForestClassifier()
model.fit(X, y)

# --- Streamlit UI ---
st.title("📦 AI Dynamic Shipping Optimization")
st.write("AI selects the best courier for each order based on historical data, cost, speed, and reliability.")

for idx, order in orders.iterrows():
    st.subheader(f"Order {order['Order_ID']} – {order['Destination_City']} ({order['Order_Value_SAR']} SAR)")

    results = []
    for courier in couriers["Courier_Name"]:
        test = order.copy()
        test["Courier_Used"] = courier
        test_df = pd.DataFrame([test])
        test_df = pd.get_dummies(test_df).reindex(columns=X.columns, fill_value=0)

        prob = model.predict_proba(test_df)[0][1]
        cost = couriers.loc[couriers.Courier_Name==courier,"Avg_Cost_AED"].values[0]
        speed = couriers.loc[couriers.Courier_Name==courier,"Avg_Delivery_Days"].values[0]
        score = prob*100 - cost*0.5 - speed*2

        results.append((courier, prob, cost, speed, score))

    df = pd.DataFrame(results, columns=["Courier", "Predicted_Success", "Cost(AED)", "Speed(Days)", "AI_Score"])
    best = df.sort_values("AI_Score", ascending=False).iloc[0]

    st.dataframe(df.sort_values("AI_Score", ascending=False))
    st.success(f"✅ Recommended: **{best['Courier']}** (Success {best['Predicted_Success']:.1%}, Cost {best['Cost(AED)']}, Speed {best['Speed(Days)']} days)")
