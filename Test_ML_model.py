import joblib
import pandas as pd


model = joblib.load('health_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')


#test data person 1 and 2
test_data = [
    [45, 2, 1, 0, 0, 1, 1, 28.5], 
    [25, 5, 0, 0, 0, 0, 2, 22.0] 
]


df_test = pd.DataFrame(test_data, columns=model_columns)


predictions = model.predict(df_test)
probabilities = model.predict_proba(df_test)


for i, pred in enumerate(predictions):
    risk = "High Risk" if pred == 1 else "Low Risk"
    conf = probabilities[i][pred] * 100
    print('คนที่', i+1, 'มีความเสี่ยงสุขภาพ', risk, 'ค่าความมั่นใจ', conf, '%')