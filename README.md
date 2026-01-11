# Predict Health-risks (Machine Learning) ##อยู่ระหว่างเรียบเรียงข้อมูล
Project นี้ทำขึ้นเพื่อศึกษาการทำงานและวิธีการสร้าง Model Machine Learning ในการทำนายความเสี่ยงต่อสุขภาพโดยใช้ ชุดข้อมูลจาก website Kaggle เพื่อ train Model 

## เครื่องมือที่ใช้

 - **Language :** Python 3.13.9
 - **Text Editor :** Anitigravity
 - **Libraries :** pandas , matplotlib , seaborn , scikit-learn , joblib
 - **AI Assistant :** Gemini 3
 - **Markdown Editor :** Stackedit.io

## ภาพรวมการทำงาน
 1. **เตรียมข้อมูล DataSet :** ใช้ข้อมูล dataset จาก [kaggle](https://www.kaggle.com/datasets/miadul/lifestyle-and-health-risk-prediction/data) ดาวน์โหลดไฟล์ .csv แล้วนำมาวาง บน **GitHub** ไว้ดึงข้อมูลเข้า Editor 
 
 2. **ติดตั้ง Library ที่จำเป็นต่อการสร้างโมเดล และ วิเคราะห์ข้อมูล :** ติดตั้งผ่าน command prompt
 3. **สำรวจข้อมูลโดยรวม :** สร้าง Data Frame และสำรวจข้อมูลโดยรวม เช็ค (Missing Values , Outliner) และ เลือกตัวแปร y **(ในที่นี้คือ Health Risk)**
