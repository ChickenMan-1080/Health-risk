# Predict Health-risks (Machine Learning) ##อยู่ระหว่างเรียบเรียงข้อมูล
Project นี้ทำขึ้นเพื่อศึกษาการทำงานและวิธีการสร้าง Model Machine Learning ในการทำนายความเสี่ยงต่อสุขภาพโดยใช้ ชุดข้อมูลจาก website Kaggle เพื่อ train Model 

## เครื่องมือที่ใช้

 - **Language :** Python 3.13.9
 - **Text Editor :** Anitigravity
 - **Libraries :** pandas , matplotlib , seaborn , scikit-learn , joblib , statsmodels
 - **AI Assistant :** Gemini 3
 - **Markdown Editor :** Stackedit.io

## ขั้นตอนการทำงาน
 1. **เตรียมข้อมูล Dataset :** ใช้ข้อมูล dataset จาก [kaggle](https://www.kaggle.com/datasets/miadul/lifestyle-and-health-risk-prediction/data) ดาวน์โหลดไฟล์ .csv แล้วนำมาวาง บน **GitHub** ไว้ดึงข้อมูลผ่าน URL เข้า Editor ได้โดยตรง
 
 2. **ติดตั้ง Library ที่จำเป็นต่อการสร้างโมเดล และ วิเคราะห์ข้อมูล :** ติดตั้งผ่าน command prompt
 3. **สำรวจข้อมูล :** สร้าง Data Frame และสำรวจข้อมูลโดยรวม เช็ค Missing Values , Outliers และ เลือกตัวแปร y **(ในที่นี้คือ Health Risk)** พร้อมสำรวจข้อมูล
 4. **ขั้นตอนการเลือกข้อมูล :** เลือกข้อมูลที่จะเข้ามา Train Model โดยการ Plot กราฟ เพื่อดูความต่างของตัวแปร y ที่เทียบกับ x ว่ามีการเปลี่ยนแปลงของ กราฟมากแค่ไหน หากตัวแปร y มีการเปลี่ยนแปลงที่ชัดเจนแปลว่า ตัวแปร x ตัวนั้นมีผลต่อการเปลี่ยนแปลงของ Target อย่างมีนัยสำคัญ ดังนั้นจะพิจารณาไม่ตัดตัวแปรนั้นออกตอนต้องลดค่า multicollinearity **(ตอนทำ VIF)**
