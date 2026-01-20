# Predict Health-risks (Machine Learning) 
Project นี้ทำขึ้นเพื่อศึกษาการทำงานและวิธีการสร้าง Model Machine Learning ในการทำนายความเสี่ยงต่อสุขภาพโดยใช้ ชุดข้อมูลจาก website Kaggle เพื่อ train Model 

## เครื่องมือที่ใช้

 - **Language :** Python 3.13.9
 - **Text Editor :** Anitigravity
 - **Libraries :** pandas , matplotlib , seaborn , scikit-learn , joblib , statsmodels
 - **AI Assistant :** Gemini 3
 - **Markdown Editor :** Stackedit.io

## วิธีการทำงาน

 1. **เตรียมข้อมูล Dataset :** ใช้ข้อมูล dataset จาก [kaggle](https://www.kaggle.com/datasets/miadul/lifestyle-and-health-risk-prediction/data)
    ดาวน์โหลดไฟล์ .csv แล้วนำมาวาง บน **GitHub** ไว้ดึงข้อมูลผ่าน URL
        เข้า Editor ได้โดยตรง
 2. **ติดตั้ง Library ที่จำเป็นต่อการสร้างโมเดล และ วิเคราะห์ข้อมูล :** ติดตั้งผ่าน command prompt

 3. **สำรวจข้อมูล :** สร้าง Data Frame และสำรวจข้อมูลโดยรวม เช็ค Missing Values , Outliers และ เลือกตัวแปร y หรือ Target
    **(ในที่นี้คือ Health Risk)** พร้อมสำรวจข้อมูล

 4. **ขั้นตอนการเลือกข้อมูล :** เลือกข้อมูลที่จะเข้ามา Train Model โดยการ Plot กราฟ โดยที่ผมเลือกที่จะทำแค่ **Bฺivariate analysis** เพื่อดูความต่างของตัวแปร y ที่เทียบกับ x ว่ามีการเปลี่ยนแปลงของ กราฟมากแค่ไหน หากตัวแปร y มีการเปลี่ยนแปลงที่ชัดเจนแปลว่า ตัวแปร x ตัวนั้นมีผลต่อการเปลี่ยนแปลงของ **Target** อย่างมีนัยสำคัญ ดังนั้นจะพิจารณาไม่ตัดตัวแปรนั้นออกตอนลดค่า **Multicollinearity**
**(ตอนทำ VIF)** 
 5. **Encoding :** เปลี่ยนตัวแปรที่เก็บ Data Type ที่เป็น Object ให้เป็น Numeric เพื่อให้สามารถนำไป Train Model ได้ โดยที่ผมเลือกใช้ **Label Encoding** ในการเปลี่ยนตัวอักษรเป็นตัวเลข เหตุผลที่เลือกใช้ **Label Encoding** แทนใช้ **one hot encoding** เนื่องจากเป็นวิธีที่เข้าใจง่ายที่สุดและการทำ one hot encoding จะมีการแยก column ออกมาเพิ่มซึ่งอาจทำให้ตอนกลับมาอ่านทำความเข้าใจยาก แล้วยังเสี่ยงที่ตอนนำไป Train Model อาจเกิด **Curse of Dimensionality** ที่ข้อมูลมีการกระจายตัวมากจน Model ไม่สามารถจับ pattern ที่ส่งผลต่อ Target เพื่อทำนายได้ และ เสี่ยงเกิด **Multicolinearity** ที่ค่าความสัมพันธ์ของตัวแปรกันและกันมีสูงมากจน Model ไม่รู้ว่าตัวไหนสำคัญต่อการนำมาทำนาย เช่น ค่า BMI ที่ คำนวณจาก ส่วนสูงและน้ำหนัก หากนำสองตัวแปรนี้มา Train จะทำให้การทำนายของโมเดลผิดพลาดและคลาดเคลื่อนได้
 6. **Setup and Training Model :**

    - กำหนดตัวแปร x และ y ในที่นี้ตัวแปร x ผมเลือกจะไม่ตัดตัวแปรไหนออกเลย
   (นอกจาก Target ที่ไว้กำหนด y) แล้วให้ y ที่เป็น Target คือ Health
   risk
  
     - กำหนดข้อมูล Training set และ Test set ซึ่งในที่่นี้จะแบ่ง Training
   set ไว้ 70% และ Test set ที่ 30% โดยเลือกใช้ Algorithm Random Forest แล้วเริ่ม train ที่ X_trian , y_train
   
 7. **ทดสอบ Model และ วัดผล :** 

    - **การทดสอบโมเดล ครั้งที่ 1 :** วัดออกมาได้ precision , recall , f1-score ที่ 
     Low risk : 0.98 , 0.99 , 0.99
     High risk : 1.00 , 0.99 , 0.99
     แม้ค่าที่ได้จะออกมาสูงมากแต่เมื่อหาค่า VIF score
     พบว่า ตัวแปรที่มีค่า VIF จากมากไปน้อยมีดังนี้
     
       |Feature|VIF|
       |--|--|
       |weight  | 76.854526  |
       |height|59.298171|
       |bmi|43.741725|
       |sleep|24.467691|
       |age|8.369773|
       |exercise|3.937447|
       |profession|3.395428|
       |sugar_intake|2.675628|
       |married|2.523411|
       |alcohol|1.338338|
       |smoking|1.243555|
       
       จากตารางจะพบว่า ค่า VIF สูงมากจน Model จะทำนายพลาดอย่างแน่นอน ดังนั้นจากการสังเกตุแล้ว ค่า height และ weight ที่นำมาคำนวณค่า bmi อยู่ก่อนแล้วไม่จำเป็นต้องมี 2 ตัวแปรนี้ดังนั้นจึงตัดออกเหลือไว้แค่ค่า bmi ในการ train ใหม่       
       #      

    - **การทดสอบครั้งที่ 2 :** หลังจากทดสอบผลอีกครั้ง พบว่า ค่า VIF ของตัวแปรอื่นๆ ลดตามไปด้วยเมื่อนำ ค่า height และ weight ออก แต่ค่า
   VIF ที่ยังสูงอยู่ยังมี sleep ที่ค่า VIF เกิน 10

   
        |Feature|VIF|
        |--|--|
        |sleep|13.286466|
        |bmi|9.129715|
        |age|7.171176|
        |exercise|3.712318|
        |profession|3.196422|
        |sugar_intake|2.567769|
        |married|2.435139|
        |alcohol|1.330617|
        |smoking|1.239583|
   
    #
    - **การทดสอบครั้งที่ 3 :** นำตัวแปรที่มีค่า VIF สูงออกซึ่งคือ sleep แล้ว Train model อีกรอบ เมื่อ train เสร็จแล้วเช็ค VIF Score คราวนี้จะพบว่า ค่าทุกค่ามี VIF score ต่ำกว่า 10 
     
        |Feature|VIF|
        |--|--|
        |bmi|7.007663|
        |age|6.244758|
        |exercise|3.507706|
        |profession|3.074933|
        |sugar_intake|2.501619|
        |married|2.379730|
        |alcohol|	1.322557|
        |smoking|1.233808|
       
   #

 8. **บันทึก Model :** ใช้ library joblip ในการบันทึก Model ลงในไฟล์ .pkl
