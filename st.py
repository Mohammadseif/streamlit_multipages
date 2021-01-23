import pickle
from pandas import DataFrame
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
from streamlit import beta_columns
from PIL import Image
import streamlit as st
from sklearn.externals import joblib


st.header("Goharzamin Iron Ore Mine CIBB")

pages = ["Home", "Input Blast Data","Prediction"]
page = st.sidebar.radio("Menu Bar", options=pages)
#st.title(page)
if page == "Input Blast Data":
    st.subheader("متغیرها را وارد کنید")
    @st.cache(allow_output_mutation=True)
    def get_data():

        return [] 

    BBN = st.text_input('شماره بلوک انفجاری')

    STONE=['Magnetite', 'Hematite', 'Soil', 'Waste Rock', 'Cong-Waste Rock',
        'Conglomerate', 'magn-Waste Rock', 'Soil-Cong', 'Soil-Waste Rock',
        'So-Co-Waste Rock']
    RT=st.selectbox('Rock Type:',STONE)

    HD=st.radio('Hole Diametee:',( '8.5', '10.5',  '6.5',  '7.5'))

    col1, col2,col3 =st.beta_columns(3)

    with col1:
        LF = st.slider("LF", 0, 20, 0)
    with col2:
        LM = st.slider("LM", 0, 20, 0)
    with col3:
        WM = st.slider("WM", 0.0, 1.0, 0.0)

    col4, col5,col6 =st.beta_columns(3)

    with col4:
        WL = st.slider("WL", 0.0, 1.0, 00.0)
    with col5:
        RLW = st.slider("RLW", 0, 25, 0)
    with col6:
        FE = st.slider("FE", 0.0, 1.0, 0.0)


    IMAGE = st.file_uploader("Upload", type="jpg")

    if st.button("Add row"):
        get_data().append({'BBN':BBN,'LF':LF,'LM':LM,'WM':WM,'WL':WL,'RLW':RLW,'FE':FE,'RT':RT,'HD':HD })
        dframe = pd.DataFrame(get_data())
        dframe.to_csv('ge17.csv')
        
    st.write("\n")

    st.write("DATABASE")

    st.write(pd.DataFrame(get_data()))

    st.write("\n")
    #st.write(data)
    #st.write(f"The predicted number of bikes today is: {int(predictions)}")
    st.sidebar.header("Goharzamin Iron Ore Mine CIBB")
    st.sidebar.markdown(
    """ 
    ****   
    [Develope and Deploy by Behin Rahbord Enfejar](https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv)
    """
    )
elif  page == "Home":
        st.write("شرکت بهینه راهبرد انفجار")
        pic1 = Image.open('behin.jpg')
        st.image(pic1, use_column_width=True)
elif  page == "Prediction":
    st.sidebar.header("متغیرها را وارد کنید")

    def user_input_features():
        LF = st.sidebar.slider("میانگین عمق چال‌ها در ردیف اول", 0.00, 20.00, 0.00)
        LM = st.sidebar.slider("میانگین عمق چال‌ها در ردیف‌های میانی", 0.00, 20.00, 0.0)
        WM = st.sidebar.slider("نسبت عمق آب به عمق چال ردیف‌های میانی", 0.00, 1.00, 0.00)
        WL = st.sidebar.slider("نسبت عمق آب به عمق چال ردیف آخر", 0.00, 1.00, 0.00)
        RLW = st.sidebar.slider("نسبت طول به عرض بلوک  انفجاری", 0.00, 25.00, 0.00)

        data = {
                'LF':LF,
                'LM':LM,
                'WM':WM,
                'WL':WL,
                'RLW':RLW,
        }

        return pd.DataFrame(data, index=[0])    

    df = user_input_features()
    st.sidebar.markdown(
    """ 
    ****   
    [Download training data](https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv)
    """
    )


    # Main

    st.header("***Mine to Mill Optimization Project in Goharzamin Iron Ore Mine***")
    st.subheader(" پیش بینی کارایی خردایش ")
    st.write("\n")
    st.write("\n")
    st.write("متغیرهای ورودی کاربر:")
    st.write(df)

    with open("xgboost.pkl", "rb") as f:
        mdl = joblib.load(f)   
    predictions = mdl.predict(df)[0]

    st.write("\n")
    st.write("\n")
    st.subheader("پیش بینی بر اساس مدل : XgBoost")
    st.write(f"The predicted Fragmentation is: {(predictions)}")

    with open("xgbBU.pkl", "rb") as f:
        mdl = joblib.load(f)   
    predbu = mdl.predict(df)[0]
    st.write(f"The predicted Bulder is: {(predbu)}")
    st.write("\n")       
    with open("xgbTF.pkl", "rb") as f:
        mdl = joblib.load(f)   
    predTF = mdl.predict(df)[0]
    st.write(f"The predicted TF is: {(predTF)}")
    if predTF == "type 1":
        st.write("عدم نیاز به تسطیح؛ - نبود پاشنه؛ - نبود قوزک و ناهمواری های کف")
    elif predTF == "type 2":
        st.write('نیاز به بلدوزرکاری کم؛ - نیاز به چکش¬کاری کم؛- نبود پاشنه؛  قوزک و ناهمواری های کم در کف پله؛ - کارایی خوب بارکننده جهت بارگیری از کف پله')
    elif predTF == "type 3":
        st.write('نیاز به بلدوزرکاری کم؛ نیاز به چکش کاری متوسط؛ وجود پاشنه کوچک در برخی نقاط بلوک؛قوزک و ناهمواری¬های کم در کف؛ کارایی خوب بارکننده جهت بارگیری از کف')
    elif predTF == 'type 4':
        st.write('بلدوزرکاری متوسط؛نیاز به چکش کاری زیاد؛وجود پاشنه در برخی از نقاط؛وجود قوزک کم و  ناهمواری کم زمین؛کارایی متوسط بارکننده جهت بارگیری از کف') 
    else:
        st.write("نیاز به بلدوزرکاری متوسط؛نیاز به چکش کاری زیاد؛وجود قوزک زیاد و شرایط ناهموار زمین؛وجود پاشنه نیازمند حفاری مجدد و انفجار کارایی بد بارکننده جهت بارگیری از کف")       

    with open("xgbOV.pkl", "rb") as f:
        mdl = joblib.load(f)   
    predOV = mdl.predict(df)[0]
    st.write(f"The predicted OV is: {(predOV)}")
    if predOV == "type 1":
        st.write("وجود دیواره های صاف وبا شیب مناسب (تقریبا80)-عدم نیاز به لق گیری-محل استقرار دستگاه حفاری مناسب با ایمنی بالا -نبود ترک و شکستگی سطحی بلوک -عدم نیاز به بلدوزرکاری")
    elif predOV == "type 2":
        st.write('وجود دیواره های صاف وبا شیب مناسب (تقریبا80)- نیاز به لق گیری کم دیواره-محل استقرار دستگاه حفاری مناسب با ایمنی بالا -نبود ترک و شکستگی  در سطح بلوک - نیاز به بلدوزرکاری کم')
    elif predOV == "type 3":
        st.write('-وجود دیواره های صاف وبا شیب مناسب (تقریبا80)-نیاز به لق گیری متوسط دیواره-محل استقرار دستگاه حفاری نسبتا مناسب با ایمنی متوسط -نبود ترک و شکستگی در سطح بلوک- عدم نیاز به بلدوزرکاری')
    elif predOV == 'type 4':
        st.write('وجود دیواره های صاف وبا شیب مناسب (تقریبا80)-نیاز به لق گیری کم دیواره-محل استقرار دستگاه حفاری نسبتا مناسب با ایمنی متوسط-وجود ترک و شکستگی کم در سطح بلوک -نیاز به بلدوزرکاری کم')
    elif predOV == 'type 5':
        st.write('-وجود دیواره های صاف وبا شیب مناسب (تقریبا80)-نیاز به لق گیری متوسط دیواره-محل استقرار دستگاه حفاری نسبتا  مناسب با ایمنی متوسط -وجود ترک و شکستگی کم در سطح بلوک -نیاز به بلدوزرکاری کم')
    elif predOV == 'type 6':
        st.write('وجود دیواره های صاف وبا شیب مناسب (تقریبا80)-نیاز به لق گیری زیاد دیواره-محل استقرار دستگاه حفاری نا مناسب با ایمنی کم   -وجود ترک و شکستگی زیاد در سطح بلوک-نیاز به بلدوزرکاری زیاد')
    elif predOV == 'type 7':
        st.write("*-محل استقرار دستگاه حفاری نسبتا مناسب با ایمنی متوسط  -نبود ترک و شکستگی  در سطح بلوک  - عدم نیاز به بلدوزرکاری")       
        st.write("-وجود دیواره با شکم دادگی بدلیل شرایط بد جنس زمین - نیاز به لق گیری متوسط")       
    else:
        st.write("محل استقرار دستگاه حفاری نا مناسب ایمنی کم   -وجود ترک و شکستگی زیاد در سطح بلوک -نیاز به بلدوزرکاری زیاد")       
        st.write("وجود دیواره  با شیب ملایم (55 تا 70) بدلیل وجود گسل -نیاز به لق گیری زیاد ")       

    with open("xgbMU.pkl", "rb") as f:
        mdl = joblib.load(f)   
    predMU= mdl.predict(df)[0]
    st.write(f"The predicted MU is: {(predMU)}")

    if predMU == "type 1":
        pic1 = Image.open('1.jpg')
        st.image(pic1, use_column_width=True)
        st.write('مساحت خیلی زیاد پاک سازی؛ایمنی مناسب بارگیری؛ استقرار آسان دستگاه‌های بارکننده؛ قفل شدگی ندارد؛ کارایی خوب بارکننده کوچک در باطله و خاک؛ کارایی خوب بارکننده متوسط در سنگ آهن')
    elif predMU =="type 2":
        pic2 == Image.open('2.jpg')
        st.image(pic2, use_column_width=True)
        st.write('مساحت کم پاک سازی؛ ایمنی متوسط بارگیری؛ استقرار سخت دستگاه‌ بارکننده؛ احتمال قفل شدگی زیاد بار؛  نیاز به بلدوزر برای کم کردن ارتفاع بار (پایین نشاندن ارتفاع بار)؛ کارایی بد بارکننده‌ ها؛ ') 

    elif predMU == "type 3":
        pic3 = Image.open('3.jpg')
        st.image(pic3, use_column_width=True)
        st.write('استقرار سخت دستگاه‌ بارکننده؛ احتمال قفل¬شدگی زیاد بار؛ نیاز به بلدوزر برای کم کردن ارتفاع بار (پایین نشاندن ارتفاع بار)؛ کارایی بد بارکننده‌ ها؛- مساحت کم پاک سازی؛ ایمنی بد بارگیری؛')

    elif predMU == 'type 4':
        pic4 = Image.open('4.jpg')
        st.image(pic4, use_column_width=True)
        st.write('مساحت زیاد پاک سازی؛ ایمنی خوب بارگیری؛استقرار آسان دستگاه‌ بارکننده؛ قفل شدگی ندارد؛ کارایی خوب بارکننده‌ متوسط؛ ')
    elif predMU == 'type 5':
        pic4 = Image.open('5.jpg')
        st.image(pic4, use_column_width=True)
        st.write('مساحت متوسط پاک سازی؛ - ایمنی خوب بارگیری؛ - استقرار نسبتا آسان دستگاه‌ بارکننده؛    - احتمال قفل شدگی متوسط بار؛ - کارایی خوب بارکننده‌ متوسط؛ ')

    elif predMU == 'type 6':
        pic4 = Image.open('6.jpg')
        st.image(pic4, use_column_width=True)
        st.write('مساحت متوسط پاک سازی؛ ایمنی متوسط بارگیری؛ استقرار نسبتا آسان دستگاه‌ بارکننده؛    احتمال قفل¬شدگی متوسط بار؛ نیاز به بلدوزر برای کم کردن ارتفاع بار(پایین نشاندن ارتفاع بار) ؛کارایی متوسط بارکننده‌ متوسط؛')

    else:
        pic5 = Image.open('7.jpg')
        st.image(pic5, use_column_width=True)
        st.write('مساحت زیاد پاک سازی؛ایمنی خوب بارگیری؛استقرار آسان دستگاه‌ بارکننده؛   احتمال قفل شدگی کم بار؛کارایی خوب بارکننده‌ متوسط؛')
