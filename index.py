# 01.
import streamlit as st


# 02.
import streamlit as st

st.title('This a Title')
st.header('This a Header')

# 03.
import streamlit as st

st.title("This a Title.")

# 04.
import streamlit as st

st.subheader("This a Subheader.")

# 05.
import streamlit as st

st.write("This is text.")

# 06.
import streamlit as st

st.caption("This a Caption.")

# 07.
import streamlit as st

with open("./contents/README.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()
st.markdown(markdown_text, unsafe_allow_html =True)

# 08.
import streamlit as st

st.title("This a Title.")
st.header("This a Header.")
st.subheader("This a Subheader.")
st.write("This is text.")
st.caption("This a Caption.")

with open("./contents/README.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()
st.markdown(markdown_text, unsafe_allow_html =True)

# 09.
import streamlit as st
import pandas as pd

# 예제 데이터 생성
data = {
    '이름': ['홍길동', '김철수', '이영희', '박영수'],
    '나이': [30, 25, 28, 32],
    '성별': ['남성', '남성', '여성', '남성']
}

# 데이터 프레임 생성
df = pd.DataFrame(data)

# 데이터 프레임 출력
st.write(df)
st.dataframe(df, height=200)
st.table(df)
st.markdown(df.to_markdown())

# 10.
import streamlit as st
import pandas as pd

# 데이터 준비
df = pd.read_csv("data/titanic.csv")
df = df[["Age", "Survived"]]
chart_df = df.groupby(["Age"]).sum()
chart_df["Age"] = chart_df.index

st.line_chart(chart_df, x="Age", y=["Survived"])

# 11.
import streamlit as st
import pandas as pd

# 데이터 준비
df = pd.read_csv("data/titanic.csv")
df = df[["Age", "Survived"]]
chart_df = df.groupby(["Age"]).sum()
chart_df["Age"] = chart_df.index

st.bar_chart(chart_df, x="Age", y=["Survived"])

# 12.
import streamlit as st
import pandas as pd

# 데이터 준비
df = pd.read_csv("data/titanic.csv")
df = df[["Age", "Survived"]]
chart_df = df.groupby(["Age"]).sum()
chart_df["Age"] = chart_df.index

st.area_chart(chart_df, x="Age", y=["Survived"])

# 13.
import streamlit as st
import pandas as pd

html = """
<a style='background:yellow'>This text has a yellow background</a>
"""
st.header("Without unsafe_allow_html=True")
st.markdown(html)

# 14.
import streamlit as st
import pandas as pd

user_text = st.text_input("Input some text here")
st.write(user_text)

# 15.
import streamlit as st
import pandas as pd

default_text = st.text_area("Input some text here", "default text")
st.write(default_text)


# 16.
import streamlit as st
import pandas as pd

default_text = st.text_area("Input some text here", "default text")
st.write(default_text)

# 17.
import streamlit as st
import pandas as pd

user_number = st.number_input("Input Number",
                            min_value=1,
                            max_value=10,
                            value=5,
                            step=1)
st.write(user_number)

# 18.
import streamlit as st
import pandas as pd

slider_number = st.slider("Select your Number",
                            min_value=1,
                            max_value=10,
                            value=5,
                            step=1)
st.write(slider_number)

# 19.
import streamlit as st
import pandas as pd

user_date = st.date_input("Select your Date",
                            value = datetime.date(2000, 6, 12),
                            min_value = datetime.date(2000, 1, 12),
                            max_value = datetime.date(2001, 1, 12)
                            )

st.write(user_date)

# 20.
import streamlit as st
import pandas as pd

user_time = st.time_input("Select your Time",
                            value = datetime.time(6, 12),
                            )
st.write(user_time)

# 21.
import streamlit as st
import pandas as pd

checked = st.checkbox("Select this checkbox")
st.write(f"Current state of checkbox: {checked}")

# 22.
import streamlit as st
import pandas as pd

state = st.button("Click to Change current state")
st.write(f"Button has been pressed: {state}")

# 23.
import streamlit as st
import pandas as pd

options = ["Red", "Blue", "Yellow"]
radio_selection = st.radio("Select Color", options)
st.write(f"Color selected is {radio_selection}")

# 24.
import streamlit as st
import pandas as pd

options = ["Red", "Blue", "Yellow"]
selectbox_selection = st.selectbox("Select Color", options)
st.write(f"Color selected is {selectbox_selection}")

# 25.
import streamlit as st
import pandas as pd

options = ["Red", "Blue", "Yellow"]
multiselect_selection = st.multiselect("Select Color", options)
st.write(f"Color selected is {multiselect_selection}")

# 26.
import streamlit as st
import datetime

user_text = st.text_input("Input some text here")
st.write(user_text)

default_text = st.text_area("Input some text here", "default text")
st.write(default_text)

user_number = st.number_input("Input Number",
                            min_value=1,
                            max_value=10,
                            value=5,
                            step=1)
st.write(user_number)

slider_number = st.slider("Select your Number",
                            min_value=1,
                            max_value=10,
                            value=5,
                            step=1)
st.write(slider_number)

user_date = st.date_input("Select your Date",
                            value = datetime.date(2000, 6, 12),
                            min_value = datetime.date(2000, 1, 12),
                            max_value = datetime.date(2001, 1, 12)
                            )

st.write(user_date)

user_time = st.time_input("Select your Time",
                            value = datetime.time(6, 12),
                            )
st.write(user_time)

checked = st.checkbox("Select this checkbox")
st.write(f"Current state of checkbox: {checked}")

state = st.button("Click to Change current state")
st.write(f"Button has been pressed: {state}")

options = ["Red", "Blue", "Yellow"]
radio_selection = st.radio("Select Color", options)
st.write(f"Color selected is {radio_selection}")

options = ["Red", "Blue", "Yellow"]
selectbox_selection = st.selectbox("Select Color", options)
st.write(f"Color selected is {selectbox_selection}")

options = ["Red", "Blue", "Yellow"]
multiselect_selection = st.multiselect("Select Color", options)
st.write(f"Color selected is {multiselect_selection}")


# 27.
import streamlit as st

st.sidebar.header("Sidebar Header")

# 28.
import streamlit as st

st.sidebar.header("Sidebar Header")

st.header("Columns")
cols = st.columns(2)
cols[0].write("Column 1")
cols[1].write("Column 2")


# 29.
import streamlit as st
import pandas as pd

st.sidebar.header("Sidebar Header")

st.header("Columns")
cols = st.columns(2)
cols[0].write("Column 1")
cols[1].write("Column 2")

expander = st.expander("This is an Expander")
expander.write("This is some text in an expander...")

# 30.
import streamlit as st
import pandas as pd

st.sidebar.header("Sidebar Header")

st.header("Columns")
cols = st.columns(2)
cols[0].write("Column 1")
cols[1].write("Column 2")

expander = st.expander("This is an Expander")
expander.write("This is some text in an expander...")

st.header("Container")
container = st.container()
container.write("This is some text inside a container...")

tabs = st.tabs(["Tab 1", "Tab 2"])
for i, tab in enumerate(tabs):
    tabs[i].write(f"Tab {i+1}")

# 31.
import streamlit as st

st.sidebar.header("Sidebar Header")

st.header("Columns")
cols = st.columns(2)
cols[0].write("Column 1")
cols[1].write("Column 2")

expander = st.expander("This is an Expander")
expander.write("This is some text in an expander...")

st.header("Container")
container = st.container()
container.write("This is some text inside a container...")

tabs = st.tabs(["Tab 1", "Tab 2"])
for i, tab in enumerate(tabs):
    tabs[i].write(f"Tab {i+1}")


st.header("Empty")

empty = st.empty()
items = ["Tom", "Fred", "Stephanie"]
for item in items:
    empty.write(item)


# 32.
import streamlit as st

if "prev_word_count" not in st.session_state:
    st.session_state["prev_word_count"] = 5

text = st.text_area("Paste text here to get word count.", "This is some default text.")
word_count = len(text.split())
change = word_count - st.session_state.prev_word_count
st.metric("Word Count", word_count, change)
st.session_state.prev_word_count = word_count


# 33.
import streamlit as st
import pandas as pd

# Cache our data
@st.cache_data()
def load_df():
    df = pd.read_csv("./data/titanic.csv")

    # 생존 여부
    survival_options = df.Survived.unique()
    # 객실
    p_class_options = df.Pclass.unique()
    # 성별
    sex_options = df.Sex.unique()
    # 출발
    embark_options = df.Embarked.unique()

    # 요금
    min_fare = df.Fare.min()
    max_fare = df.Fare.max()

    # 나이
    min_age = df.Age.min()
    max_age = df.Age.max()

    return df, survival_options, p_class_options, sex_options, embark_options, min_fare, max_fare, min_age, max_age

def check_rows(column, options):
    return res.loc[res[column].isin(options)]

st.title("Demo DataFrame Query App")

df, survival_options, p_class_options, sex_options, embark_options, min_fare, max_fare, min_age, max_age = load_df()
res = df

name_query = st.text_input("String match for Name")

cols = st.columns(4)
survival = cols[0].multiselect("Survived", survival_options)
p_class = cols[1].multiselect("Passenger Class", p_class_options)
sex = cols[2].multiselect("Sex", sex_options)
embark = cols[3].multiselect("Embarked", embark_options)

range_cols = st.columns(3)
min_fare_range, max_fare_range = range_cols[0].slider("Lowest Fare", float(min_fare), float(max_fare),
                                        [float(min_fare), float(max_fare)])
min_age_range, max_age_range = range_cols[2].slider("Lowest Age", float(min_age), float(max_age),
                                        [float(min_age), float(max_age)])

if name_query != "":
    res = res.loc[res.Name.str.contains(name_query)]

if survival:
    res = check_rows("Survived", survival)
if p_class:
    res = check_rows("Pclass", p_class)
if sex:
    res = check_rows("Sex", sex)
if embark:
    res = check_rows("Embarked", embark)
if range_cols[0].checkbox("Use Fare Range"):
    res = res.loc[(res.Fare > min_fare_range) & (res.Age < max_fare_range)]
if range_cols[2].checkbox("Use Age Range"):
    res = res.loc[(res.Age > min_age_range) & (res.Age < max_age_range)]
removal_columns = st.multiselect("Select Columns to Remove", df.columns.tolist())
for column in removal_columns:
    res = res.drop(column, axis=1)
st.write(res)

# 34.
import platform
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Linux':
    rc('font', family='NanumGothic')

# 스트림릿 앱 생성
st.title("데이터 프로파일링 실습")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("데이터 파일 업로드", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 업로드한 파일을 DataFrame으로 변환
    df = pd.read_csv(uploaded_file)  # 엑셀 파일일 경우 pd.read_excel 사용
    
    # 데이터 프로파일링
    st.header("데이터 미리보기")
    st.write(df.head())

    st.header("기본 정보")
    st.write("행 수:", df.shape[0])
    st.write("열 수:", df.shape[1])

    st.header("누락된 값")
    missing_data = df.isnull().sum()
    st.write(missing_data)

    st.header("중복된 행 수")
    duplicated_rows = df.duplicated().sum()
    st.write(duplicated_rows)

    st.header("수치형 데이터 기술 통계량")
    numerical_stats = df.describe()
    st.write(numerical_stats)

    st.header("이상치 탐지 (상자 그림)")
    plt.figure(figsize=(10, 6))
    plt.boxplot(df.select_dtypes(include=['number']).values)
    plt.xticks(range(1, len(df.columns) + 1), df.columns, rotation=45)
    plt.title("Outlier detection (box plot)")
    st.pyplot(plt)

    st.header("데이터 분포 시각화")
    column_to_plot = st.selectbox("열 선택", df.columns)
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_to_plot], bins=20, edgecolor='k')
    plt.xlabel(column_to_plot)
    plt.ylabel("빈도")
    plt.title(f"{column_to_plot} Data Distribution")
    st.pyplot(plt)

# 35.
import streamlit as st

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

# 36.
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

# 아이리스 데이터셋 불러오기
@st.cache_data
def load_data():
# GitHub에서 아이리스 데이터 다운로드
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_df = pd.read_csv(url)
    return iris_df

iris_data = load_data()

# 스트림릿 앱 제목 설정
st.title('아이리스 데이터 시각화')

# 데이터프레임 출력
st.subheader('아이리스 데이터셋')
st.write(iris_data)

# 품종별 특성 분포 시각화
st.subheader('품종별 특성 분포')
for feature in iris_data.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=iris_data)
    plt.title(f'{feature} Distribution')
    plt.xlabel('species')
    plt.ylabel(feature)
    st.pyplot()

# 특성 간 상관 관계 시각화
st.subheader('특성 간 상관 관계')
correlation_matrix = iris_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# 품종별 특성 산점도 시각화
st.subheader('품종별 특성 산점도')
sns.pairplot(iris_data, hue='species', diag_kind='kde')
st.pyplot()

# 37.
# 스트림릿 앱에서 Matplotlib의 그림(figure)을 스트림릿에 전달할 때 발생하는 PyplotGlobalUseWarning 경고를 방지하려면 아래와 같이 코드를 수정할 수 있습니다.
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 아이리스 데이터셋 불러오기
# 아이리스 데이터셋 불러오기
@st.cache_data
def load_data():
# GitHub에서 아이리스 데이터 다운로드
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_df = pd.read_csv(url)
    return iris_df


iris_data = load_data()

# 스트림릿 앱 제목 설정
st.title('아이리스 데이터 시각화')

# 데이터프레임 출력
st.subheader('아이리스 데이터셋')
st.write(iris_data)

# 품종별 특성 분포 시각화
st.subheader('품종별 특성 분포')
for feature in iris_data.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=iris_data)
    plt.title(f'{feature} Distribution')
    plt.xlabel('species')
    plt.ylabel(feature)
    st.pyplot(plt)

# 특성 간 상관 관계 시각화
st.subheader('특성 간 상관 관계')
correlation_matrix = iris_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt)

# 품종별 특성 산점도 시각화
st.subheader('품종별 특성 산점도')
sns.pairplot(iris_data, hue='species', diag_kind='kde')
sns.pairplot(iris_data, hue='species', diag_kind='kde')
st.pyplot(plt)
