import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from joblib import load

def plotBar(data, x, y):
    return px.bar(data_frame=data, x = x, y = y)
    
def plotLine(data, x, y):
    return px.line(data_frame=data, x = x, y = y)

def plotPie(data, labels):
    return px.pie(data_frame=data, labels=labels)   

def readData():
    Video_Games = pd.read_csv('vgsales.csv')
    Video_Games.rename(columns={'Platform':'Plateform'}, inplace=True)
    Video_Games['Year'] = Video_Games['Year'].fillna(0).astype('int')
    return Video_Games

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

df = readData()

# Load the trained model
model = load('video_game_sales_prediction.joblib')

# Load the dataset for encoding purposes
data_for_encoding = pd.read_csv('vgsales.csv', index_col='Rank')

# Encode categorical variables
label_encoder_platform = LabelEncoder()
label_encoder_genre = LabelEncoder()
label_encoder_publisher = LabelEncoder()

data_for_encoding['Platform'] = label_encoder_platform.fit_transform(data_for_encoding['Platform'])
data_for_encoding['Genre'] = label_encoder_genre.fit_transform(data_for_encoding['Genre'])
data_for_encoding['Publisher'] = label_encoder_publisher.fit_transform(data_for_encoding['Publisher'])

# Reverse mapping for display
reverse_mapping_platform = dict(zip(label_encoder_platform.classes_, label_encoder_platform.transform(label_encoder_platform.classes_)))
reverse_mapping_genre = dict(zip(label_encoder_genre.classes_, label_encoder_genre.transform(label_encoder_genre.classes_)))
reverse_mapping_publisher = dict(zip(label_encoder_publisher.classes_, label_encoder_publisher.transform(label_encoder_publisher.classes_)))

# Sidebar
st.sidebar.header('Video Game Sales Prediction')

platform = st.sidebar.selectbox('Select Platform', list(reverse_mapping_platform.keys()))
year = st.sidebar.number_input('Enter Year of Release', min_value=int(data_for_encoding['Year'].min()), max_value=int(data_for_encoding['Year'].max()), value=int(data_for_encoding['Year'].mean()))
genre = st.sidebar.selectbox('Select Genre', list(reverse_mapping_genre.keys()))
publisher = st.sidebar.selectbox('Select Publisher', list(reverse_mapping_publisher.keys()))

# Encode user input
platform_encoded = label_encoder_platform.transform([platform])[0]
genre_encoded = label_encoder_genre.transform([genre])[0]
publisher_encoded = label_encoder_publisher.transform([publisher])[0]

# Make prediction
user_input = pd.DataFrame({
    'Platform': [platform_encoded],
    'Year': [year],
    'Genre': [genre_encoded],
    'Publisher': [publisher_encoded]
})

if st.sidebar.button('Predict Global Sales'):
    prediction = model.predict(user_input)[0]
    st.sidebar.success(f'Prediction: {prediction:.2f} million')


st.title("Video Games Sales Data Analysis")
st.image("bg.jpg")
st.subheader("vgsales.csv dataset")
st.dataframe(df)

st.markdown("""
- ##### Games released between 1980 to 2020
- ##### Released for all the different gaming consoles
- ##### Contains data of multiple regions and publishers""")
st.markdown("---")

st.markdown("## Sales in Various Regions")
start, end = st.slider("Select Year Range", value=[2000, 2010], min_value=1980, max_value=2020)
selRegion = st.selectbox("Select Region", ['NA_Sales', 'EU_Sales', 'JP_Sales','Other_Sales','Global_Sales'])
year_count = (i for i in range(start, end))
count_in_range = df.loc[df['Year'].isin(year_count)] 
ns = sum(count_in_range[selRegion])
st.header(round(ns))

st.markdown("---")

st.markdown("## Top Genre in Various Regions")
genregion = st.selectbox("Select Region:", ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])
data = df.groupby('Genre', as_index=False).sum().sort_values(genregion, ascending=False)

fig = plotBar(data, 'Genre', genregion)
st.plotly_chart(fig, use_container_width=True)

st.info("- Action genre appears to be quite popular in almost all regions.")

st.markdown("---")

st.markdown("## Top 10 Publishers in Various Regions")
pubregion = st.selectbox("Select a Region", ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])
data = df.groupby('Publisher', as_index=False).sum().sort_values(pubregion, ascending=False).head(10)

fig = plotBar(data,'Publisher', pubregion)
st.plotly_chart(fig, use_container_width=True)

st.info("- Nintendo and Electronic Arts are popular in all regions.")

st.markdown("---")

st.markdown('## Top 10 Video Games by Sales')
top_10_sales = st.selectbox("Select region", ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])
data = df.groupby('Name', as_index=False).sum().sort_values(top_10_sales, ascending=False).head(10)

fig = plotBar(data,'Name', top_10_sales)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown('## No. of Games Published Per Year')
data = df[df['Year'] != 0].groupby('Year', as_index=False).count()
fig = plotLine(data, 'Year', 'Publisher')
st.plotly_chart(fig, use_container_width=True)

st.info("- 2009 saw the highest number of video games released globally.")

st.markdown("---")

st.markdown("## Word Cloud for Video Game Titles")
game_titles = ' '.join(df['Name'].astype(str).tolist())
generate_word_cloud(game_titles)

st.markdown("---")

st.markdown('## Most Popular Genres Globally')
data1 = df.groupby('Genre', as_index=False).count()
fig= px.pie(data1, labels='Genre', values='Rank', names='Genre')

data3 = df[df['Year']!=0].groupby('Year', as_index=False).sum()
st.plotly_chart(fig, use_container_width=True)

st.info("- Action is the most popular genre across the globe.")

st.markdown("---")

st.markdown('## Sales (in millions) in Various Regions')
data = df[df['Year'] != 0].groupby('Year', as_index=False).count()
px.line(data, 'Year', 'Name')

fig = go.Figure()
fig.add_trace(go.Line(x = data3.Year, y = data3.NA_Sales, name="NA Sales"))
fig.add_trace(go.Line(x = data3.Year, y = data3.EU_Sales, name="EU Sales"))
fig.add_trace(go.Line(x = data3.Year, y = data3.JP_Sales, name="JP Sales"))
fig.add_trace(go.Line(x = data3.Year, y = data3.Other_Sales, name="Other Sales"))
fig.add_trace(go.Line(x = data3.Year, y = data3.Global_Sales, name="Global Sales"))
st.plotly_chart(fig, use_container_width=True)

st.info("- 2008 saw the highest of all time sales with 678.9 million dollars.")
