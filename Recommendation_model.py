#importing librarires
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', 50)

# Importing Plotly Packages

import plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px

# Importing sklearn Packages

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Importing scipy Packages
#from scipy.sparse.linalg import svds
#from haversine import haversine, Unit

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly_express as px





# Importing sklearn Packages

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Yelp data into a DataFrame
Final_data = pd.read_csv("Business_tampa.csv")
Final_data.drop(['Unnamed: 0'], axis=1, inplace = True)
Final_data = Final_data.dropna(subset=['address'])
location = Final_data[['longitude','latitude']]
distortions = []
K = range(1,15)
for k in K:
    kmeansModel = KMeans(n_clusters=k)
    kmeansModel = kmeansModel.fit(location)
    distortions.append(kmeansModel.inertia_)
kmeans = KMeans(n_clusters = 12, init = 'k-means++')
kmeans.fit(location)
y = kmeans.labels_

user_list = Final_data[['address','longitude','latitude']]

Final_data['cluster'] = kmeans.predict(Final_data[['longitude','latitude']])
nearby_best = Final_data.sort_values(by=['stars', 'review_count'], ascending=False)

def location_based_recommendation(df, latitude, longitude):
    """Predict the cluster for longitude and latitude provided"""
    cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    print("This restaurant belongs to cluster:", cluster)
  
    """Get the best restaurant in this cluster along with the relevant information for a user to make a decision"""
    return df[df['cluster']==cluster].iloc[0:10][['name', 'latitude','longitude','categories','stars', 'review_count','cluster']]

# Create a sidebar with two tabs: Table and Map
st.sidebar.title("Location-Based Recommendation system")
st.sidebar.write("Here you can select the type of graph you want to view!")
tabs = ["Customizable recommendations","Table View", "Map View"]
selected_tab = st.sidebar.radio("", tabs)

if selected_tab == "Customizable recommendations":
    st.title("About the Recommendation sytem")
    st.write("This location-based recommendation system is a powerful tool that harnesses the vast amount of Yelp data to provide personalized recommendations to users based on their location. With a sleek and intuitive interface, users can easily select their address and receive a tailored list of the top ten restaurants in their vicinity. Whether viewing the results in a table format or on an interactive map powered by Mapbox, this system provides an engaging and user-friendly experience. Powered by cutting-edge machine learning algorithms, this recommendation system is a must-have for foodies and restaurant enthusiasts looking to discover new and exciting dining options in Florida.")
    filtered_df = Final_data

    # Filters
    review_count_slider = st.slider("Filter by review count", 0, 200, 0, 20)
    stars_slider = st.slider("Filter by stars", 1.0, 5.0, 1.0, 0.1)
    #categories_input = st.selectbox("Filter by categories", filtered_df['categories'])

    if review_count_slider > 0:
        filtered_df = filtered_df[filtered_df["review_count"] >= review_count_slider]
    if stars_slider > 1.0:
        filtered_df = filtered_df[filtered_df["stars"] >= stars_slider]
    



    st.write("Showing results for filters:")
    st.write("- Review count >= ", review_count_slider)
    st.write("- Stars >= ", stars_slider)
    


    st.write("Total number of results:", len(filtered_df))

    fig = px.scatter_mapbox(
        filtered_df, 
        lat="latitude", 
        lon="longitude", 
        hover_name="name", 
        hover_data=["stars", "review_count", "categories"],
        zoom=10,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

# Show Table tab
if selected_tab == "Table View":
    st.title('Table view')
    st.write("Your Top 10 recommendations in a table format for the selected location.")
    # Create a selectbox to select the address from the user_list dataframe
    selected_address = st.selectbox('Select an address', user_list['address'])

    # Get the longitude and latitude corresponding to the selected address
    longitude = user_list.loc[user_list['address'] == selected_address, 'longitude'].values[0]
    latitude = user_list.loc[user_list['address'] == selected_address, 'latitude'].values[0]

    # Call the location_based_recommendation function with the selected longitude and latitude
    result = location_based_recommendation(nearby_best, latitude, longitude)

    st.table(result)

# Show Map tab
elif selected_tab == "Map View":
    st.title('Map view')
    st.write("Your Top 10 recommendations in a Visual format for the selected location.")
    # Create a selectbox to select the address from the user_list dataframe
    selected_address = st.selectbox('Select an address', user_list['address'])

    # Get the longitude and latitude corresponding to the selected address
    longitude = user_list.loc[user_list['address'] == selected_address, 'longitude'].values[0]
    latitude = user_list.loc[user_list['address'] == selected_address, 'latitude'].values[0]

    # Call the location_based_recommendation function with the selected longitude and latitude
    result = location_based_recommendation(nearby_best, latitude, longitude)

    # Show Map using plotly express
    fig = px.scatter_mapbox(result, lat="latitude", lon="longitude", hover_name="name", hover_data=["categories", "stars", "review_count"],
                            color_discrete_sequence=["red"], zoom=10, height=600)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)
