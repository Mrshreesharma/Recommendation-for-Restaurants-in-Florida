


# Title: Building a Location-Based Recommendation System for Tampa City using Yelp Dataset

### Abstract:
This Project highlights the development of a location-based recommendation system for Tampa City using Yelp data. The system utilizes machine learning algorithms and streamlit framework to create an interactive web interface that allows users to explore personalized restaurant recommendations based on their location. This project leverages the vast amount of Yelp data to provide users with tailored recommendations for restaurants in Tampa City. By combining the power of machine learning algorithms and streamlit framework, we aimed to create an intuitive and user-friendly interface that enhances the dining experience of users.

### Data Preprocessing:
The project started with the collection and preprocessing of Yelp data. The dataset included information such as restaurant addresses, coordinates, review counts, stars, and categories. Preprocessing steps involved cleaning the data, handling missing values, and performing feature engineering to extract relevant information.

### Clustering and Recommendation:
To group similar restaurants together, we employed the K-means clustering algorithm. By considering the longitude and latitude coordinates of each restaurant, we assigned them to one of the clusters. This clustering step allowed us to create a recommendation system based on the user's selected location.

### Streamlit Web Interface:
Streamlit, a popular Python library for building interactive web applications, was utilized to create the web interface. The interface consists of two main tabs: "Customizable Recommendations," "Table View," and "Map View." The "Customizable Recommendations" tab provides users with the ability to filter recommendations based on review count and stars. Users can visualize the results in both table and map views. 
Here is the link for our web-app https://reccomendationsystem3.onrender.com/


![image](https://github.com/Mrshreesharma/Recommendation-for-Restaurants-in-Florida/assets/60129674/1fa13194-5713-404a-8118-14ef19558613)


### Deployment and Hosting:
To make the recommendation system accessible to a wider audience, we deployed the web application using Render https://render.com/ . This deployment allowed users to access the system through a web browser without any local installations or dependencies.

### Conclusion:
In conclusion, we successfully developed a location-based recommendation system for Tampa City using Yelp data. The system provides personalized restaurant recommendations based on the user's selected location and preferences. The integration of streamlit framework and machine learning algorithms enhances the user experience by providing an interactive and visually appealing interface. With its deployment on Render, the system is readily available to users, making it a valuable tool for exploring and discovering new dining experiences in Tampa City.
