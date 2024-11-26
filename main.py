import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage

from pages import poxAnalysis
# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('cover.jfif')
display = np.array(display)
st.image(display)
st.title("Skin Image based Pox Analysis")
st.text("Pox Affected Or Not: To detect chances of MonkeyPox, Measles and ChickenPox")

# col1 = st.columns(1)
# col1, col2 = st.columns(2)
# col1.image(display, width = 400)
# col2.title("Data Storyteller Application")

# Add all your application here
app.add_page("Pox Analysis", poxAnalysis.app)
# app.add_page("Detect Disaster Type", detectDisaster.app)


# The main app
app.run()
