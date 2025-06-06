import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load image
img = Image.open("./Grafiken/cathead.png")

# Use st_canvas with the image as background
canvas_result = st_canvas(
    background_image=img,
    stroke_width=3,
    stroke_color="red",
    drawing_mode="freedraw",
    height=img.height,
    width=img.width,
    key="canvas",
)

if canvas_result.json_data:
    st.write("You drew something!")
