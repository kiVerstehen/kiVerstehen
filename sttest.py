
from modules.streamlitsliders import SyncedSlider
import streamlit as st

st.title("Multi-Slider Example")

slider1 = SyncedSlider("Slider A", 0, 100, 25, key_prefix="slider_a", step=1)
slider2 = SyncedSlider("Slider B", 0, 200, 50, key_prefix="slider_b", step=1)
slider3 = SyncedSlider("Slider C", -50, 50, 0, key_prefix="slider_c", step=1)

st.write("### Values:")
st.write(f"Slider A: {slider1.value()}")
st.write(f"Slider B: {slider2.value()}")
st.write(f"Slider C: {slider3.value()}")


