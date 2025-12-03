import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

import sys
sys.path.append( '../' )
from modules.streamlitsliders import SyncedSlider

# Daten
cat_heights = [20, 24, 31, 44, 50]
cat_weights = [30, 20, 15, 25, 30]

dog_heights = [18, 30, 35, 60, 45]
dog_weights = [37, 27, 35, 30, 38]

# Lade Bilder EINMAL als Arrays (schnell und wiederverwendbar)
cat_img_arr = mpimg.imread('../Grafiken/cathead.png')
cat_img_grey_arr = mpimg.imread('../Grafiken/cathead_grey.png')
dog_img_arr = mpimg.imread('../Grafiken/doghead.png')
dog_img_grey_arr = mpimg.imread('../Grafiken/doghead_grey.png')

def get_image_from_array(arr, zoom=0.2):
    return OffsetImage(arr, zoom=zoom)

def plot_counting(steigung=1.0, y_achsenabschnitt=0.0):
    fig, ax = plt.subplots()

    # Abstände berechnen und anzeigen
    total_distance_cats = 0
    total_distance_dogs = 0
    
    for i in range(len(cat_heights)):
        y_on_line = steigung * cat_heights[i] + y_achsenabschnitt
        if cat_weights[i]>y_on_line:
            image_data = cat_img_grey_arr
            distance = cat_weights[i] - y_on_line
            total_distance_cats += distance
            # Linie für distanz zeichnen
            ax.plot([cat_heights[i], cat_heights[i]], [y_on_line, cat_weights[i]], 'b-')
        else: 
            image_data = cat_img_arr
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (cat_heights[i], cat_weights[i]), frameon=False)
        ax.add_artist(ab)
        #if image_data is cat_img_grey_arr:
        #    cat_count += 1

    
    for i in range(len(dog_heights)):
        y_on_line = steigung*dog_heights[i]+y_achsenabschnitt
        if dog_weights[i]<y_on_line:
            image_data = dog_img_grey_arr 
            distance = y_on_line - dog_weights[i]
            total_distance_dogs += distance
            # Linie für distanz zeichnen
            ax.plot([dog_heights[i], dog_heights[i]], [dog_weights[i], y_on_line], 'g-')
        else:
            image_data = dog_img_arr
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (dog_heights[i], dog_weights[i]), frameon=False)
        ax.add_artist(ab)
        #if image_data is dog_img_grey_arr:
        #    dog_count += 1

    # Draw decision boundary
    x_vals = np.linspace(10, 70, 100)
    y_vals = y_achsenabschnitt + steigung * x_vals
    ax.plot(x_vals, y_vals, color='royalblue', label=f'line: y = {steigung:.1f}x + {y_achsenabschnitt:.2f}')

    # Bereich einfärben
    
    ax.fill_between(x_vals, y_vals, 42, where=(y_vals < 42), color='navajowhite', alpha=1, label='categorized as a dog')
    ax.fill_between(x_vals, y_vals, 12, color='lightblue', alpha=1, label='categorized as a cat')

    ax.set_xlim(10, 68)
    ax.set_ylim(12, 42)
    ax.set_xlabel('size (cm)')
    ax.set_ylabel('weight (kg)')
    ax.legend()

    
    st.table([["weight differences of falsly categorized cats",f"{total_distance_cats:.2f}"],["weight differences of falsly categorized dogs",f"{total_distance_dogs:.2f}"],["**loss**", f"**{total_distance_cats+total_distance_dogs:.2f}**"]])

    st.pyplot(fig)



# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1.5, 3])  

with col1:
    st.write("#")
    st.write("###")

    
    steigung = SyncedSlider("slope", -1.0, 1.0, round(random.uniform(-1.0, 1.0),1), key_prefix="slider_s", step=0.1)

    y_achsenabschnitt = SyncedSlider('y-intercept', 10, 50, random.randint(10, 50), key_prefix="slider_y", step=1)

with col2:
    plot_counting(steigung.value(), y_achsenabschnitt.value())
    

