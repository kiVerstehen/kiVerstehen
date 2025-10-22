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

def plot_with_new_function(w1=0.3, b1=17.0, w2=0.3, b2=17.0):
    fig, ax = plt.subplots()

    # Achsenbeschriftungen und -limits setzen
    ax.set_xlim(10, 68)
    ax.set_ylim(12, 42)
    ax.set_xlabel('Größe (cm)')
    ax.set_ylabel('Gewicht (kg)')


    # Neue Funktion definieren
    def new_function(x):
        y_cat = np.maximum(0, w1 * x + b1)
        y_dog = np.maximum(0, w2 * x + b2)
        return y_cat + y_dog
    
    # Draw decision boundary
    x_vals = np.linspace(10, 70, 100)
    y_vals = new_function(x_vals)
    ax.plot(x_vals, y_vals, '-', color='royalblue', label=f'2 Neuronen addiert')

    # Abstände berechnen und anzeigen
    total_distance_cats = 0
    total_distance_dogs = 0


    for i in range(len(cat_heights)):
        y_on_line = new_function(cat_heights[i])
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
        y_on_line = new_function(dog_heights[i])
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

    
    

    # Bereich einfärben
    
    ax.fill_between(x_vals, y_vals, 42, where=(y_vals < 42), color='navajowhite', alpha=1, label='kategorisiert als Hund')
    ax.fill_between(x_vals, y_vals, 12, color='lightblue', alpha=1, label='kategorisiert als Katze')

    ax.set_xlim(10, 68)
    ax.set_ylim(12, 42)
    ax.set_xlabel('Größe (cm)')
    ax.set_ylabel('Gewicht (kg)')
    ax.legend()

    
    st.table([["**Verlust**", f"**{total_distance_cats+total_distance_dogs:.2f}**"]])

    st.pyplot(fig)


# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1.5, 3])  

with col1:
    st.write("#")
    st.write("###")
    steigung1 = SyncedSlider("Gewicht 1", -2.0, 2.0, round(random.uniform(-2.0, 2.0),1), key_prefix="slider_s1", step=0.1)
    y_achsenabschnitt1 = SyncedSlider('Bias 1', -50, 50, random.randint(-50, 50), key_prefix="slider_y1", step=1)
    #st.divider()
    steigung2 = SyncedSlider("Gewicht 2", -2.0, 2.0, round(random.uniform(-2.0, 2.0),1), key_prefix="slider_s2", step=0.1)
    y_achsenabschnitt2 = SyncedSlider('Bias 2', -50, 50, random.randint(-50, 50), key_prefix="slider_y2", step=1)

with col2:
    plot_with_new_function(steigung1.value(), y_achsenabschnitt1.value(), steigung2.value(), y_achsenabschnitt2.value())