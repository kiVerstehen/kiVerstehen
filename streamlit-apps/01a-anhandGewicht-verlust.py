import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import random
from IPython.display import clear_output

# Daten
cat_heights = [20, 24, 31, 44, 50]
cat_weights = [30, 20, 15, 25, 30]

dog_heights = [18, 30, 35, 60, 45]
dog_weights = [37, 27, 35, 30, 38]

#Load images ONCE as arrays (fast and reusable)
cat_img_arr = mpimg.imread('../Grafiken/cathead.png')
cat_img_grey_arr = mpimg.imread('../Grafiken/cathead_grey.png')
dog_img_arr = mpimg.imread('../Grafiken/doghead.png')
dog_img_grey_arr = mpimg.imread('../Grafiken/doghead_grey.png')

# Abstände berechnen und anzeigen
total_distance_cats = 0
total_distance_dogs = 0

def calc_verlust(x):
    cat_heights = [20, 24, 31, 44, 50]
    cat_weights = [30, 20, 15, 25, 30]

    dog_heights = [18, 30, 35, 60, 45]
    dog_weights = [37, 27, 35, 30, 38]

    total_distance_cats = 0
    total_distance_dogs = 0
    
    for i in range(len(cat_heights)):
        if cat_weights[i]> x:
            total_distance_cats += 1
        
    for i in range(len(dog_heights)):
        if dog_weights[i]<x:
            total_distance_dogs += 1
            
    return total_distance_cats+total_distance_dogs

def get_image_from_array(arr, zoom=0.2):
    return OffsetImage(arr, zoom=zoom)

def plot_counting(y_achsenabschnitt=0.0):
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    steigung=0
    # Abstände berechnen und anzeigen
    total_distance_cats = 0
    total_distance_dogs = 0
    
    for i in range(len(cat_heights)):
        y_on_line = steigung * cat_heights[i] + y_achsenabschnitt
        if cat_weights[i]>y_on_line:
            image_data = cat_img_grey_arr
            total_distance_cats += 1
            # Linie für distanz zeichnen
            ax[0].plot([cat_heights[i], cat_heights[i]], [y_on_line, cat_weights[i]], 'b-')
        else: 
            image_data = cat_img_arr
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (cat_heights[i], cat_weights[i]), frameon=False)
        ax[0].add_artist(ab)
        
    for i in range(len(dog_heights)):
        y_on_line = steigung*dog_heights[i]+y_achsenabschnitt
        if dog_weights[i]<y_on_line:
            image_data = dog_img_grey_arr 
            total_distance_dogs += 1
            # Linie für distanz zeichnen
            ax[0].plot([dog_heights[i], dog_heights[i]], [dog_weights[i], y_on_line], 'g-')
        else:
            image_data = dog_img_arr
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (dog_heights[i], dog_weights[i]), frameon=False)
        ax[0].add_artist(ab)

    # Draw decision boundary
    x_vals = np.linspace(10, 70, 100)
    y_vals = y_achsenabschnitt + 0 * x_vals
    ax[0].plot(x_vals, y_vals, color='royalblue', label=f'Grenze bei {y_achsenabschnitt:.2f} kg')

    # Bereich einfärben
    ax[0].fill_between(x_vals, y_vals, 12, color='lightblue', alpha=1, label='kategorisiert als Katze')
    ax[0].fill_between(x_vals, y_vals, 42, where=(y_vals < 42), color='navajowhite', alpha=1, label='kategorisiert als Hund')

    ax[0].set_xlim(10, 68)
    ax[0].set_ylim(12, 42)
    ax[0].set_xlabel('Größe (cm)')
    ax[0].set_ylabel('Gewicht (kg)')
    ax[0].legend()

    # Vectorize the function to handle arrays
    vectorized_verlust = np.vectorize(calc_verlust)
    
    x2_vals = np.linspace(10,50,100)
    y2_vals = vectorized_verlust(x2_vals)

    ax[1].set_xlabel('Gewichtsgrenze (kg)')
    ax[1].set_ylabel('Verlust')
    ax[1].plot(x2_vals, y2_vals, color="royalblue")
    ax[1].plot(y_achsenabschnitt, calc_verlust(y_achsenabschnitt), 'ro') 

    st.pyplot(fig)

# Zufälliger Wert für den Slider
if 'randomY' not in st.session_state:
    st.session_state.randomY = random.uniform(10, 50)

# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1, 3])  # Seitenverhältnis: 1 Teil Slider, 3 Teile Plot

with col1:
    y_achsenabschnitt = st.slider(
        'Grenze für Gewicht (kg)', 
        min_value=10.0, 
        max_value=50.0, 
        step=0.05, 
        value=st.session_state.randomY
    )

with col2:
    plot_counting(y_achsenabschnitt)

