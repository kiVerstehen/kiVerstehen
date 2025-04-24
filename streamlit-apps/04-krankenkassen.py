import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random



# Daten
hoffset = 140
woffset = 50
cat_heights = [20, 24, 31, 44, 50]  # Größe zwischen 22 und 30 cm
cat_weights = [30, 20, 15, 25, 30]  # Gewicht zwischen 4 und 8 kg

cat_heights = [height+hoffset for height in cat_heights] 
cat_weights = [weight+woffset for weight in cat_weights] 

dog_heights = [18, 30, 35, 60, 45]  # Größe zwischen 45 und 65 cm
dog_weights = [37, 27, 35, 30, 38]  # Gewicht zwischen 15 und 35 kg

dog_heights = [height+hoffset for height in dog_heights] 
dog_weights = [weight+woffset for weight in dog_weights] 


# Lade Bilder EINMAL als Arrays (schnell und wiederverwendbar)
cat_img_arr=[]
cat_img_grey_arr=[]
dog_img_arr=[]
dog_img_grey_arr=[]

for i in range(len(cat_heights)):
    cat_img_arr.append(mpimg.imread('../Grafiken/person'+str(i+1)+'a.png'))
    cat_img_grey_arr.append(mpimg.imread('../Grafiken/person'+str(i+1)+'ab.png'))
    dog_img_arr.append(mpimg.imread('../Grafiken/person'+str(i+6)+'a_krank.png'))
    dog_img_grey_arr.append(mpimg.imread('../Grafiken/person'+str(i+6)+'ab_krank.png'))

def get_image_from_array(arr, zoom=0.07):
    return OffsetImage(arr, zoom=zoom)

def plot_counting(steigung=1.0, y_achsenabschnitt=0.0):
    fig, ax = plt.subplots()

    # Abstände berechnen und anzeigen
    total_distance_cats = 0
    total_distance_dogs = 0
    
    for i in range(len(cat_heights)):
        y_on_line = steigung * (cat_heights[i]-150) + y_achsenabschnitt
        if cat_weights[i]>y_on_line:
            image_data = cat_img_grey_arr[i]
            distance = cat_weights[i] - y_on_line
            total_distance_cats += distance
            # Linie für distanz zeichnen
            ax.plot([cat_heights[i], cat_heights[i]], [y_on_line, cat_weights[i]], 'b-')
        else: 
            image_data = cat_img_arr[i]
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (cat_heights[i], cat_weights[i]), frameon=False)
        ax.add_artist(ab)
        #if image_data is cat_img_grey_arr:
        #    cat_count += 1

    
    for i in range(len(dog_heights)):
        y_on_line = steigung * (dog_heights[i]-150) + y_achsenabschnitt
        if dog_weights[i]<y_on_line:
            image_data = dog_img_grey_arr[i]
            distance = y_on_line - dog_weights[i]
            total_distance_dogs += distance
            # Linie für distanz zeichnen
            ax.plot([dog_heights[i], dog_heights[i]], [dog_weights[i], y_on_line], 'g-')
        else:
            image_data = dog_img_arr[i]
        image = get_image_from_array(image_data)
        ab = AnnotationBbox(image, (dog_heights[i], dog_weights[i]), frameon=False)
        ax.add_artist(ab)


    # Gerade hinzufügen
    x_vals = np.linspace(148, 208, 100)  # Erzeuge 100 Werte zwischen 10 und 70
    y_vals = y_achsenabschnitt + steigung * (x_vals-150)  # Berechne die y-Werte basierend auf der Geradengleichung
    ax.plot(x_vals, y_vals, color='royalblue', label=f'Gerade: y = {steigung:.2f} * (x-150) + {y_achsenabschnitt:.2f}')

    # Bereich einfärben
    
    ax.fill_between(x_vals, y_vals, 92, where=(y_vals < 92), color='navajowhite', alpha=1, label='kategorisiert als krank')
    ax.fill_between(x_vals, y_vals, 62, color='lightblue', alpha=1, label='kategorisiert als gesund')

    # Achsenbeschriftungen und -limits setzen
    ax.set_xlim(148, 208)
    ax.set_ylim(62, 92)
    ax.set_xlabel('Größe (cm)')
    ax.set_ylabel('Gewicht (kg)')
    ax.legend()

    
    st.table([["Gewichtsdifferenzen der falsch kategorisierten Gesunden",f"{total_distance_cats:.2f}"],["Gewichtsdifferenzen der falsch kategorisierten Kranken",f"{total_distance_dogs:.2f}"],["**Verlust**", f"**{total_distance_cats+total_distance_dogs:.2f}**"]])

    st.pyplot(fig)


# Zufälliger Wert für den Slider
if 'randomS' not in st.session_state:
    st.session_state.randomS = random.uniform(-1, 1)

if 'randomY' not in st.session_state:
    st.session_state.randomY = random.uniform(10, 150)


# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1, 3])  # Seitenverhältnis: 1 Teil Slider, 3 Teile Plot

with col1:
    st.write("#")
    st.write("###")

    steigung = st.slider('Steigung', min_value=-1.0, max_value=1.0, step=0.05, value=st.session_state.randomS)
    y_achsenabschnitt = st.slider(
        'y-Achsenabschnitt', 
        min_value=10.0, 
        max_value=150.0, 
        step=0.05, 
        value=st.session_state.randomY
    )

with col2:
    plot_counting(steigung, y_achsenabschnitt)
    

