import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append( '../' )
from modules.streamlitsliders import SyncedSlider

def plot_with_new_function(w1=0.3, b1=17.0, w2=0.3, b2=17.0):
    fig, ax = plt.subplots()

    # Neue Funktion definieren
    def new_function(x):
        y = np.maximum(0, w1 * x + b1)
        z = np.maximum(0, w2 * x + b2)
        return y+z
    
    x_vals = np.linspace(0, 100, 100)  # Erzeuge 100 Werte zwischen 10 und 70
    y_new = new_function(x_vals)

    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 50)
    ax.plot(x_vals, y_new, '-', color='purple', label=f'2 Neuronen addiert')
    ax.plot(x_vals, w1 * x_vals + b1, '--', color='grey', label=f'{w1:.1f} * x + {b1}')
    ax.plot(x_vals, w2 * x_vals + b2, '--', color='grey', label=f'{w2:.1f} * x + {b2}')
    ax.legend()  # Legende aktualisieren

    #plt.show()
    st.pyplot(fig)



# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1.5, 3])  # Seitenverhältnis: 1 Teil Slider, 3 Teile Plot

with col1:
    steigung1 = SyncedSlider("Gewicht 1", -2.0, 2.0, round(random.uniform(-2.0, 2.0),1), key_prefix="slider_s1", step=0.1)
    y_achsenabschnitt1 = SyncedSlider('Bias 1', -50, 50, random.randint(-50, 50), key_prefix="slider_y1", step=1)
    #st.divider()
    steigung2 = SyncedSlider("Gewicht 2", -2.0, 2.0, round(random.uniform(-2.0, 2.0),1), key_prefix="slider_s2", step=0.1)
    y_achsenabschnitt2 = SyncedSlider('Bias 2', -50, 50, random.randint(-50, 50), key_prefix="slider_y2", step=1)

with col2:
    plot_with_new_function(steigung1.value(), y_achsenabschnitt1.value(), steigung2.value(), y_achsenabschnitt2.value())