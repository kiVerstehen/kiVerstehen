import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_with_new_function(w1=0.3, b1=17.0, w2=0.3, b2=17.0):
    fig, ax = plt.subplots()

    # Neue Funktion definieren
    def new_function(x):
        y_cat = w1 * x + b1
        y_dog = w2 * x + b2
        return y_cat + y_dog
    
    x_vals = np.linspace(10, 70, 100)  # Erzeuge 100 Werte zwischen 10 und 70
    y_new = new_function(x_vals)
    ax.plot(x_vals, y_new, '-', color='purple', label=f'{w1} * x + {b1} + {w2} * x + {b2}')
    ax.plot(x_vals, w1 * x_vals + b1, '--', color='grey', label=f'{w1} * x + {b1}')
    ax.plot(x_vals, w2 * x_vals + b2, '--', color='grey', label=f'{w2} * x + {b2}')
    ax.legend()  # Legende aktualisieren

    plt.show()
    st.pyplot(fig)


# Zufälliger Wert für den Slider
if 'randomS1' not in st.session_state:
    st.session_state.randomS1 = round(random.uniform(-2, 2),2)
if 'randomS2' not in st.session_state:
    st.session_state.randomS2 = round(random.uniform(-2, 2),2)

if 'randomY1' not in st.session_state:
    st.session_state.randomY1 = round(random.uniform(-50, 50),2)
if 'randomY2' not in st.session_state:
    st.session_state.randomY2 = round(random.uniform(-50, 50),2)


# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1, 3])  # Seitenverhältnis: 1 Teil Slider, 3 Teile Plot

with col1:
    

    steigung1 = st.slider('Steigung 1', min_value=-2.0, max_value=2.0, step=0.05, value=st.session_state.randomS1)
    y_achsenabschnitt1 = st.slider(
        'y-Achsenabschnitt 1', 
        min_value=-50.0, 
        max_value=50.0, 
        step=0.05, 
        value=st.session_state.randomY1
    )
    #st.divider()
    steigung2 = st.slider('Steigung 2', min_value=-2.0, max_value=2.0, step=0.05, value=st.session_state.randomS2)
    y_achsenabschnitt2 = st.slider(
        'y-Achsenabschnitt 2', 
        min_value=-50.0, 
        max_value=50.0, 
        step=0.05, 
        value=st.session_state.randomY2
    )

with col2:
    plot_with_new_function(steigung1, y_achsenabschnitt1, steigung2, y_achsenabschnitt2)