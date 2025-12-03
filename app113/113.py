import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go

import sys
sys.path.append( '../' )
from modules.streamlitsliders import SyncedSlider

def plot_with_new_function(w1=0.3, b1=17.0, w2=0.3):
    #fig, ax = plt.subplots()

    # Neue Funktion definieren
    def new_function(x,y):
        z = np.maximum(0, (w1 * x + w2 * y + b1))
        return z
    
    x_vals = np.linspace(0, 3, 100) 
    y_vals= np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = new_function(X,Y)

    #x_flat = X.flatten()
    #y_flat = Y.flatten()
    #z_flat = Z.flatten()
    
    fig = plt.figure()
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(title='neuron with two inputs', scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ))

    st.plotly_chart(fig)
   


# Erstelle zwei Spalten: links für den Slider, rechts für das Diagramm
col1, col2 = st.columns([1.5, 3])  # Seitenverhältnis: 1 Teil Slider, 3 Teile Plot

with col1:
    steigung1 = SyncedSlider("weight 1", -3.0, 3.0, 1.0, key_prefix="slider_s1", step=0.1)
    steigung2 = SyncedSlider("weight 2", -3.0, 3.0, 2.0, key_prefix="slider_s2", step=0.1)
    bias1 = SyncedSlider('bias', -5.0, 5.0, -4.0, key_prefix="slider_y1", step=0.1)
    #st.divider()
    

with col2:
    plot_with_new_function(steigung1.value(), bias1.value(), steigung2.value())