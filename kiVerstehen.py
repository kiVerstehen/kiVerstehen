import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import torch
import torch.nn as nn
import torch.optim as optim

import simple_image_download.simple_image_download as simp
import os
from fastbook import *
from fastai.vision.widgets import *
from IPython.display import Image as Ima

 
def aufgabe3():

    # Feste Werte für Größe (in cm) und Gewicht (in kg)
    # Daten für Katzen
    cat_heights = [20, 24, 31, 44, 50]  # Größe zwischen 22 und 30 cm
    cat_weights = [30, 20, 15, 25, 30]  # Gewicht zwischen 4 und 8 kg

    # Daten für Hunde
    dog_heights = [18, 30, 35, 60, 45]  # Größe zwischen 45 und 65 cm
    dog_weights = [37, 27, 35, 30, 38]  # Gewicht zwischen 15 und 35 kg

    # Bilder laden
    cat_image_path = 'Grafiken/cathead.png'
    dog_image_path = 'Grafiken/doghead.png'

    def get_image(path, zoom=0.2):  # Angepasste Zoomstufe
        return OffsetImage(mpimg.imread(path), zoom=zoom)

    # Funktion zum Berechnen und Plotten der Abstände zur Geraden
    def plot_with_distances(steigung=1.0, y_achsenabschnitt=0.0, save=False):
        fig, ax = plt.subplots()

        # Scatterplot für Katzen erstellen
        for i in range(len(cat_heights)):
            ab = AnnotationBbox(get_image(cat_image_path), (cat_heights[i], cat_weights[i]), frameon=False)
            ax.add_artist(ab)
        
        # Scatterplot für Hunde erstellen
        for i in range(len(dog_heights)):
            ab = AnnotationBbox(get_image(dog_image_path), (dog_heights[i], dog_weights[i]), frameon=False)
            ax.add_artist(ab)

        # Gerade hinzufügen
        x_vals = np.linspace(10, 70, 100)  # Erzeuge 100 Werte zwischen 10 und 70
        y_vals = y_achsenabschnitt + steigung * x_vals  # Berechne die y-Werte basierend auf der Geradengleichung
        ax.plot(x_vals, y_vals, '--', color='red', label=f'Gerade: y = {steigung:.2f}x + {y_achsenabschnitt:.2f}')

        # Achsenbeschriftungen und -limits setzen
        ax.set_xlim(10, 68)
        ax.set_ylim(12, 42)
        ax.set_xlabel('Größe (cm)')
        ax.set_ylabel('Gewicht (kg)')
        ax.legend()  # Legende hinzufügen

        # Abstände berechnen und anzeigen
        total_distance_cats = 0
        total_distance_dogs = 0

        # Abstände für Katzen (oberhalb der Geraden)
        for i in range(len(cat_heights)):
            y_on_line = steigung * cat_heights[i] + y_achsenabschnitt
            if cat_weights[i] > y_on_line:  # nur Katzen oberhalb der Geraden
                distance = cat_weights[i] - y_on_line
                total_distance_cats += distance
                # Linie für Abstand zeichnen
                ax.plot([cat_heights[i], cat_heights[i]], [y_on_line, cat_weights[i]], 'b-')

        # Abstände für Hunde (unterhalb der Geraden)
        for i in range(len(dog_heights)):
            y_on_line = steigung * dog_heights[i] + y_achsenabschnitt
            if dog_weights[i] < y_on_line:  # nur Hunde unterhalb der Geraden
                distance = y_on_line - dog_weights[i]
                total_distance_dogs += distance
                # Linie für Abstand zeichnen
                ax.plot([dog_heights[i], dog_heights[i]], [dog_weights[i], y_on_line], 'g-')

        # Gesamtabstände ausgeben
        print(f'Gesamtabstand der falsch kategorisierten Katzen: {total_distance_cats:.2f}')
        print(f'Gesamtabstand der falsch kategorisierten Hunde: {total_distance_dogs:.2f}')
        print(f'Kostenfunktion - Summe beider Werte:  {total_distance_cats+total_distance_dogs:.2f}')

        # Speichere den Plot als Bilddatei, wenn gewünscht
        if save:
            plt.savefig('aufgabe3.png', dpi=300, bbox_inches='tight')  # Speicher als PNG
            print("Plot gespeichert als 'aufgabe3.png'")
        
        plt.show()

    # Interaktiver Plot mit anpassbarer Gerade und Möglichkeit, den Plot zu speichern
    interact(plot_with_distances, 
            steigung=widgets.FloatSlider(min=-1, max=1, step=0.05, value=0.3),
            y_achsenabschnitt=widgets.FloatSlider(min=10, max=50, step=0.05, value=17.0),
            save=widgets.Checkbox(value=False, description='Plot speichern'))
    
def aufgabe6():
    # Feste Werte für Größe (in cm) und Gewicht (in kg)
    # Daten für Katzen
    cat_heights = [20, 24, 31, 44, 50]  # Größe zwischen 20 und 50 cm
    cat_weights = [30, 20, 15, 25, 30]  # Gewicht zwischen 15 und 30 kg

    # Daten für Hunde
    dog_heights = [18, 30, 35, 60, 45]  # Größe zwischen 18 und 60 cm
    dog_weights = [37, 27, 35, 30, 38]  # Gewicht zwischen 27 und 38 kg

    # Bilder laden
    cat_image_path = 'Grafiken/cathead.png'
    dog_image_path = 'Grafiken/doghead.png'

    def get_image(path, zoom=0.2):  # Angepasste Zoomstufe
        return OffsetImage(mpimg.imread(path), zoom=zoom)

    # Funktion zum Berechnen und Plotten der Abstände zur neuen Funktion
    def plot_with_new_function(w1=0.3, b1=17.0, w2=0.3, b2=17.0, save=False):
        fig, ax = plt.subplots()

        # Scatterplot für Katzen erstellen
        for i in range(len(cat_heights)):
            ab = AnnotationBbox(get_image(cat_image_path), (cat_heights[i], cat_weights[i]), frameon=False)
            ax.add_artist(ab)
        
        # Scatterplot für Hunde erstellen
        for i in range(len(dog_heights)):
            ab = AnnotationBbox(get_image(dog_image_path), (dog_heights[i], dog_weights[i]), frameon=False)
            ax.add_artist(ab)

        # Achsenbeschriftungen und -limits setzen
        ax.set_xlim(10, 68)
        ax.set_ylim(12, 42)
        ax.set_xlabel('Größe (cm)')
        ax.set_ylabel('Gewicht (kg)')
    # ax.legend()  # Legende hinzufügen

        # Neue Funktion definieren
        def new_function(x):
            y_cat = np.maximum(0, w1 * x + b1)
            y_dog = np.maximum(0, w2 * x + b2)
            return y_cat + y_dog
        
        x_vals = np.linspace(10, 70, 100)  # Erzeuge 100 Werte zwischen 10 und 70
        y_new = new_function(x_vals)
        ax.plot(x_vals, y_new, '-', color='purple', label=f'max(0,{w1} * x + {b1}) + max(0,{w2} * x + {b2})')
        ax.legend()  # Legende aktualisieren

        # Abstände berechnen und anzeigen
        total_distance_cats_above = 0
        total_distance_dogs_below = 0

        # Abstände für Katzen
        for i in range(len(cat_heights)):
            y_on_line = new_function(cat_heights[i])
            if cat_weights[i] > y_on_line:  # nur Katzen oberhalb der Geraden
                distance = cat_weights[i] - y_on_line
                total_distance_cats_above += distance
                # Linie für Abstand zeichnen
                ax.plot([cat_heights[i], cat_heights[i]], [y_on_line, cat_weights[i]], 'b-')

        # Abstände für Hunde
        for i in range(len(dog_heights)):
            y_on_line = new_function(dog_heights[i])
            if dog_weights[i] < y_on_line:  # nur Hunde unterhalb der Geraden
                distance = y_on_line - dog_weights[i]
                total_distance_dogs_below += distance
                # Linie für Abstand zeichnen
                ax.plot([dog_heights[i], dog_heights[i]], [dog_weights[i], y_on_line], 'g-')

        # Gesamtabstände ausgeben
        print(f'Gesamtabstand der falsch kategorisierten Katzen: {total_distance_cats_above:.2f}')
        print(f'Gesamtabstand der falsch kategorisierten Hunde: {total_distance_dogs_below:.2f}')
        print(f'Kostenfunktion - Summe beider Werte: {total_distance_cats_above + total_distance_dogs_below:.2f}')

        # Speichere den Plot als Bilddatei, wenn gewünscht
        if save:
            plt.savefig('aufgabe6.png', dpi=300, bbox_inches='tight')  # Speicher als PNG
            print("Plot gespeichert als 'aufgabe6.png'")
        
        plt.show()

    # Interaktiver Plot mit anpassbarer Funktion und Möglichkeit, den Plot zu speichern
    interact(plot_with_new_function, 
            w1=widgets.FloatSlider(min=-2, max=1, step=0.05, value=0.-0.5),
            b1=widgets.FloatSlider(min=0, max=50, step=0.05, value=25.0),
            w2=widgets.FloatSlider(min=-0.5, max=2, step=0.05, value=0.3),
            b2=widgets.FloatSlider(min=-10, max=20, step=0.05, value=13.0),
            save=widgets.Checkbox(value=False, description='Plot speichern'))

def aufgabe9(epochen=1000):
    # Originaldaten
    cat_heights = np.array([20, 24, 31, 44, 50], dtype=np.float32).reshape(-1, 1)
    cat_weights = np.array([30, 20, 15, 25, 30], dtype=np.float32).reshape(-1, 1)

    dog_heights = np.array([18, 30, 35, 60, 45], dtype=np.float32).reshape(-1, 1)
    dog_weights = np.array([37, 27, 35, 30, 38], dtype=np.float32).reshape(-1, 1)

    # Zusätzliche Datenpunkte generieren
    #np.random.seed(42)
    additional_cat_heights = [[28.72700594], [57.53571532], [46.59969709], [39.93292421], [17.80093202], [17.79972602], [12.90418061], [53.30880729], [40.05575059], [45.40362889], [11.02922471], [58.49549261], [51.62213204], [20.61695553], [19.09124836], [19.17022549], [25.21211215], [36.23782158], [31.59725093], [24.56145701], [40.59264474], [16.97469303], [24.60723243], [28.31809216], [32.80349921], [49.25879807], [19.98368911], [35.71172192], [39.62072844], [12.32252064]]
    additional_cat_weights = [[22.15089704], [13.41048247], [11.30103186], [28.97771075], [29.31264066], [26.16794696], [16.09227538], [11.95344228], [23.68466053], [18.80304987], [12.4407647], [19.9035382], [10.68777042], [28.18640804], [15.17559963], [23.25044569], [16.23422152], [20.40136042], [20.93420559], [13.69708911], [29.39169256], [25.50265647], [28.78997883], [27.89654701], [21.95799958], [28.4374847], [11.76985004], [13.91965725], [10.90454578], [16.50660662]]

    additional_dog_heights = [[33.32063738], [26.28094191], [59.72425055], [31.4051996], [26.85607058], [42.56176499], [18.4554535], [58.13181885], [14.47303862], [69.2132162], [56.33468616], [21.92294089], [10.33132703], [58.92768571], [52.41144063], [53.74043008], [56.2762208], [14.4426791], [31.50794371], [16.95214357], [61.78620555], [47.39788761], [29.85388149], [13.81350102], [28.6589393], [29.51099932], [53.7763707], [48.25344828], [63.23276455], [38.33289551]]
    additional_dog_weights = [[27.39188492], [39.26489574], [40.21570097], [36.22554395], [40.4193436], [34.87591193], [35.45465659], [33.55082037], [25.50838253], [27.15782854], [25.62858371], [37.72820823], [31.28711962], [35.17141382], [43.15132948], [29.98584458], [33.20765846], [40.11102277], [29.57596331], [26.5395982], [30.79502906], [28.22442575], [43.59395305], [41.16240759], [37.66807513], [42.4292118], [41.07344154], [28.73140118], [42.85117997], [35.78684484]]


    # Kombiniere die Originaldaten mit den zusätzlichen Datenpunkten
    all_cat_heights = np.vstack((cat_heights, additional_cat_heights))
    all_cat_weights = np.vstack((cat_weights, additional_cat_weights))

    all_dog_heights = np.vstack((dog_heights, additional_dog_heights))
    all_dog_weights = np.vstack((dog_weights, additional_dog_weights))

    # Labels erstellen: 0 für Katzen, 1 für Hunde
    cat_labels = np.zeros((all_cat_heights.shape[0], 1))
    dog_labels = np.ones((all_dog_heights.shape[0], 1))

    # Daten und Labels kombinieren
    all_heights = np.vstack((all_cat_heights, all_dog_heights))
    all_weights = np.vstack((all_cat_weights, all_dog_weights))
    all_labels = np.vstack((cat_labels, dog_labels))

    # Kombiniere Höhen und Gewichte zu Eingabedaten
    all_data = np.hstack((all_heights, all_weights))

    # In Tensoren konvertieren
    data_tensor = torch.tensor(all_data, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    # Definiere das neuronale Netzwerk
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 10)
            self.fc4 = nn.Linear(10, 10)
            self.fc5 = nn.Linear(10, 10)
            self.fc6 = nn.Linear(10, 10)
            self.fc7 = nn.Linear(10, 10)
            self.fc8 = nn.Linear(10, 10)
            self.fc9 = nn.Linear(10, 10)
            self.fc10 = nn.Linear(10, 10)
            self.fc11 = nn.Linear(10, 10)
            self.fc12 = nn.Linear(10, 10)
            self.fc13 = nn.Linear(10, 10)
            self.fc14 = nn.Linear(10, 10)
            self.fc15 = nn.Linear(10, 10)
            self.fc16 = nn.Linear(10, 10)
            self.fc17 = nn.Linear(10, 10)
            self.fc18 = nn.Linear(10, 10)
            self.fc19 = nn.Linear(10, 10)
            self.fc20 = nn.Linear(10, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc4(x))
            x = self.relu(self.fc5(x))
            x = self.relu(self.fc6(x))
            x = self.relu(self.fc7(x))
            x = self.relu(self.fc8(x))
            x = self.relu(self.fc9(x))
            x = self.relu(self.fc10(x))
            x = self.relu(self.fc11(x))
            x = self.relu(self.fc12(x))
            x = self.relu(self.fc13(x))
            x = self.relu(self.fc14(x))
            x = self.relu(self.fc15(x))
            x = self.relu(self.fc16(x))
            x = self.relu(self.fc17(x))
            x = self.relu(self.fc18(x))
            x = self.relu(self.fc19(x))
            x = self.sigmoid(self.fc20(x))
            return x

    # Initialisiere das Netzwerk
    model = SimpleNN()

    # Verlustfunktion und Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training des Netzes
    def train_model(model, data, labels, epochs=epochen):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoche [{epoch+1}/{epochs}], Kosten: {loss.item():.4f}')
                

    train_model(model, data_tensor, labels_tensor)

    # Visualisiere die Ergebnisse
    def plot_data_and_decision_boundary(model, data, labels):
        plt.figure(figsize=(10, 6))

        # Gesamtdaten plotten
        plt.scatter(all_cat_heights, all_cat_weights, color='blue', label='Katzen')
        plt.scatter(all_dog_heights, all_dog_weights, color='red', label='Hunde')

        # Entscheidunggrenze plotten
        x_min, x_max = all_heights.min() - 1, all_heights.max() + 1
        y_min, y_max = all_weights.min() - 1, all_weights.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            decision_boundary = model(grid_tensor).numpy().reshape(xx.shape)
        contour = plt.contourf(xx, yy, decision_boundary, alpha=0.5, cmap='coolwarm', levels=np.linspace(0, 1, 11))
        plt.colorbar(contour, ticks=np.linspace(0, 1, 11))
        
        # Achsenbeschriftungen und Titel setzen
        plt.xlabel('Größe (cm)')
        plt.ylabel('Gewicht (kg)')
        #plt.title('Scatterplot von Katzen und Hunden mit Entscheidunggrenze')
        plt.legend()
        plt.show()

    plot_data_and_decision_boundary(model, data_tensor, labels_tensor)

def zeigeBeispielBilder(projektname):
    #laden der bilder in fns
    path = Path(f'Beispiel-Modelle/{projektName}')
    fns = get_image_files(path)
    #gibt liste von failed downloaded images
    #failed = verify_images(fns)
    #print(f"Es gab {len(failed)} fehlerhafte Downloads. Ich behebe das automatisch!")
    #unlink the failed downloaded images
    #failed.map(Path.unlink)
    daten = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
    dls = daten.dataloaders(path)
    dls.valid.show_batch(max_n=8, nrows=2)

def testeBildInModell(projektname, bildname):

    def andereKat(zahl):
        if zahl==tensor(1): return tensor(0)
        if zahl==tensor(0): return tensor(1)

    im = Image.open(f'Beispiel-Modelle/Testbilder/{bildname}')
    #frag das Modell, ob es sich beim Bild um x oder y handelt.
    #load pkl-model
    learn_inf = load_learner(f'Beispiel-Modelle/Modelle/{projektname}.pkl')
    #predict for image 'blabla.jpeg'
    pred,pred_idx,probs = learn_inf.predict(f'Beispiel-Modelle/Testbilder/{bildname}')
    #gebe die prediction aus
    print(f'Das Bild ist zu {probs[pred_idx]*100:.2f}% {learn_inf.dls.vocab[pred_idx].capitalize()} und zu {100-probs[pred_idx]*100:.2f}% {learn_inf.dls.vocab[andereKat(pred_idx)].capitalize()}.')
    return im.to_thumb(256,256)

    


