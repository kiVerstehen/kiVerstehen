import torch
import torch.nn as nn
import streamlit as st


def hundOderKatzeNNtesten(größe=50,gewicht=10):
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

    # Modell laden
    model_path="../Beispiel-Modelle/Modelle/katzeOderHundGrößeGewicht.pth"
    model = SimpleNN()  # Initialisiere das Modell
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Setze das Modell in den Evaluationsmodus
    #print("Modell erfolgreich geladen!")
    # Beispielvorhersage
    test_data = torch.tensor([[größe, gewicht]], dtype=torch.float32)  # Größe und Gewicht
    prediction = model(test_data)
    hundchance = round(prediction.item(),3)
    katzchance = 1 - hundchance
    #print(f"Vorhersage für {größe} kg und {gewicht} cm:")
    #print(f"Hund:  {hundchance}")
    #print(f"Katze: {katzchance}")

    return hundchance,katzchance

left, middle, right = st.columns(3)
größe = int(left.text_input("Größe", "20"))
gewicht = int(left.text_input("Gewicht", "25"))



if left.button("KI fragen", icon="🤖", use_container_width=True):
    hundchance,katzchance = hundOderKatzeNNtesten(größe,gewicht)
    left.markdown(f"Vorhersage für **{größe} cm** und **{gewicht} kg**:")
    left.markdown(f"**Hund**:  {hundchance}")
    left.markdown(f"**Katze**: {katzchance}")