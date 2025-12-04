from fastbook import *
from fastai.vision.widgets import *
from PIL import Image
import platform
import streamlit as st

def testeBildInModell(projektname, bildname):

    def andereKat(zahl):
        if zahl==tensor(1): return tensor(0)
        if zahl==tensor(0): return tensor(1)

    im = Image.open(f'../Beispiel-Modelle/Testbilder/{bildname}')
    #frag das Modell, ob es sich beim Bild um x oder y handelt.
    
    #load pkl-model
    if platform.system() == "Linux":
        learn_inf = load_learner(f'../Beispiel-Modelle/Modelle/{projektname}-linux.pkl')
    elif platform.system() == "Windows":
        learn_inf = load_learner(f'../Beispiel-Modelle/Modelle/{projektname}.pkl')
    
    #predict for image 'blabla.jpeg'
    img = PILImage.create(f'../Beispiel-Modelle/Testbilder/{bildname}') 
    pred,pred_idx,probs = learn_inf.predict(img)
    #gebe die prediction aus
    #print(f'Das Bild ist zu {probs[pred_idx]*100:.2f}% {learn_inf.dls.vocab[pred_idx].capitalize()} und zu {100-probs[pred_idx]*100:.2f}% {learn_inf.dls.vocab[andereKat(pred_idx)].capitalize()}.')
    
    
    return probs[pred_idx]*100, learn_inf.dls.vocab[pred_idx].capitalize(), 100-probs[pred_idx]*100, learn_inf.dls.vocab[andereKat(pred_idx)].capitalize()

left, middle, right = st.columns(([1, 3, 1]))

modell = middle.selectbox(
    "Which modell do you want to test?",
    ("dog or cat", "tomato or apple", "doctor or construction worker","cool or uncool"),
)

if modell=="dog or cat":
    testpics = middle.selectbox(
        "Which picture do you want to test?",
        ("katze.jpg", "hund.jpg", "chihuahua.jpg", "bär.jpg"),
    )
    modellname="hund oder katze"
elif modell=="tomato or apple":
    testpics = middle.selectbox(
        "Which picture do you want to test?",
        ("tomate.jpg", "apfel.jpg", "paprika.jpg"),
    )
    modellname="tomate oder apfel"
elif modell=="doctor or construction worker":
    testpics = middle.selectbox(
        "Which picture do you want to test?",
        ("arzt.jpg", "bauarbeiter.jpg", "weißeTasse.jpg", "arztMitHelm.jpg", "mannImAnzug.jpg"),
    )
    modellname="arzt oder bauarbeiter"
elif modell=="cool or uncool":
    testpics = middle.selectbox(
        "Which picture do you want to test?",
        ("sonnenbrille.jpg", "crocs.jpg", "vokuhila.jpg"),
    )
    modellname="cool oder uncool"



if middle.button("ask the AI", icon="🤖", use_container_width=True):
    #st.text(modell + testpics)
    proKat1, dic1, proKat2, dic2 = testeBildInModell(modellname, testpics)
    middle.image(f'../Beispiel-Modelle/Testbilder/{testpics}', width=500)
    middle.text(f"The picture is {proKat1:.2f}% {dic1} and {proKat2:.2f}% {dic2}")

