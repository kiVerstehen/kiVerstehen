import dash
from dash import html
from dash_canvas import DashCanvas
import base64

# Encode your image file
def encode_image(image_file):
    with open(image_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

app = dash.Dash(
    __name__,
    requests_pathname_prefix='/app006/',
    assets_url_path='/app006/assets'
)

image_path = '../Grafiken/diabetes2.png'
image_base64 = encode_image(image_path)

app.layout = html.Div([
    DashCanvas(
        id='canvas',
        width=800,
        height=600,
        image_content='data:image/png;base64,{}'.format(image_base64),
        lineWidth=10,
        lineColor='blue',
        #tool='line', 
        hide_buttons=['zoom', 'pan', 'select', 'rectangle'],  # Example: hiding some buttons
    )
])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8506)