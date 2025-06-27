import streamlit as st

class SyncedSlider:
    def __init__(self, label, min_val, max_val, default_val, key_prefix, step=1):
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.key_prefix = key_prefix

        # Session keys
        self.value_key = f"{key_prefix}_value"
        self.slider_key = f"{key_prefix}_slider"
        self.input_key = f"{key_prefix}_input"
        self.minus_key = f"{key_prefix}_minus"
        self.plus_key = f"{key_prefix}_plus"

        # Initialize central value
        if self.value_key not in st.session_state:
            st.session_state[self.value_key] = default_val

        self._render()

    def _on_slider_change(self):
        st.session_state[self.value_key] = st.session_state[self.slider_key]

    def _on_input_change(self):
        st.session_state[self.value_key] = st.session_state[self.input_key]

    def _on_button_press(self, buttonkey):
        if buttonkey == "+":
            st.session_state[self.value_key] = min(
                self.max_val, st.session_state[self.value_key] + self.step
            )
        elif buttonkey == "-":
            st.session_state[self.value_key] = max(
                self.min_val, st.session_state[self.value_key] - self.step
            )

    def _render(self):
        value = st.session_state[self.value_key]

        # Inject minimal button styling > button
        st.markdown("""
            <style>
            /* Style buttons inside stButton containers */
            div[data-testid="stButton"] * {     
                padding: 0px 0px;
                font-size: 6px !important;
                height: 5px;
                width: 100%;
                margin: 0px 0;
                border-radius: 3px;
            }
            div[data-testid="stColumn"] * {
                gap: 0rem !important;
                
            }
            </style>
        """, unsafe_allow_html=True)


        st.text(self.label)
        # Layout: slider | input | +/- buttons stacked
        col1, col2, col3, col4 = st.columns([10, 3, 1, 1])

        with col1:
            st.slider(
                label=self.label,
                min_value=self.min_val,
                max_value=self.max_val,
                value=value,
                key=self.slider_key,
                on_change=self._on_slider_change,
                label_visibility="collapsed" 
            )

        with col2:
            st.number_input(
                label=self.label,
                min_value=self.min_val,
                max_value=self.max_val,
                step=self.step,
                value=value,
                key=self.input_key,
                on_change=self._on_input_change,
                label_visibility="collapsed",
                format="%d" if isinstance(self.step, int) else "%0.1f"
            )

        with col3:
            st.button("➖", key=self.minus_key, on_click=lambda: self._on_button_press("-"))

        with col4:
            st.button("➕", key=self.plus_key, on_click=lambda: self._on_button_press("+"))

    def value(self):
        return st.session_state[self.value_key]
    