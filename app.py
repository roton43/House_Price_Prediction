import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Loading the trained model pipeline
model = joblib.load("/media/maverick/F/M_ML/Course/house_price_prediction/models/best_price_pipeline.pkl")

# Automatically load feature names from trained model
FEATURE_NAMES = model.feature_names_in_

# Load original dataset to get feature ranges for UI components
data = pd.read_csv('/media/maverick/F/M_ML/Course/house_price_prediction/data/USA_Housing.csv')

# Prediction function that takes user inputs, creates a DataFrame, and returns the predicted price
def predict_price(*inputs):
    
    input_df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    
    prediction = model.predict(input_df)[0]
    
    return f"🏠 Estimated House Price: ${prediction:,.2f}"


# Building dynamic input components based on feature types and ranges
def build_inputs():
    
    inputs = []
    
    for feature in FEATURE_NAMES:
        
        col = data[feature]
        
        # If feature is integer & small range → slider
        if pd.api.types.is_integer_dtype(col) and col.nunique() < 20:
            inputs.append(
                gr.Slider(
                    minimum=int(col.min()),
                    maximum=int(col.max()),
                    step=1,
                    value=int(col.median()),
                    label=feature,
                    info=f"Range: {col.min()} – {col.max()}"
                )
            )
        
        # If numeric continuous → number box
        elif pd.api.types.is_numeric_dtype(col):
            inputs.append(
                gr.Number(
                    value=float(col.median()),
                    label=feature,
                    info=f"Typical range: {col.min():.0f} – {col.max():.0f}"
                )
            )
        
        else:
            # If categorical (optional handling)
            inputs.append(
                gr.Dropdown(
                    choices=col.unique().tolist(),
                    value=col.mode()[0],
                    label=feature
                )
            )
    
    return inputs


# Interface construction using Gradio Blocks
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    
    gr.Markdown("""
    # 🏠 House Price Prediction System
    
    This application predicts house prices using a trained machine learning pipeline.
    
    Please enter property details below.
    """)
    
    input_components = build_inputs()
    
    predict_button = gr.Button("🔍 Predict Price", variant="primary")
    
    output = gr.Textbox(
        label="Predicted House Price",
        placeholder="Prediction will appear here..."
    )
    
    predict_button.click(
        fn=predict_price,
        inputs=input_components,
        outputs=output
    )
    
    gr.Markdown("### 🏡 Example Houses")
    
    gr.Examples(
        examples=[
            data[FEATURE_NAMES].iloc[0].tolist(),
            data[FEATURE_NAMES].iloc[10].tolist(),
            data[FEATURE_NAMES].iloc[25].tolist()
        ],
        inputs=input_components
    )


# ---------------------------------------------------
# 5️⃣ Launch
# ---------------------------------------------------
interface.launch(share=True)