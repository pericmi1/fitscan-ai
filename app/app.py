import os
import pickle
import numpy as np
import gradio as gr
from openai import OpenAI
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from dotenv import load_dotenv
load_dotenv()

# ── Load ML model & encoders ──────────────────────────────────────────────────
with open("calories_model.pkl", "rb") as f:
    calories_model = pickle.load(f)

with open("label_encoder_workout.pkl", "rb") as f:
    le_workout = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("cv_class_names.pkl", "rb") as f:
    cv_class_names = pickle.load(f)

# ── Load CV model ─────────────────────────────────────────────────────────────
cv_model = tf.keras.models.load_model("cv_model_final.keras")

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── CV Block: classify exercise image ─────────────────────────────────────────
def classify_exercise(img):
    """Takes a PIL image, returns predicted workout type and confidence."""
    img_resized = img.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    probs = cv_model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_class = cv_class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_probs = {cls: float(p) for cls, p in zip(cv_class_names, probs)}
    return predicted_class, confidence, all_probs


# ── NLP Block: extract parameters from free text ──────────────────────────────
def extract_parameters(user_text, cv_workout_type):
    """Uses OpenAI to extract workout parameters from natural language input."""
    system_prompt = """You are a fitness parameter extractor. 
Extract workout parameters from the user's text and return ONLY a valid JSON object.
Use the cv_workout_type provided as the workout_type value.

Return exactly this JSON structure (fill in values from the text, use defaults if not mentioned):
{
  "age": <int, default 30>,
  "weight_kg": <float, default 75.0>,
  "height_m": <float, default 1.75>,
  "session_duration_h": <float, default 1.0>,
  "workout_type": "<use the cv_workout_type provided>",
  "gender": "<Male or Female, default Male>",
  "max_bpm": <int, default 185>,
  "avg_bpm": <int, default 145>,
  "resting_bpm": <int, default 65>,
  "fat_percentage": <float, default 20.0>,
  "water_intake": <float, default 2.5>,
  "workout_frequency": <int, default 3>,
  "experience_level": <int 1-3, default 2>
}

Rules:
- height: if given in cm, convert to meters (e.g. 180cm = 1.80)
- experience_level: beginner=1, intermediate=2, advanced=3
- workout_type: always use the cv_workout_type value provided
- Return ONLY the JSON, no explanation, no markdown"""

    user_prompt = f"""cv_workout_type: {cv_workout_type}
User text: {user_text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    import json
    json_str = response.choices[0].message.content.strip()
    params = json.loads(json_str)
    return params


# ── ML Block: predict calories ────────────────────────────────────────────────
def predict_calories(params):
    """Uses the XGBoost model to predict calories burned."""
    import pandas as pd

    workout_enc = le_workout.transform([params["workout_type"]])[0]
    gender_enc = le_gender.transform([params["gender"]])[0]
    bmi = params["weight_kg"] / (params["height_m"] ** 2)
    heart_rate_intensity = params["avg_bpm"] / params["max_bpm"] if params["max_bpm"] > 0 else 0.8

    features = pd.DataFrame([{
        "Age": params["age"],
        "Weight (kg)": params["weight_kg"],
        "Height (m)": params["height_m"],
        "Session_Duration (hours)": params["session_duration_h"],
        "Workout_Type_encoded": workout_enc,
        "Gender_encoded": gender_enc,
        "Max_BPM": params["max_bpm"],
        "Avg_BPM": params["avg_bpm"],
        "Resting_BPM": params["resting_bpm"],
        "Fat_Percentage": params["fat_percentage"],
        "Water_Intake (liters)": params["water_intake"],
        "Workout_Frequency (days/week)": params["workout_frequency"],
        "Experience_Level": params["experience_level"],
        "BMI": bmi,
        "Heart_Rate_Intensity": heart_rate_intensity
    }])

    calories = calories_model.predict(features)[0]
    return round(float(calories), 0)


# ── NLP Block: generate explanation ───────────────────────────────────────────
def generate_explanation(params, calories, cv_confidence):
    """Uses OpenAI to generate a personalized fitness explanation."""
    exp_map = {1: "beginner", 2: "intermediate", 3: "advanced"}
    experience_str = exp_map.get(params["experience_level"], "intermediate")

    prompt = f"""You are FitScan AI, a friendly and motivating personal fitness coach.

A user just completed a workout. Here is their data:
- Age: {params['age']} years
- Gender: {params['gender']}
- Weight: {params['weight_kg']} kg, Height: {params['height_m']} m
- Workout type: {params['workout_type']} (detected from image with {cv_confidence:.0%} confidence)
- Session duration: {params['session_duration_h']} hours
- Average heart rate: {params['avg_bpm']} BPM
- Experience level: {experience_str}
- Workout frequency: {params['workout_frequency']} days/week
- Predicted calories burned: {calories:.0f} kcal

Write a motivating, personalized response (3-4 sentences) that:
1. Acknowledges their workout and calories burned
2. Gives one specific tip based on their data
3. Encourages them to keep going

Be warm, specific, and concise."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_workout(image, user_text):
    """Main function combining all 3 blocks."""
    import json

    if image is None:
        return "❌ Please upload an exercise image.", "", "", ""

    if not user_text.strip():
        return "❌ Please describe yourself and your workout.", "", "", ""

    try:
        # ── BLOCK 1: CV ──────────────────────────────────────────────────────
        cv_class, cv_conf, cv_probs = classify_exercise(image)
        cv_result = f"🏋️ **Detected:** {cv_class}  \n📊 **Confidence:** {cv_conf:.1%}  \n\nProbabilities:\n"
        for cls, prob in cv_probs.items():
            bar = "█" * int(prob * 20)
            cv_result += f"- {cls}: {prob:.1%} {bar}\n"

        # ── BLOCK 2: NLP – Extract parameters ────────────────────────────────
        params = extract_parameters(user_text, cv_class)
        params_json = json.dumps(params, indent=2)

        # ── BLOCK 3: ML – Predict calories ───────────────────────────────────
        calories = predict_calories(params)
        ml_result = f"🔥 **Predicted Calories Burned: {calories:.0f} kcal**"

        # ── BLOCK 2b: NLP – Generate explanation ─────────────────────────────
        explanation = generate_explanation(params, calories, cv_conf)

        return cv_result, params_json, ml_result, explanation

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""


# ── Example inputs ────────────────────────────────────────────────────────────
examples = [
    ["I am 28 years old, male, 80kg, 1.80m tall. I trained for 1 hour, 3 times a week. I am intermediate level."],
    ["Female, 35 years old, 65kg, 165cm. I did a 45 minute session. I train 4 days a week. Advanced level."],
    ["I'm 22, male, 75kg and 178cm. Trained for 1.5 hours today. Beginner, only started 2 months ago."],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FitScan AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏋️ FitScan AI – Multimodal Fitness Coach
    **Upload an exercise photo + describe yourself → Get your calorie prediction & personalized coaching**
    
    This app combines three AI blocks:
    - 🖼️ **Computer Vision** – Classifies your exercise from the image
    - 💬 **NLP** – Extracts your body metrics from natural language
    - 📊 **ML** – Predicts calories burned using XGBoost
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📸 Upload Exercise Photo")
            text_input = gr.Textbox(
                label="💬 Describe yourself & your workout",
                placeholder="e.g. I am 28 years old, male, 80kg, 1.80m tall. I trained for 1 hour, 3 times a week.",
                lines=4
            )
            submit_btn = gr.Button("🚀 Analyze My Workout", variant="primary", size="lg")

            gr.Markdown("### 💡 Example inputs:")
            gr.Examples(
                examples=examples,
                inputs=[text_input],
                label="Click to try"
            )

        with gr.Column(scale=1):
            cv_output = gr.Markdown(label="🖼️ CV Block – Exercise Classification")
            params_output = gr.Code(label="💬 NLP Block – Extracted Parameters (JSON)", language="json")
            ml_output = gr.Markdown(label="📊 ML Block – Calorie Prediction")
            explanation_output = gr.Markdown(label="🤖 AI Coach Explanation")

    submit_btn.click(
        fn=analyze_workout,
        inputs=[image_input, text_input],
        outputs=[cv_output, params_output, ml_output, explanation_output]
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. Upload a photo of your exercise (push-up, squat, deadlift, etc.)
    2. Describe your body metrics and workout in natural language
    3. The CV model classifies the exercise type (HIIT or Strength)
    4. The NLP model extracts your parameters from the text
    5. The ML model predicts how many calories you burned
    6. The AI coach gives you personalized feedback
    """)

if __name__ == "__main__":
    demo.launch()
