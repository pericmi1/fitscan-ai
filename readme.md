# FitScan AI – Multimodal Fitness Coach

FitScan AI is a multimodal AI application that combines Computer Vision, Machine Learning, and Natural Language Processing to analyze workout photos and predict calorie burn.

## What the App Does

1. Upload a photo of your exercise
2. Describe yourself in natural language (age, weight, duration, etc.)
3. The app detects the exercise type from the image
4. Extracts your body metrics from the text
5. Predicts how many calories you burned
6. Generates a personalized coaching response

## Live Demo

[Open FitScan AI on HuggingFace Spaces](https://huggingface.co/spaces/pericmi1/fitscan)

## How the Three AI Blocks Work Together

```
Exercise Photo  -->  [CV]  EfficientNetB0  -->  workout_type (HIIT / Strength)
                                                        |
Free Text  -->  [NLP]  GPT-4o-mini  -->  JSON parameters (age, weight, ...)
                                                        |
                        [ML]  XGBoost  -->  calories_burned
                                                        |
                        [NLP]  GPT-4o-mini  -->  personalized explanation
```

## Dataset Description

### ML Block

- **Dataset:** Gym Members Exercise Dataset (Kaggle)
- **Link:** https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
- **Size:** 973 rows, 15 columns
- **Features:** Age, Weight, Height, Session Duration, Workout Type, BPM values, Experience Level, etc.
- **Target:** Calories Burned

### CV Block

- **Dataset:** Workout/Exercise Images (Kaggle)
- **Link:** https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images
- **Size:** 22 exercise classes, approximately 600-700 images each (13,853 total)
- **Classes used:** Mapped to 2 categories: HIIT and Strength

### NLP Block

- **Data:** Runtime user text input (no fixed dataset)
- **Model:** OpenAI GPT-4o-mini via API

## Preprocessing Steps

### ML Preprocessing

- No missing values in dataset
- Label encoding for Workout_Type and Gender
- Engineered feature: Heart_Rate_Intensity = Avg_BPM / Max_BPM
- 80/20 train/test split

### CV Preprocessing

- EfficientNet-specific preprocess_input() applied (not simple 0-1 rescaling)
- Images resized to 224x224 pixels
- Training augmentation: rotation, flip, zoom, brightness adjustment
- Class imbalance handled with computed class weights (HIIT=1.74, Strength=0.70)
- Dataset split: 70% train / 15% val / 15% test

### NLP Preprocessing

- Raw user text passed directly to GPT-4o-mini
- Prompt engineering with explicit JSON schema and unit conversion rules

## Models and Evaluation

### ML Model

| Model                    | CV R2  | Test R2 | MAE        |
| ------------------------ | ------ | ------- | ---------- |
| Random Forest (baseline) | 0.9657 | 0.9713  | 37.60 kcal |
| XGBoost (baseline)       | 0.9795 | 0.9822  | 27.48 kcal |
| Random Forest (tuned)    | 0.9129 | 0.9141  | 64.53 kcal |
| XGBoost (tuned)          | 0.9847 | 0.9873  | 24.09 kcal |

Selected model: XGBoost (tuned) with Test R2 = 0.9873 and MAE = 24.09 kcal

### CV Model

| Iteration | Method                                     | Val Accuracy |
| --------- | ------------------------------------------ | ------------ |
| 1         | EfficientNetB0 frozen (feature extraction) | 99.61%       |
| 2         | EfficientNetB0 fine-tuned (top 30 layers)  | 99.76%       |

Final test accuracy: 99.7% (only 7 errors out of 2,099 test images)

### NLP Model

| Iteration | Approach                                         | Result                                         |
| --------- | ------------------------------------------------ | ---------------------------------------------- |
| 1         | Zero-shot, simple prompt                         | JSON parseable but inconsistent                |
| 2         | Zero-shot with explicit JSON schema and defaults | Always valid JSON, workout_type always from CV |

## Links

- **App (HuggingFace Space):** https://huggingface.co/spaces/perimi1/fitscan-ai
- **ML Model:** saved as calories_model.pkl (XGBoost)
- **CV Model:** saved as best_cv_model.keras (EfficientNetB0)
- **GitHub Repository:** https://github.com/pericmi1/fitscan-ai

## Project Structure

```
fitscan-ai/
├── ml/
│   ├── train_model.ipynb       # ML training notebook
│   └── calories_model.pkl      # trained XGBoost model
├── cv/
│   ├── train_cv.ipynb          # CV training notebook
│   └── best_cv_model.keras     # trained EfficientNetB0 model
├── app/
│   ├── app.py                  # Gradio application
│   └── requirements.txt        # dependencies
├── documentation.md            # full project documentation
└── readme.md                   # this file
```

## Local Setup

```bash
conda create -n fitscan python=3.10 -y
conda activate fitscan
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn xgboost pillow gradio openai
brew install libomp  # macOS only

export OPENAI_API_KEY=KEY
cd app/
python app.py
```

## Author

Milos Peric, ZHAW School of Management and Law, AI Applications Module, 2026
