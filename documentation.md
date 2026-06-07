# AI Applications Project Documentation

## Project Metadata

- **Project title:** FitScan AI – Multimodal Fitness Coach
- **Student:** Milos Peric
- **GitHub repository URL:** https://github.com/pericmi1/fitscan-ai.git
- **Deployment URL:** https://huggingface.co/spaces/pericmi1/fitscan
- **Submission date:** 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution:

- Primary block 1: ML Numeric Data
- Primary block 2: Computer Vision

Third block (NLP) implemented as extra work for parameter extraction and result explanation.

## 1. Project Foundation

### 1.1 Problem Definition

- **Problem statement:** Users cannot easily estimate how many calories they burn during a workout without specialized equipment or manual data entry.
- **Goal:** Build a multimodal AI application that classifies an exercise from a photo, extracts body metrics from natural language, and predicts calorie burn in one unified pipeline.
- **Success criteria:** CV accuracy above 85%, ML R2 above 0.85, working deployed app that combines all three blocks end-to-end.

### 1.2 Integration Logic

The three blocks interact in a sequential pipeline:

```
[User Input]
  |
  |-- Exercise Photo  -->  [CV Block] EfficientNetB0
  |                              |
  |                              v  workout_type (HIIT / Strength)
  |
  |-- Free Text  -->  [NLP Block] GPT-4o-mini (parameter extraction)
                             |
                             v  structured JSON (age, weight, duration, ...)
                      [ML Block] XGBoost
                             |
                             v  calories_burned
                      [NLP Block] GPT-4o-mini (personalized explanation)
```

- The **CV block** classifies the exercise image and provides workout_type to the NLP block
- The **NLP block** extracts user parameters from free text and injects workout_type from CV
- The **ML block** receives the complete JSON and predicts calories burned
- The **NLP block** generates a personalized coaching response using all outputs

See [`app.py`](app/app.py) for the full pipeline implementation.

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link                                                                                                 | Type | Size                 | Role in this block                         |
| ----- | ------------------------------------------------------------------------------------------------------------------- | ---- | -------------------- | ------------------------------------------ |
| 1     | [Gym Members Exercise Dataset (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) | CSV  | 973 rows, 15 columns | Training data for calorie prediction model |

#### 2A.2 Preprocessing and Features

See _Preprocessing and Feature Engineering_ in [`ml/train_model.ipynb`](ml/train_model.ipynb)

- **Cleaning steps:** No missing values found; dataset used as-is after validation
- **Preprocessing steps:**
  - Label encoding of Workout_Type (Cardio=0, HIIT=1, Strength=2, Yoga=3)
  - Label encoding of Gender (Female=0, Male=1)
  - BMI already present in dataset (verified)
  - 80/20 train/test split with random_state=42
- **Feature engineering:**
  - New feature Heart_Rate_Intensity = Avg_BPM / Max_BPM, captures how hard the user works relative to their maximum capacity
  - Final feature set: 15 features including Age, Weight, Height, Session_Duration, Workout_Type_encoded, Gender_encoded, Max_BPM, Avg_BPM, Resting_BPM, Fat_Percentage, Water_Intake, Workout_Frequency, Experience_Level, BMI, Heart_Rate_Intensity

#### 2A.3 Model Selection

- **Models tested:** Random Forest, XGBoost
- **Why these models:** Both handle tabular data well and support feature importance analysis. XGBoost is known for strong performance on structured data. Random Forest provides a good interpretable baseline.

#### 2A.4 Model Comparison and Iterations

See _Iteration 1 and 2_ in [`ml/train_model.ipynb`](ml/train_model.ipynb)

| Iteration | Objective                                  | Key changes                                                               | Models used                            | Main metric (CV R2)      | Change vs previous      |
| --------- | ------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------- | ------------------------ | ----------------------- |
| 1         | Establish baseline                         | Default hyperparameters                                                   | Random Forest (n=100), XGBoost (n=100) | RF: 0.9657 / XGB: 0.9795 | Baseline                |
| 2         | Reduce overfitting, improve generalization | RF: max_depth=15, min_samples_split=5; XGB: n=300, lr=0.05, subsample=0.8 | Random Forest (tuned), XGBoost (tuned) | RF: 0.9129 / XGB: 0.9847 | XGB +0.0052, RF -0.0528 |

Selected model: XGBoost Iteration 2 (best CV R2 and lowest MAE)

#### 2A.5 Evaluation and Error Analysis

See _Error Analysis_ in [`ml/train_model.ipynb`](ml/train_model.ipynb)

- **Metrics used:** R2 (cross-validation 5-fold and test), MAE, RMSE
- **Final results (XGBoost tuned):**
  - CV R2: 0.9847 (plus/minus 0.0023)
  - Test R2: 0.9873
  - MAE: 24.09 calories
  - RMSE: 32.59 calories
- **Error patterns:**
  - Slight underestimation at very high calorie values (above 1400 kcal), likely due to fewer training samples in this range
  - Residuals approximately normally distributed around 0, no systematic bias
  - Workout_Type has very low feature importance (~0.005); workout type alone does not predict calories well; duration and experience level dominate

#### 2A.6 Integration with Other Block(s)

- **Inputs received:** workout_type string from CV block (via NLP parameter extraction)
- **Outputs provided:** Predicted calories_burned (float) to NLP block for explanation generation

See [`app.py`, lines 60-100](app/app.py#L60-L100) for the predict_calories() function.

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link                           | Type            | Size       | Role in this block                 |
| ----- | --------------------------------------------- | --------------- | ---------- | ---------------------------------- |
| 1     | User free-text input (runtime)                | Text            | Variable   | Input for parameter extraction     |
| 2     | CV block output (workout_type)                | String          | 1 label    | Injected into extraction prompt    |
| 3     | ML block output (calories) and all parameters | Structured data | ~15 fields | Input for personalized explanation |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** No preprocessing applied; raw user input is passed directly to GPT-4o-mini
- **Prompt design:**
  - Extraction prompt: System prompt defines exact JSON schema with defaults for missing values. Includes conversion rules (cm to m). workout_type is injected from CV block output. Temperature=0 for deterministic extraction.
  - Explanation prompt: User prompt includes all workout parameters and predicted calories. Instructs the model to write 3-4 motivating sentences with one specific tip. Temperature=0.7 for natural variation.

See [`app.py`, lines 103-145](app/app.py#L103-L145) for extract_parameters() and [`app.py`, lines 148-185](app/app.py#L148-L185) for generate_explanation().

#### 2B.3 Approach Selection

- **Approach used:** Prompt engineering with OpenAI GPT-4o-mini (zero-shot)
- **Alternatives considered:** Few-shot prompting (tested but not needed given zero-shot accuracy), local LLM (rejected due to deployment complexity on HuggingFace CPU)

#### 2B.4 Comparison and Iterations

| Iteration | Objective                       | Key changes                                                                                | Model or prompt setup              | Main metric or qualitative check               | Change vs previous                           |
| --------- | ------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| 1         | Basic parameter extraction      | Simple prompt, no schema                                                                   | GPT-4o-mini, zero-shot             | JSON parseable, values reasonable              | Baseline                                     |
| 2         | Robust extraction with defaults | Added explicit JSON schema, defaults, unit conversion rules, injected workout_type from CV | GPT-4o-mini, zero-shot with schema | JSON always valid, workout_type always from CV | Eliminated parsing errors, consistent output |

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** Manual testing with 10 varied inputs covering different languages, missing fields, and unusual formats
- **Results:** JSON extraction successful in all test cases; defaults applied correctly when fields are missing; unit conversion (cm to m) works reliably
- **Error patterns:**
  - Model sometimes estimates BPM values unrealistically if user mentions intensity without numbers; mitigated by sensible defaults in the prompt
  - Very short inputs (e.g. "male, 80kg") produce mostly default values, which is acceptable behavior

#### 2B.6 Integration with Other Block(s)

- **Inputs received:** workout_type from CV block; calories_burned and all parameters from ML block
- **Outputs provided:** Structured JSON parameters to ML block; natural language explanation to user

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link                                                                                        | Type             | Size                                                    | Role in this block                        |
| ----- | ---------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------- | ----------------------------------------- |
| 1     | [Workout/Exercise Images (Kaggle)](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images) | Images (JPG/PNG) | 22 exercise classes, ~600-700 images each, 13,853 total | Training data for exercise classification |

#### 2C.2 Preprocessing and Augmentation

See _Preprocessing and Augmentation_ in [`cv/train_cv.ipynb`](cv/train_cv.ipynb)

- **Image preprocessing:**
  - EfficientNet-specific preprocess_input() applied (critical fix: not simple rescale to 0-1)
  - Resize to 224x224 pixels
  - 22 exercise classes mapped to 2 categories: HIIT (pull-up, push-up, plank, leg raises, russian twist, squat) and Strength (all gym machine and barbell exercises)
  - Dataset split: 70% train / 15% val / 15% test
  - Class imbalance handled via class_weight (Strength: 9,866 images vs HIIT: 3,987 images; weights HIIT=1.74, Strength=0.70)
- **Augmentation strategy (training only):**
  - Random rotation (plus/minus 15 degrees)
  - Width and height shift (plus/minus 10%)
  - Horizontal flip
  - Zoom (plus/minus 10%)
  - Brightness adjustment (0.8 to 1.2)

#### 2C.3 Model Selection

- **Vision model used:** EfficientNetB0 pretrained on ImageNet
- **Why chosen:** Strong performance/size tradeoff; well-suited for transfer learning on small-to-medium datasets; fast inference on CPU for deployment

#### 2C.4 Model Comparison and Iterations

See _Iteration 1 and 2_ in [`cv/train_cv.ipynb`](cv/train_cv.ipynb)

| Iteration | Objective                   | Key changes                                                             | Model(s) used                     | Main metric (Val Accuracy) | Change vs previous |
| --------- | --------------------------- | ----------------------------------------------------------------------- | --------------------------------- | -------------------------- | ------------------ |
| 1         | Feature extraction baseline | EfficientNetB0 frozen, only custom head trained (Dense 256 to 128 to 2) | EfficientNetB0 (frozen)           | 99.61%                     | Baseline           |
| 2         | Fine-tuning                 | Unfreeze top 30 layers, learning rate reduced to 1e-5                   | EfficientNetB0 (partial unfreeze) | 99.76%                     | +0.15%             |

Note: Initial attempt without preprocess_input() yielded only 72% val accuracy. Switching to EfficientNet-specific preprocessing immediately improved accuracy to 97.97% in Epoch 1.

#### 2C.5 Evaluation and Error Analysis

See _Evaluation_ in [`cv/train_cv.ipynb`](cv/train_cv.ipynb)

- **Metrics:** Accuracy, Loss, Confusion Matrix, Classification Report
- **Final results (best model, test set):**
  - Test Accuracy: 99.7%
  - HIIT: 598/603 correct (99.2%)
  - Strength: 1494/1496 correct (99.9%)
  - Only 7 misclassifications out of 2,099 test images
- **Error patterns and limitations:**
  - 5 HIIT images misclassified as Strength, likely exercises that share visual features with gym equipment (e.g. weighted squats)
  - 2 Strength images misclassified as HIIT, likely bodyweight exercises performed in a gym setting
  - Model limited to 2 categories (HIIT and Strength); Cardio and Yoga not classifiable from this dataset

#### 2C.6 Integration with Other Block(s)

- **Inputs received:** Raw exercise image from user (PIL format)
- **Outputs provided:** workout_type string (HIIT or Strength) and confidence score, injected into NLP extraction prompt and used as feature in ML block

See [`app.py`, lines 35-50](app/app.py#L35-L50) for classify_exercise().

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/milosperic/fitscan-ai
- **Main user flow:**
  1. User uploads a photo of their exercise
  2. User types a free-text description (age, weight, duration, etc.)
  3. CV block classifies the exercise type from the image
  4. NLP block extracts parameters and injects workout_type from CV
  5. ML block predicts calories burned
  6. NLP block generates personalized coaching explanation
  7. All results displayed: CV label and confidence, extracted JSON, calorie prediction, explanation

- **Key files on HuggingFace Space:** app.py, requirements.txt, calories_model.pkl, label_encoder_workout.pkl, label_encoder_gender.pkl, cv_class_names.pkl, best_cv_model.keras
- **OPENAI_API_KEY** set as HuggingFace Secret

## 4. Execution Instructions

**Environment setup:**

```bash
conda create -n fitscan python=3.10 -y
conda activate fitscan
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn xgboost pillow gradio openai
brew install libomp  # macOS only, required for XGBoost
```

**Data setup:**

1. Download Gym Members Exercise Dataset from Kaggle and place the CSV in ml/
2. Download Workout Exercise Images from Kaggle and extract to cv/images/

**Training ML block:**

```bash
cd ml/
jupyter notebook train_model.ipynb
# Run all cells
# Output: calories_model.pkl, label_encoder_workout.pkl, label_encoder_gender.pkl
```

**Training CV block:**

```bash
cd cv/
jupyter notebook train_cv.ipynb
# Run all cells
# Output: best_cv_model.keras, cv_class_names.pkl
# Note: Training takes approximately 60 minutes on CPU
```

**Run app locally:**

```bash
cd app/
cp ../ml/calories_model.pkl .
cp ../ml/label_encoder_workout.pkl .
cp ../ml/label_encoder_gender.pkl .
cp ../cv/best_cv_model.keras .
cp ../cv/cv_class_names.pkl .
export OPENAI_API_KEY=your_key_here
python app.py
# Open http://127.0.0.1:7860
```

**Reproducibility notes:**

- All notebooks use random_state=42
- Versions: TensorFlow 2.21.0, scikit-learn 1.7.2, XGBoost 3.2.0, Python 3.10

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality: NLP block used for both parameter extraction (structured JSON output) and personalized explanation generation, with two distinct prompt strategies and a documented iteration
- [x] More than two data sources used with clear added value: Kaggle tabular dataset (ML), Kaggle image dataset (CV), runtime user text (NLP)
- [x] A core section is done exceptionally well: CV block achieves 99.7% test accuracy; the preprocessing bug discovery (72% to 99.7% improvement from preprocess_input fix) is documented as a key learning
- [x] Extended evaluation: Both ML and CV blocks include confusion matrix, residual analysis, feature importance plots, and training history visualization

**Evidence summary:**

- CV: 99.7% test accuracy on 2,099 unseen images with only 7 errors
- ML: R2=0.987 with MAE of 24 calories on a target range of 303 to 1783 kcal
- All three blocks are technically and conceptually integrated: CV output feeds NLP prompt, NLP output feeds ML features, ML output feeds NLP explanation
