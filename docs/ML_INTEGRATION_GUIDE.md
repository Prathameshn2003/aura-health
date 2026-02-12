# 🔗 ML Model Integration Guide for Beginners

This guide explains **step-by-step** how to connect your Machine Learning models to this frontend application.

---

## 📚 Table of Contents
1. [Understanding the Current Setup](#1-understanding-the-current-setup)
2. [How the Frontend Currently Works](#2-how-the-frontend-currently-works)
3. [Option A: Replace TypeScript Logic with Real ML Model](#3-option-a-replace-typescript-logic-with-real-ml-model)
4. [Option B: Use Hugging Face (Recommended for Beginners)](#4-option-b-use-hugging-face-recommended)
5. [Option C: Self-Hosted Python API](#5-option-c-self-hosted-python-api)
6. [Code Flow Explanation](#6-code-flow-explanation)

---

## 1. Understanding the Current Setup

### What We Have Now

Your app currently uses **TypeScript-based scoring logic** that mimics ML predictions. This is located in:

```
📁 src/lib/ml-predictions.ts
```

This file contains:
- `predictPCOS()` - Calculates PCOS risk based on user inputs
- `predictMenopause()` - Determines menopause stage
- `predictNextCycle()` - Predicts next period date

### Why TypeScript Instead of Real ML?

Currently, the ML logic is "translated" from Python to TypeScript so it runs **directly in the browser** without needing a server. This is fast but doesn't use your actual trained `.pkl` or `.h5` model files.

---

## 2. How the Frontend Currently Works

### The Flow (Step by Step)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User fills     │ --> │  Form submits    │ --> │  predictPCOS()  │
│  assessment     │     │  data            │     │  function runs  │
│  form           │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User sees      │ <-- │  Component       │ <-- │  Returns result │
│  results on     │     │  displays        │     │  object with    │
│  screen         │     │  results         │     │  risk & advice  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Files Involved

| File | Purpose |
|------|---------|
| `src/pages/PCOSModule.tsx` | PCOS assessment page |
| `src/pages/MenopauseModule.tsx` | Menopause assessment page |
| `src/components/health/PCOSAssessmentForm.tsx` | Collects user data |
| `src/components/health/MenopauseAssessmentForm.tsx` | Collects user data |
| `src/lib/ml-predictions.ts` | **THE BRAIN** - all prediction logic |
| `src/components/health/PCOSResultsDisplay.tsx` | Shows PCOS results |
| `src/components/health/MenopauseResultsDisplay.tsx` | Shows menopause results |

### Code That Connects Form to ML

In `src/pages/PCOSModule.tsx` (line ~40):

```typescript
// When user submits the form:
const handleAssessmentSubmit = async (data: PCOSInputData) => {
  // THIS LINE calls the ML prediction function! 👇
  const result = predictPCOS(data);
  
  // Store result and show results page
  setPcosResult(result);
  setCurrentStep("results");
  
  // Save to database
  await supabase.from("health_assessments").insert({...});
};
```

---

## 3. Option A: Replace TypeScript Logic with Real ML Model

### When to Use This

If you want to use your actual trained `.pkl` model files (RandomForest, XGBoost, etc.)

### Steps

#### Step 1: Create a Backend API (Edge Function)

Create file: `supabase/functions/ml-predict/index.ts`

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { model_type, input_data } = await req.json();
    
    // Call your external ML API here
    // This is where you connect to Hugging Face, AWS, or your own server
    const ML_API_URL = Deno.env.get("ML_API_URL");
    
    const response = await fetch(`${ML_API_URL}/predict/${model_type}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input_data),
    });
    
    const result = await response.json();
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
```

#### Step 2: Update Frontend to Call API

Modify `src/lib/ml-predictions.ts`:

```typescript
import { supabase } from "@/integrations/supabase/client";

// NEW: API-based prediction
export async function predictPCOSFromAPI(data: PCOSInputData): Promise<PCOSResult> {
  const { data: result, error } = await supabase.functions.invoke('ml-predict', {
    body: {
      model_type: 'pcos',
      input_data: data
    }
  });
  
  if (error) throw error;
  return result;
}
```

#### Step 3: Update PCOSModule.tsx

```typescript
// Change from:
const result = predictPCOS(data);

// To:
const result = await predictPCOSFromAPI(data);
```

---

## 4. Option B: Use Hugging Face (Recommended)

### Why Hugging Face?

- ✅ Free tier available
- ✅ Easy to deploy ML models
- ✅ No server management
- ✅ Supports scikit-learn, PyTorch, TensorFlow

### Step 1: Create Hugging Face Account

1. Go to https://huggingface.co
2. Click "Sign Up"
3. Create a new account

### Step 2: Create a Space

1. Click "New" → "Space"
2. Choose "Gradio" or "Docker" template
3. Name it: `your-username/pcos-predictor`

### Step 3: Upload Your Model Files

Create these files in your Space:

**requirements.txt:**
```
fastapi
uvicorn
scikit-learn
joblib
numpy
pandas
```

**app.py:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your trained models
pcos_model = joblib.load("pcos_model.pkl")
menopause_model = joblib.load("menopause_model.pkl")

class PCOSInput(BaseModel):
    age: int
    bmi: float
    cycle_length: int
    cycle_regular: bool
    weight_gain: bool
    hair_growth: bool
    skin_darkening: bool
    hair_loss: bool
    pimples: bool
    fast_food: bool
    regular_exercise: bool
    follicle_left: int
    follicle_right: int
    endometrium: float

@app.post("/predict/pcos")
def predict_pcos(data: PCOSInput):
    # Prepare features array (same order as training)
    features = np.array([[
        data.age,
        data.bmi,
        data.cycle_length,
        1 if data.cycle_regular else 0,
        1 if data.weight_gain else 0,
        1 if data.hair_growth else 0,
        1 if data.skin_darkening else 0,
        1 if data.hair_loss else 0,
        1 if data.pimples else 0,
        1 if data.fast_food else 0,
        1 if data.regular_exercise else 0,
        data.follicle_left,
        data.follicle_right,
        data.endometrium
    ]])
    
    # Get prediction
    prediction = pcos_model.predict(features)[0]
    probability = pcos_model.predict_proba(features)[0]
    
    return {
        "has_pcos": bool(prediction),
        "risk_percentage": round(probability[1] * 100, 2),
        "severity": "high" if probability[1] > 0.7 else "medium" if probability[1] > 0.4 else "low"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
```

### Step 4: Upload Your .pkl Files

Upload `pcos_model.pkl` and `menopause_model.pkl` to the Space.

### Step 5: Get Your API URL

Once deployed, your API URL will be:
```
https://your-username-pcos-predictor.hf.space
```

### Step 6: Add Secret to Lovable

1. Go to Supabase Dashboard → Settings → Secrets
2. Add: `ML_API_URL` = `https://your-username-pcos-predictor.hf.space`

---

## 5. Option C: Self-Hosted Python API

### When to Use

If you have your own server or VPS

### Basic Flask API

**app.py:**
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models at startup
pcos_model = joblib.load("models/pcos_model.pkl")
menopause_model = joblib.load("models/menopause_model.pkl")

@app.route("/predict/pcos", methods=["POST"])
def predict_pcos():
    data = request.json
    
    # Convert to features array
    features = np.array([[
        data["age"],
        data["bmi"],
        data["cycle_length"],
        # ... other features
    ]])
    
    prediction = pcos_model.predict(features)[0]
    probability = pcos_model.predict_proba(features)[0]
    
    return jsonify({
        "has_pcos": bool(prediction),
        "risk_percentage": round(float(probability[1]) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### Deploy on Railway/Render/Heroku

1. Create account on Railway.app or Render.com
2. Connect your GitHub repo
3. Deploy the Flask API
4. Get the URL and add to Lovable secrets

---

## 6. Code Flow Explanation

### Visual Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │ PCOSModule.tsx  │ ◄─── User lands on PCOS page             │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────┐                                   │
│  │ PCOSAssessmentForm.tsx  │ ◄─── User fills form              │
│  │ - Age, BMI, symptoms    │                                   │
│  │ - Lifestyle questions   │                                   │
│  └────────┬────────────────┘                                   │
│           │ onSubmit(data)                                      │
│           ▼                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              handleAssessmentSubmit(data)              │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  const result = predictPCOS(data);  ◄── ML HERE  │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────┐                                   │
│  │ PCOSResultsDisplay.tsx  │ ◄─── Shows risk, recommendations  │
│  │ - RiskGauge component   │                                   │
│  │ - ScoreBreakdown        │                                   │
│  │ - Diet/Exercise tips    │                                   │
│  └─────────────────────────┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (If using real ML API)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Supabase Edge Function)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  supabase/functions/ml-predict/index.ts                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Receives data from frontend                            │  │
│  │  - Calls external ML API (Hugging Face, AWS, etc.)       │  │
│  │  - Returns prediction result                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL ML SERVICE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hugging Face Space / AWS SageMaker / Google Cloud AI          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Loads your trained .pkl / .h5 model                   │  │
│  │  - Processes input features                               │  │
│  │  - Returns: { has_pcos: true, risk: 65, severity: "med" }│  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 Summary: What You Need To Do

### Minimum Steps (Hugging Face Method)

1. **Train your model** in Python → Save as `.pkl`
2. **Create Hugging Face Space** → Upload model + API code
3. **Add API URL secret** in Supabase
4. **Create edge function** to call Hugging Face
5. **Update frontend** to use API instead of local function

### Files You'll Modify

| File | Change |
|------|--------|
| `supabase/functions/ml-predict/index.ts` | NEW - API caller |
| `src/lib/ml-predictions.ts` | Add API-based functions |
| `src/pages/PCOSModule.tsx` | Use new async function |
| `src/pages/MenopauseModule.tsx` | Use new async function |

---

## 🆘 Need Help?

1. **Model not loading?** Check if `.pkl` file format matches your scikit-learn version
2. **CORS errors?** Make sure your API has CORS headers
3. **Slow predictions?** Consider using a smaller model or caching results
4. **Feature mismatch?** Ensure frontend sends features in same order as training data

---

*Created for NaariCare Health App*
