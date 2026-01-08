# Alzheimer's Clinical Decision Support System

LightGBM-based diagnosis prediction using comprehensive biomarker panel. Deployed on Vercel.

## 🧠 Your ML Model Powers Everything

This system uses **YOUR trained LightGBM model** as the brain for all diagnoses:
- Trains on YOUR 438-patient dataset
- Uses YOUR exact feature engineering
- Makes predictions using YOUR model's weights
- Zero synthetic data, 100% real ML

---

## 📁 Project Structure

```
your-project/
├── index.html              # Frontend UI
├── api/
│   └── predict.py          # Serverless function (trains & predicts)
├── dataset.xlsx            # YOUR 438-patient dataset
├── vercel.json             # Vercel configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🚀 Deploy to Vercel

### Step 1: Prepare Your Repository

1. **Create a new folder** for your project
2. **Add these files**:
   - `index.html` (frontend)
   - `vercel.json` (config)
   - `requirements.txt` (dependencies)
   - Create `api/` folder
   - Add `predict.py` inside `api/`
   - **IMPORTANT: Add your `dataset.xlsx`** in the root folder

Your folder should look like:
```
alzheimers-diagnosis/
├── index.html
├── dataset.xlsx          ← YOUR DATASET HERE
├── api/
│   └── predict.py
├── vercel.json
└── requirements.txt
```

### Step 2: Push to GitHub

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/alzheimers-diagnosis.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. Import your GitHub repository
4. Vercel will auto-detect the configuration
5. Click **"Deploy"**

**That's it!** Your app will be live at `https://your-project.vercel.app`

---

## ⚙️ How It Works

### First Request (30-60 seconds):
1. User submits patient data
2. Vercel function loads `dataset.xlsx`
3. Trains YOUR LightGBM model (80/20 split, SMOTE, feature engineering)
4. Caches the trained model in memory
5. Makes prediction
6. Returns diagnosis + probabilities

### Subsequent Requests (<2 seconds):
1. User submits patient data
2. Uses cached model (no retraining)
3. Makes instant prediction
4. Returns results

**Note**: Model stays cached for ~5 minutes of inactivity. After that, next request will retrain (30-60s).

---

## 🧪 Testing Locally (Optional)

If you want to test before deploying:

```bash
# Install Vercel CLI
npm install -g vercel

# Run locally
vercel dev
```

Open `http://localhost:3000` in your browser.

---

## 📊 Input Requirements

### Biomarkers (7):
- Aβ40, Aβ42, pTau181, pTau217, GFAP, NfL, Aβ Ratio

### Clinical Markers (5):
- MMSE Score (0-30)
- DHA, Formic Acid, Lactoferrin, NTP

### Cognitive Assessment (6):
- DXMPTR1-6 (Yes=1, Maybe=0.5, No=0)

---

## 🔒 Security Note

Your `dataset.xlsx` will be in a **public GitHub repository**. If your dataset contains sensitive patient information:

**Option A**: Anonymize data before uploading
**Option B**: Use a private GitHub repository (requires Vercel Pro plan)
**Option C**: Remove identifiable information from the dataset

---

## 🐛 Troubleshooting

### "dataset.xlsx not found"
- Make sure `dataset.xlsx` is in the **root folder** (not in `api/`)
- Check the filename is exactly `dataset.xlsx` (case-sensitive)

### "500 Internal Server Error"
- Check Vercel function logs: Project → Deployments → Click deployment → Functions tab
- Common issue: Missing columns in dataset

### First request times out
- Vercel free tier has 10-second timeout for functions
- If training takes >10s, you'll need to pre-train the model (see Option A in original message)

### Model keeps retraining
- This is normal - Vercel functions are stateless
- Model cache lasts ~5 minutes of inactivity
- Each cold start retrains the model

---

## 📝 License

For clinical research and decision support purposes only. Not for automated diagnosis.

---

## 🎯 Next Steps After Deployment

1. Test with sample patient data
2. Share the Vercel URL with your team
3. Monitor prediction accuracy
4. Collect feedback

Your live app: `https://YOUR-PROJECT.vercel.app`
