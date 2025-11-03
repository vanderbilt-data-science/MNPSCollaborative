# How to Use the Google Colab Notebook

## 🚀 Quick Start

### 1. Download the Notebook
[Download MNPS_Three_Pass_Classification.ipynb](computer:///mnt/user-data/outputs/MNPS_Three_Pass_Classification.ipynb)

### 2. Upload to Google Colab
1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Select the downloaded `.ipynb` file
4. Wait for it to load

### 3. Set Your API Key
In the "API Key Setup" cell, replace:
```python
ANTHROPIC_API_KEY = "your-api-key-here"
```

With your actual Anthropic API key.

**More secure option:** Use Colab secrets
1. Click the 🔑 key icon in the left sidebar
2. Add a secret named `ANTHROPIC_API_KEY`
3. Uncomment these lines in the notebook:
```python
from google.colab import userdata
ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
```

### 4. Upload Your MNPS Files
When you reach the "Upload MNPS Resources" cell, upload these 4 files:
- MNPS Roles.csv
- MNPS KSACs.csv
- Ground Truth Masterfile.csv
- Sample JDs.csv

### 5. Run the Notebook
Click **Runtime → Run all** or run cells one by one using Shift+Enter

---

## 📋 What the Notebook Does

### Pass 1: Initial Classification
- Loads all 63 MNPS roles and their KSACs
- LLM classifies each job with full context
- **You can modify the Pass 1 prompt for iterations**

### Pass 2: Self-Consistency Check
- LLM reviews its own classification
- Catches justification mismatches (e.g., "says Coordinator but classified as Manager")
- **You can modify the Pass 2 prompt for iterations**

### Pass 3: Validation Pipeline
- Applies 6 role-specific rules
- Enforces NO minor sub-grouping
- Ensures field consistency
- **Do NOT modify - keep stable between iterations**

---

## 🧪 Testing Flow

The notebook includes:

1. **Test on 5 jobs first** - Verify everything works
2. **Run on 43-job test set** - Full evaluation
3. **Save results** - Download CSV with all classifications
4. **Evaluate accuracy** - Compare against ground truth
5. **Analyze corrections** - See which validations fired

---

## 🔄 Iteration Workflow

To refine your prompts:

1. **Find the prompt cells:**
   - `PASS1_PROMPT_TEMPLATE` (in Pass 1 section)
   - `PASS2_PROMPT_TEMPLATE` (in Pass 2 section)

2. **Modify the prompts:**
   - Update classification guidance
   - Add role definitions
   - Adjust minor sub-group criteria
   - Change consistency check approach

3. **Re-run from "Run on 43-Job Test Set"**

4. **Compare results:**
   - Check accuracy metrics
   - Review error details
   - Analyze correction frequency

5. **Iterate and improve**

---

## 📊 Expected Results

After running on the 43-job test set, you should see:

```
EVALUATION RESULTS
==================================================
Total jobs evaluated: 43
Major role accuracy: 35-36/43 (81-84%)
Minor sub-group accuracy: 42/43 (97.7%)
Both correct: 33-34/43 (77-79%)
==================================================
```

---

## ⚠️ Important Notes

### API Usage
- **2 API calls per job** (Pass 1 + Pass 2)
- **43 jobs = 86 API calls total**
- **Cost:** ~$2-5 for full test set
- **Time:** ~5-10 seconds per job

### DO Modify
✅ Pass 1 prompt template  
✅ Pass 2 prompt template  
✅ Number of test jobs  

### DO NOT Modify
❌ Pass 3 validation code  
❌ Helper functions  
❌ Evaluation logic  

### Between Iterations
- Only change the two prompt templates
- Keep everything else stable
- Always re-run full test set
- Compare to previous iteration

---

## 🐛 Troubleshooting

### "Invalid API key"
- Check that you set `ANTHROPIC_API_KEY` correctly
- Make sure there are no quotes or spaces

### "File not found"
- Upload all 4 MNPS resource files
- Make sure filenames match exactly

### "JSON parsing error"
- LLM may have returned malformed JSON
- Check the error message
- May need to adjust prompt to be more explicit about JSON format

### "Rate limit exceeded"
- Anthropic API has rate limits
- Add delays between jobs if needed
- Consider upgrading your API plan

---

## 💾 Output Files

The notebook generates:

1. **three_pass_results.csv** - All classifications with:
   - Job description name
   - Major role group
   - Minor sub-group
   - New job title
   - Grouping justification
   - Corrections applied
   - Processing time

2. **Downloaded to your computer** automatically

---

## 🎯 Success Metrics

### Must Have
- [ ] Notebook runs without errors
- [ ] All 43 jobs process successfully
- [ ] Results file downloads
- [ ] Both correct ≥ 75%

### Target Performance
- [ ] Major role ≥ 81%
- [ ] Minor sub-group ≥ 97%
- [ ] Both correct ≥ 77%

---

## 📚 Additional Resources

For more details, see these documents:
- **THREE_PASS_README.md** - Overview and navigation
- **THREE_PASS_QUICK_REFERENCE.md** - One-page summary
- **THREE_PASS_CLASSIFICATION_SYSTEM.md** - Complete technical spec
- **THREE_PASS_IMPLEMENTATION_GUIDE.md** - Detailed code walkthrough

---

## 🚀 Next Steps

After your first successful run:

1. **Review the errors** - See which jobs were misclassified
2. **Check correction frequency** - See which validations fired most
3. **Refine Pass 1 prompt** - Add guidance for problematic roles
4. **Refine Pass 2 prompt** - Improve consistency checking
5. **Re-run and compare** - Measure improvement
6. **Iterate until satisfied**

---

## ✅ Quick Checklist

Before running:
- [ ] Uploaded notebook to Google Colab
- [ ] Set API key
- [ ] Uploaded all 4 MNPS resource files
- [ ] Verified files loaded correctly

After running:
- [ ] Check accuracy metrics
- [ ] Review error details
- [ ] Analyze correction frequency
- [ ] Download results CSV
- [ ] Document findings

For next iteration:
- [ ] Modify prompts based on errors
- [ ] Re-run test set
- [ ] Compare to baseline
- [ ] Track improvements

---

**You're ready to go! Upload the notebook to Google Colab and start classifying.** 🎯
