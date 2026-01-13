# Quick Start Guide - Endangered Wildlife Classification

## ⚡ Fast Track (5 Steps)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Download Dataset
```bash
# Option A: Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d brsdincer/danger-of-extinction-animal-image-set
unzip danger-of-extinction-animal-image-set.zip -d data/danger-of-extinction/

# Option B: Manual - Download from Kaggle website
# Extract to: data/danger-of-extinction/
```

### 3️⃣ Run the Notebook
```bash
jupyter notebook notebooks/wildlife_classification.ipynb
```
**Then**: Run All Cells (Cell → Run All)

### 4️⃣ Review Results
Check `results/` folder for:
- Training curves
- Confusion matrices
- Model comparison charts
- Classification reports

### 5️⃣ Test Prediction
```bash
python predict_demo.py --image path/to/test/image.jpg --model resnet50
```

---

## 📋 Expected Runtime

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| EDA | 2-5 min | 2-5 min |
| Custom CNN Training | 20-40 min | 5-10 min |
| ResNet50 Training | 40-60 min | 10-20 min |
| Evaluation | 5-10 min | 2-5 min |
| **Total** | **~1.5 hours** | **~30 min** |

---

## 🎯 Minimal Test Run

Want to test quickly? Reduce epochs in notebook:
```python
EPOCHS = 5  # Instead of 30
```

This will complete in ~15 minutes (CPU) or ~5 minutes (GPU).

---

## ✅ Checklist Before Submission

- [ ] Dataset downloaded and extracted
- [ ] All notebook cells executed without errors
- [ ] Both models trained successfully
- [ ] Results generated in `results/` folder
- [ ] Models saved in `models/` folder
- [ ] Report filled with your results
- [ ] Prediction demo tested with sample image
- [ ] All figures inserted in report
- [ ] References updated
- [ ] Student info filled in report

---

## 🆘 Need Help?

**Dataset Issues:**
- Ensure dataset is in: `data/danger-of-extinction/`
- Each class should be in its own subdirectory
- Supported formats: .jpg, .jpeg, .png

**Training Issues:**
- Out of memory? Reduce `BATCH_SIZE` to 16 or 8
- Slow training? Reduce `EPOCHS` or use smaller dataset subset
- GPU not detected? Continue with CPU (just slower)

**Results Issues:**
- Missing plots? Check `results/` folder was created
- Low accuracy? Ensure dataset has enough images per class
- Model not loading? Check `models/` folder for .h5 files

**Report Issues:**
- Copy metrics from notebook output
- Insert figures from `results/` folder
- Fill placeholders marked with [X] or [Your ...]

---

## 📊 What Gets Graded?

According to UTM rubric (50 marks):

1. **Dataset & EDA (8 marks)**
   - ✅ Use 3+ classes
   - ✅ Show class distribution
   - ✅ Display sample images

2. **Preprocessing (7 marks)**
   - ✅ Resize to standard size
   - ✅ Normalize pixels
   - ✅ Implement augmentation

3. **Models (10 marks)**
   - ✅ Custom CNN architecture
   - ✅ Transfer learning (ResNet50)

4. **Training & Evaluation (13 marks)**
   - ✅ Train/Val/Test split
   - ✅ Performance comparison
   - ✅ Accuracy, Precision, Recall, F1
   - ✅ Confusion matrices

5. **Demo (12 marks)**
   - ✅ Interactive prediction
   - ✅ Visual display

---

## 💡 Pro Tips

1. **Save Often**: The notebook auto-saves, but manually save after important cells
2. **Monitor Training**: Watch training curves - if not improving, stop early
3. **Compare Models**: ResNet50 usually wins on accuracy, Custom CNN on speed
4. **Document Everything**: Take screenshots of results for your report
5. **Test Prediction**: Try various images to show model robustness

---

## 🚀 Ready to Go?

Start with: `jupyter notebook notebooks/wildlife_classification.ipynb`

**Good luck! 🐾**
