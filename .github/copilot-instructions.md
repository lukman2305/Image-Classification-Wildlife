# AI Agent Instructions - UTM SAIA 2133 Endangered Wildlife Identification

## Project Context
**Course:** SAIA 2133: Computer Vision - Universiti Teknologi Malaysia (UTM)  
**Project:** Endangered Wildlife Identification using Deep Learning  
**Dataset:** [Danger of Extinction Animal Image Set](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)

## Project Requirements (50 marks rubric)

### 1. Dataset & EDA (8 marks)
- Use **minimum 3 animal classes** from Kaggle dataset
- Implement comprehensive EDA showing:
  - Class distribution (bar charts)
  - Sample images from each class (grid visualization)
  - Image dimension statistics
  - Dataset split proportions

### 2. Preprocessing & Augmentation (7 marks)
- **Standardization:** Resize all images to consistent dimensions (e.g., 224×224)
- **Normalization:** Scale pixel values to [0,1] or use ImageNet stats
- **Augmentation:** Implement rotation, horizontal/vertical flipping, brightness adjustment
- Use `ImageDataGenerator` (Keras) or `transforms` (PyTorch)

### 3. Model Development (10 marks)
**Model A - Custom CNN:**
- Lightweight architecture designed by student
- 3-5 convolutional layers with MaxPooling
- Dropout for regularization
- Fully connected classifier head

**Model B - Transfer Learning:**
- Use **ResNet50** or **EfficientNet** pre-trained on ImageNet
- Freeze base layers initially
- Add custom classification head
- Fine-tune if needed

### 4. Training & Evaluation (13 marks)
- **Data Split:** Train (70%), Validation (15%), Test (15%)
- **Comparison:** Performance, training time, model complexity
- **Metrics:** 
  - Overall accuracy
  - Per-class Precision, Recall, F1-score
  - Confusion matrix visualization
- **Training History:** Plot loss and accuracy curves

### 5. Interactive Demo (Rubric Part B)
- Single image prediction function
- Visual display: original image + predicted class + confidence score
- Handle various image formats

## Code Structure

```
Image-Classification-Wildlife/
├── data/
│   └── danger-of-extinction/     # Kaggle dataset (gitignored)
├── notebooks/
│   └── wildlife_classification.ipynb  # Main implementation
├── src/
│   ├── data_loader.py           # Dataset loading utilities
│   ├── models.py                # Custom CNN and transfer learning
│   ├── train.py                 # Training pipeline
│   └── evaluate.py              # Evaluation and metrics
├── predict_demo.py              # Interactive prediction script
├── report/
│   └── project_report.md        # Report outline
├── requirements.txt
└── README.md
```

## Implementation Patterns

### Data Loading (Keras Example)
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_directory(
    'data/danger-of-extinction/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### Custom CNN Architecture
```python
from tensorflow.keras import layers, models

def create_custom_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### Transfer Learning (ResNet50)
```python
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Evaluation Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
```

## Report Structure (3-4 pages)

1. **Introduction** (0.5 page)
   - Wildlife conservation importance
   - Project objectives

2. **Methodology** (1.5 pages)
   - Dataset description with class distribution
   - Preprocessing pipeline
   - Model architectures (diagram + explanation)

3. **Results** (1 page)
   - Performance comparison table
   - Confusion matrices
   - Training curves

4. **Ethical & Practical Reflections** (0.5 page)
   - Wildlife conservation applications
   - Model limitations and biases
   - Real-world deployment considerations

5. **Conclusion** (0.5 page)

## Development Workflow

1. **Setup:** Download Kaggle dataset, install dependencies
2. **EDA:** Explore data, visualize samples
3. **Build Custom CNN:** Train and evaluate
4. **Build Transfer Model:** Train and evaluate
5. **Compare:** Generate comparison metrics
6. **Create Demo:** Implement prediction function
7. **Document:** Fill report with results

## Key Libraries
```
tensorflow>=2.10.0
keras>=2.10.0
opencv-python>=4.6.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.1.0
numpy>=1.23.0
pandas>=1.5.0
pillow>=9.0.0
```

## Important Notes
- **Dataset Path:** Place Kaggle data in `data/danger-of-extinction/`
- **Reproducibility:** Set random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`)
- **Class Selection:** Choose 3+ distinct, well-represented classes
- **Training Time:** Custom CNN faster, Transfer Learning more accurate
- **Rubric Focus:** Ensure all deliverables match marking criteria exactly
