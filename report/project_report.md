# Endangered Wildlife Image Classification - Project Report
## SAIA 2133: Computer Vision - Universiti Teknologi Malaysia (UTM)

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** [Date]  
**Course:** SAIA 2133 - Computer Vision  
**Instructor:** [Instructor Name]

---

## 1. Introduction (0.5 page)

### 1.1 Background and Motivation
Wildlife conservation is a critical global challenge, with many species facing extinction due to habitat loss, poaching, and climate change. Traditional methods of wildlife monitoring are time-consuming and labor-intensive, often relying on manual identification by experts. Computer vision and deep learning offer promising solutions for automated wildlife identification, enabling faster and more accurate monitoring of endangered species.

**[Insert: Brief statistics about endangered species and conservation efforts]**

### 1.2 Project Objectives
This project aims to develop an automated image classification system for identifying endangered wildlife species using deep learning techniques. Specifically, the project will:

1. Implement a comprehensive image classification pipeline for wildlife identification
2. Compare two distinct deep learning approaches:
   - **Model A**: A custom lightweight CNN designed from scratch
   - **Model B**: Transfer learning using pre-trained ResNet50
3. Evaluate model performance using standard classification metrics
4. Provide practical insights for wildlife conservation applications

### 1.3 Dataset
**Source:** [Danger of Extinction Animal Image Set](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)  
**Number of Classes:** [X]  
**Classes Used:** [List your 3+ animal classes]  
**Total Images:** [Total number]

**[INSERT: Class distribution bar chart from notebook - Figure 1]**

---

## 2. Methodology (1.5 pages)

### 2.1 Dataset Preparation and Exploratory Data Analysis

#### 2.1.1 Data Collection
The dataset was obtained from Kaggle and contains images of endangered wildlife species. The images vary in resolution, lighting conditions, and backgrounds, reflecting real-world wildlife photography scenarios.

**Dataset Statistics:**
- Total Images: [X]
- Training Set: [X] (70%)
- Validation Set: [X] (15%)
- Test Set: [X] (15%)

**[INSERT: Sample images grid from notebook - Figure 2]**

#### 2.1.2 Data Characteristics
**[INSERT: Image dimensions distribution plots - Figure 3]**

Analysis of image dimensions revealed:
- Width Range: [min-max] pixels
- Height Range: [min-max] pixels
- Average Dimensions: [WxH] pixels
- All images were standardized to 224×224 pixels for model input

**Class Distribution Analysis:**
**[INSERT: Pie chart and statistics]**

- Class balance ratio: [X:Y:Z]
- Imbalance factor: [X]x
- **Note:** [Comment on whether class imbalance was significant and how it was addressed]

### 2.2 Data Preprocessing and Augmentation

To improve model robustness and generalization, comprehensive preprocessing and augmentation techniques were applied:

#### 2.2.1 Preprocessing
- **Standardization**: All images resized to 224×224 pixels
- **Normalization**: Pixel values scaled to [0, 1] range
- **Color Space**: RGB format maintained for all images

#### 2.2.2 Data Augmentation
Applied to training data only to prevent overfitting:
- Rotation: Random rotation up to ±20 degrees
- Horizontal Flip: 50% probability
- Vertical Flip: 50% probability
- Brightness Adjustment: Range [0.8, 1.2]
- Width/Height Shift: Up to 20% translation
- Zoom: Random zoom up to 20%

**[INSERT: Augmentation examples visualization - Figure 4]**

### 2.3 Model Architectures

#### 2.3.1 Model A: Custom Lightweight CNN
A custom Convolutional Neural Network was designed with the following architecture:

**Architecture Diagram:**
```
Input (224×224×3)
    ↓
Conv2D (32 filters, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv2D (64 filters, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv2D (128 filters, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
Conv2D (256 filters, 3×3) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
GlobalAveragePooling2D
    ↓
Dense (256) → BatchNorm → Dropout(0.5)
    ↓
Dense (128) → Dropout(0.4)
    ↓
Dense (num_classes, softmax)
```

**Model Specifications:**
- Total Parameters: [X]
- Trainable Parameters: [X]
- Model Size: [X] MB
- Training Time: [X] minutes

**Design Rationale:**
- Progressive filter increase (32→64→128→256) for hierarchical feature learning
- BatchNormalization for training stability
- Dropout layers for regularization
- GlobalAveragePooling to reduce parameters
- Lightweight design for efficient deployment

#### 2.3.2 Model B: Transfer Learning with ResNet50
Transfer learning leverages pre-trained weights from ImageNet:

**Architecture:**
```
ResNet50 Base (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense (512) → Dropout(0.5) → BatchNorm
    ↓
Dense (256) → Dropout(0.4)
    ↓
Dense (num_classes, softmax)
```

**Model Specifications:**
- Total Parameters: [X]
- Trainable Parameters: [X] (custom head only)
- Non-Trainable Parameters: [X] (frozen ResNet50 base)
- Model Size: [X] MB
- Training Time: [X] minutes

**Design Rationale:**
- ResNet50 pre-trained on ImageNet provides robust feature extraction
- Frozen base layers preserve learned features
- Custom classification head adapted to wildlife dataset
- Fine-tuning option available for further optimization

### 2.4 Training Configuration

**Hyperparameters:**
- Optimizer: Adam
- Learning Rate: 0.001 (with ReduceLROnPlateau)
- Batch Size: 32
- Epochs: 30 (with EarlyStopping)
- Loss Function: Categorical Crossentropy

**Callbacks:**
- EarlyStopping: Patience=7, monitor='val_loss'
- ModelCheckpoint: Save best model based on 'val_accuracy'
- ReduceLROnPlateau: Factor=0.5, patience=3
- CSVLogger: Record training metrics

**Training Strategy:**
- Train on augmented data
- Validate on original data (no augmentation)
- Monitor validation metrics to prevent overfitting
- Restore best weights upon early stopping

---

## 3. Results (1 page)

### 3.1 Training Performance

**[INSERT: Training curves for both models - Figure 5]**

**Custom CNN Training:**
- Final Training Accuracy: [X]%
- Final Validation Accuracy: [X]%
- Best Validation Accuracy: [X]%
- Training Time: [X] minutes
- Convergence: Epoch [X]

**ResNet50 Transfer Learning:**
- Final Training Accuracy: [X]%
- Final Validation Accuracy: [X]%
- Best Validation Accuracy: [X]%
- Training Time: [X] minutes
- Convergence: Epoch [X]

**Observations:**
- [Comment on convergence speed]
- [Comment on overfitting/underfitting]
- [Comment on training stability]

### 3.2 Test Set Evaluation

**Performance Metrics Summary:**

| Metric | Custom CNN | ResNet50 Transfer | Winner |
|--------|-----------|-------------------|---------|
| **Accuracy** | [X]% | [X]% | [Model] |
| **Precision** | [X] | [X] | [Model] |
| **Recall** | [X] | [X] | [Model] |
| **F1-Score** | [X] | [X] | [Model] |

**Per-Class Performance (Custom CNN):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| [Class 1] | [X] | [X] | [X] | [X] |
| [Class 2] | [X] | [X] | [X] | [X] |
| [Class 3] | [X] | [X] | [X] | [X] |

**Per-Class Performance (ResNet50):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| [Class 1] | [X] | [X] | [X] | [X] |
| [Class 2] | [X] | [X] | [X] | [X] |
| [Class 3] | [X] | [X] | [X] | [X] |

### 3.3 Confusion Matrices

**[INSERT: Side-by-side confusion matrices - Figure 6]**

**Analysis:**
- Most confused classes: [Class X] ↔ [Class Y]
- Possible reasons: [Similar visual features, habitat overlap, etc.]
- Classes with highest accuracy: [List]
- Classes requiring improvement: [List]

### 3.4 Model Comparison

**[INSERT: Model comparison charts - Figure 7]**

**Key Findings:**
1. **Accuracy**: [Model] achieved [X]% higher accuracy
2. **Training Efficiency**: Custom CNN trains [X]x faster
3. **Model Complexity**: Custom CNN has [X]x fewer parameters
4. **Deployment Considerations**: [Analysis]

**Trade-off Analysis:**
- **Custom CNN**: Faster, lighter, good baseline performance
- **ResNet50**: Higher accuracy, leverages pre-trained features, larger model

**Recommendation:** [Based on use case - conservation monitoring, mobile deployment, etc.]

---

## 4. Ethical & Practical Reflections (0.5 page)

### 4.1 Wildlife Conservation Applications

**Practical Use Cases:**
1. **Automated Camera Trap Monitoring**: Classify species from camera trap images without manual review
2. **Population Tracking**: Monitor endangered species populations over time
3. **Anti-Poaching Surveillance**: Rapid identification of protected species in surveillance systems
4. **Citizen Science**: Enable public participation in wildlife monitoring through mobile apps

**Benefits:**
- Reduced labor costs for wildlife monitoring
- Faster response times for conservation efforts
- Scalable monitoring across large geographical areas
- Non-invasive species identification

### 4.2 Ethical Considerations

**1. Data Privacy and Security:**
- Wildlife location data must be protected to prevent poaching
- Images may inadvertently capture humans or private property
- Secure storage and access control are essential

**2. Model Bias and Fairness:**
- Dataset may not represent all geographical regions or subspecies
- Underrepresented classes may lead to biased predictions
- Regular updates needed as species distributions change

**3. Human Oversight:**
- AI should assist, not replace, wildlife experts
- Critical decisions (e.g., endangered status) require human verification
- Model uncertainty should be clearly communicated

### 4.3 Limitations and Challenges

**Technical Limitations:**
- Performance degradation on low-quality images
- Limited ability to detect multiple animals in single image
- Struggles with occluded or partially visible subjects
- Requires substantial training data for new species

**Practical Limitations:**
- Computational requirements for deployment in remote areas
- Need for periodic model retraining with new data
- Dependence on image quality and camera equipment
- Limited to visual identification (no behavioral analysis)

**Environmental Considerations:**
- Seasonal variations (e.g., winter coats, breeding plumage)
- Juvenile vs. adult appearance differences
- Similar-looking species in same habitat

### 4.4 Future Improvements

**Model Enhancements:**
- Multi-label classification for images with multiple species
- Object detection to locate and count animals
- Integration of temporal data for behavior analysis
- Transfer learning from larger wildlife datasets

**Deployment Strategies:**
- Model optimization for edge devices (TensorFlow Lite, ONNX)
- Cloud-based inference with offline fallback
- Mobile applications for field researchers
- Integration with existing conservation databases

---

## 5. Conclusion (0.5 page)

### 5.1 Summary of Achievements
This project successfully developed and compared two deep learning approaches for endangered wildlife image classification. Key achievements include:

1. **Comprehensive Pipeline**: Implemented end-to-end pipeline from data preprocessing to model deployment
2. **Model Development**: Created custom CNN and transfer learning models with strong performance
3. **Rigorous Evaluation**: Conducted thorough evaluation using multiple metrics and visualizations
4. **Practical Application**: Developed interactive demo for real-world prediction

### 5.2 Key Findings
- **Performance**: [Winner model] achieved [X]% test accuracy, demonstrating viability for wildlife monitoring
- **Efficiency**: Custom CNN provides [X]x faster inference while maintaining [X]% accuracy
- **Trade-offs**: Choice between models depends on deployment constraints (accuracy vs. efficiency)
- **Robustness**: Data augmentation significantly improved generalization

### 5.3 Contributions to Wildlife Conservation
The developed system demonstrates the potential of AI for:
- Automated, scalable wildlife monitoring
- Reduced costs for conservation organizations
- Faster response to conservation threats
- Enhanced public engagement through citizen science

### 5.4 Recommendations
Based on the results and analysis:

1. **For High-Accuracy Applications**: Deploy ResNet50 transfer learning model
2. **For Resource-Constrained Environments**: Use custom CNN for faster, lightweight inference
3. **For Production Systems**: Implement ensemble methods combining both models
4. **For Continuous Improvement**: Establish feedback loop with wildlife experts

### 5.5 Lessons Learned
- Transfer learning significantly reduces training time and improves accuracy
- Data augmentation is crucial for generalization to real-world conditions
- Model complexity doesn't always correlate with better performance
- Ethical considerations must be integrated from project inception

### 5.6 Future Work
- Expand dataset to include more endangered species
- Implement object detection for multi-animal images
- Develop mobile application for field deployment
- Collaborate with conservation organizations for real-world validation
- Explore few-shot learning for rare species with limited data

---

## References

1. Kaggle Dataset: [https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint*.
4. Norouzzadeh, M. S., et al. (2018). Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. *PNAS*.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## Appendices

### Appendix A: Code Repository
GitHub Repository: [Link to repository]

### Appendix B: Generated Outputs
All figures, confusion matrices, and training logs are available in the `results/` directory.

### Appendix C: Model Files
Trained models are saved in the `models/` directory:
- `custom_cnn_final.h5`
- `resnet50_transfer_final.h5`

### Appendix D: Dataset Statistics
**[Include detailed dataset statistics if needed]**

### Appendix E: Hyperparameter Tuning
**[Include details of any hyperparameter experiments if conducted]**

---

**End of Report**
