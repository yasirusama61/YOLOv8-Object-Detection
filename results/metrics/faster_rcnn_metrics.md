# Faster R-CNN Metrics

## Overview:
Faster R-CNN with ResNet-50 backbone and CBAM attention mechanism was used to detect 10 thoracic abnormalities.

### **Performance Metrics**
- **AP@25**: 0.62
- **AP@50**: 0.56
- **AP@75**: 0.48

### **Detailed Observations**
- **Strengths**:
  - High AP for abnormalities such as Consolidation and Nodule due to CBAM's enhanced feature extraction.
- **Challenges**:
  - Lower performance on small regions, like Mass, due to dataset imbalances despite oversampling and augmentation.

