# Animal-Classifier-Transfer-Learning
# Overview 
This task involved building a 4-class classifier (Lion, Tiger, Cat, Dog) by merging and balancing two separate Kaggle datasets. The implementation utilizes a pre-trained ResNet50 backbone to leverage ImageNet feature extraction for a specialized animal classification task.
# Data Strategy & AnalysisDownsampling for Balance:
To maintain a 1:1:1:1 class ratio and prevent majority-class bias, the Cat and Dog datasets were downsampled to 180 images each, matching the available count for the Lion and Tiger classes.
Data Quality Issues:Exploratory data analysis revealed significant Label Noise within the "Lion" class, which mistakenly included images of Leopards and Jaguars. The model was trained on these labels as provided, while documenting this inconsistency for the final report.
# Technical ImplementationArchitecture:
ResNet50 (Keras Applications).Fine-Tuning: The base model was frozen except for the last 10 layers, allowing for the refinement of high-level spatial filters (e.g., fur textures and facial structures).Training Specs: 25 Epochs using the Adam optimizer ($1e-5$ learning rate) and ImageDataGenerator for real-time augmentation (horizontal flips and rescaling).Visualizations: The project includes Accuracy/Loss history graphs and a Confusion Matrix to evaluate inter-class misclassifications.
