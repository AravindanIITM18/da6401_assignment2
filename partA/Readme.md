## Building a Convolutional Neural Network for Classification Task

### Model Implementation
The file `FlexibleCNN_model_class.py` contains the implementation of the FlexibleCNN model class as specified in Question 1 of Part A.

### Sample Usage
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlexibleCNN(
    input_channels=3,
    num_classes=10,
    conv_filters=[64, 64, 64, 64, 64],
    filter_size=3,
    activation="GELU",
    dense_units=128,
    dropout_rate=0.3
).to(device)

A2_partA.ipynb contains all the code blocks used and has detailed descriptions of each block and reasoning as well.
