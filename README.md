 Understanding Forward Pass, Backward Pass, Activation Functions, and Binary Classification in Neural Networks
🔁 1. Forward Propagation – What Happens in a Neural Network?
When data enters a neural network, it goes through the forward pass — this is where predictions are made.

✅ Steps of Forward Propagation:
Linear Transformation:

𝑧
=
𝑊
⋅
𝑥
+
𝑏
z=W⋅x+b
W = weights

x = input

b = bias

This gives us a raw output called a logit

Activation Function: Converts the logit into a meaningful output (e.g., a probability).

🔒 2. Activation Functions – Turning Numbers into Understanding
✅ Sigmoid (Used in Binary Classification):
𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 
Output is between 0 and 1

Interpreted as probability of class 1

Used in the output layer of binary classifiers

✅ ReLU (Used in Hidden Layers):
ReLU
(
𝑥
)
=
max
⁡
(
0
,
𝑥
)
ReLU(x)=max(0,x)
Sets negative values to 0, passes positive values as-is

Introduces non-linearity

Helps the model learn complex patterns efficiently

Very fast and effective

🧪 3. Loss Function – Measuring the Error
Once the network produces a prediction, we need to compare it to the true label and compute how wrong it was. That's the role of the loss function.

✅ Binary Cross Entropy (BCE):
For binary classification:

BCE Loss
=
−
[
𝑦
log
⁡
(
𝑦
^
)
+
(
1
−
𝑦
)
log
⁡
(
1
−
𝑦
^
)
]
BCE Loss=−[ylog( 
y
^
​
 )+(1−y)log(1− 
y
^
​
 )]
Where:

𝑦
y = true label (0 or 1)

𝑦
^
y
^
​
  = predicted probability (from sigmoid)

🔍 Example:
If true label is 1, and prediction is 0.9:

Loss
=
−
log
⁡
(
0.9
)
=
0.105
Loss=−log(0.9)=0.105
🔄 4. Backward Propagation – How the Model Learns
Once the loss is calculated, we do the backward pass:

PyTorch calculates how each weight in the model contributed to the error using the chain rule

Gradients are stored in .grad for each parameter

An optimizer (like SGD or Adam) updates the weights to reduce the loss

✅ In Code:
python
Copy
Edit
loss.backward()   # computes gradients
optimizer.step()  # updates weights using gradients
🧱 5. Putting It All Together — A Minimal PyTorch Model
python
Copy
Edit
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, new_features):
        super().__init__()
        self.linear = nn.Linear(new_features, 1)  # Linear Layer
        self.sigmoid = nn.Sigmoid()              # Output Activation

    def forward(self, features):
        out = self.linear(features)              # z = Wx + b
        out = self.sigmoid(out)                  # y_pred = sigmoid(z)
        return out
💡 What It Does:
Takes input of size new_features

Applies linear transformation

Outputs a probability between 0 and 1

Perfect for binary classification tasks

🎯 Summary Table

Concept	Description
Forward Pass	Computes prediction using weights and activations
Backward Pass	Computes gradients of loss w.r.t. parameters
Sigmoid	Used in output to get probabilities
ReLU	Used in hidden layers, fast & efficient
BCE Loss	Measures error in binary classification
Model Class	Combines layers and defines prediction flow
✅ Final Thoughts
This flow — linear transformation → activation → loss → gradients → update weights — is the heart of every deep learning model.
