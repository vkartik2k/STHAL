# Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

The implementation differs from the original paper in the following ways :
  a) lexicons are not considered
  b) Bucketing is used to speed up the training
  c) nadam optimizer used instead of SGD

 ## To run the script
 ```bash
    python3 nn.py
 ```
 ## Requirements
    0) nltk
    1) numpy==1.16.1 or numpy 1.16.2
    2) Keras==2.1.2
    3) Tensorflow==1.4.1
 

## Inference on trained model

```python
from ner import Parser

p = Parser()

p.load_models("models/")

p.predict("% Add a Statement")
##Output
```