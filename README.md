
CntkCatalyst
=================
[![Build Status](https://machinelearning.visualstudio.com/cntkcatalyst-build/_apis/build/status/mdabros.CntkCatalyst?branchName=master)](https://machinelearning.visualstudio.com/cntkcatalyst-build/_build/latest?definitionId=29&branchName=master)

CntkCatalyst is an ongoing effort to learn about the [Python Keras API](https://github.com/keras-team/keras),
while trying to reproduce the functionality in a CNTK based C# API.
The approach is to reproduce the Keras examples from the book/notebook [Deep Learning With Python](https://github.com/fchollet/deep-learning-with-python-notebooks),
and use these as a guide to build the API. 

The CntkCatalyst API will not be a one-to-one mapping of the Keras API, 
but will use Keras as inspiration, and make changes where it makes sense. 
CntkCatalyst will be created on top of Microsoft C# API for [CNTK](https://github.com/Microsoft/CNTK),
with all the limitations that this presents.

Many of the Keras examples have already been reproduced using C# CNTK [here](https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk).
However, these are implemented using the raw CNTK C# API directly, 
which does not provide a very desirable user experience. 

Hopefully, when the end-goal has been reached, The CntkCatalyst API 
will provide a deep learning experience in C#, very close to what you get with Keras in Python.

Since this is an ongoing effort, the API under development can be expected to change a lot,
and should **not** be considered as production ready.

Below, an example from the Keras python  API is reproduced using the current
CntkCatalyst C# API:

### Keras Python API:

```python
# Create the network and model
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compile the network with the selected learner, loss and metric.
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Train the model using the training set.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evaluate the model using the test set.
test_loss, test_acc = network.evaluate(test_images, test_labels)

# Save the model
model.save('mnist.h5')
```

### CntkCatalyst C# API:

```csharp
// Create the network.
var network = Layers.Input(inputShape, dataType)
    .Dense(512, weightInit(), biasInit, device, dataType)
    .ReLU()
    .Dense(10, weightInit(), biasInit, device, dataType)
    .Softmax();

// setup loss and metric.
var lossFunc = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
var metricFunc = Metrics.Accuracy(network.Output, targetVariable);

// setup trainer.
var learner = Learners.RMSProp(network.Parameters());
var trainer = Trainer.CreateTrainer(network, lossFunc, metricFunc, new List<Learner> { learner });

// Create the model.
var model = new Model(trainer, network, dataType, device);

// Train the model using the training set.
model.Fit(trainingSource, epochs: 5, batchSize: 128);

// Evaluate the model using the test set.
var (loss, metric) = model.Evaluate(testSource);

// Predict the test set using the model.
var predictions = model.Predict(testSource);  

// Save model.
model.Network.Save("mnist.cntk");
```

Projects
------------
The solution file contains 3 project:
 - *CntkCatalyst:* Main project containing the CntkCatalyst API.
 - *CntkCatalyst.Test:* Test project for the CntkCatalyst API.
 - *CntkCatalyst.Examples:* Examples project for implementing the Keras examples for guiding the API development.

Initial code was started in a branch of SharpLearning: [sharplearning-neural-cntk](https://github.com/mdabros/SharpLearning/tree/sharplearning-neural-cntk/src).

Contributing
------------
Contributions are welcome in the following areas:

 1. Add new issues with bug descriptions or feature suggestions.
 2. Implement additional functionality in *CntkCatalyst*, by creating a pull request. 
 3. Implement additional examples in *CntkCatalyst.Examples*, by creating a pull request. 
 
When contributing, please follow the [contribution guide](https://github.com/mdabros/CntkCatalyst/blob/master/CONTRIBUTING.md).
