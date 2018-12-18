using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 02: First look at a Neural Network:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/2.1-a-first-look-at-a-neural-network.ipynb
    /// 
    /// This example needs manual download of the MNIST dataset in CNTK format.
    /// Instruction on how to download and convert the dataset can be found here:
    /// https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/MNIST
    /// </summary>
    [TestClass]
    public class Ch_21_First_Look_At_A_Neural_Network
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\Mnist";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "Train-28x28_cntk_text.txt");
            var testFilePath = Path.Combine(baseDataDirectoryPath, "Test-28x28_cntk_text.txt");

            // Define the input and output shape.
            var inputShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.GlorotNormal(random.Next());
            var biasInit = Initializers.Zero();

            // Create the architecture.
            var input = Layers.Input(inputShape, dataType);
            // scale input between 0 and 1.
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(0.00390625f, device), input);

            var network = scaledInput
                .Dense(512, weightInit(), biasInit, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, weightInit(), biasInit, device, dataType)
                .Softmax();

            // Get input and target variables from network.
            var inputVariable = network.Arguments[0];
            var targetVariable = Variable.InputVariable(outputShape, dataType);

            // setup loss and metric.
            var lossFunc = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
            var metricFunc = Metrics.Accuracy(network.Output, targetVariable);
            
            // setup trainer.
            var learner = Learners.RMSProp(network.Parameters());
            var trainer = Trainer.CreateTrainer(network, lossFunc, metricFunc, new List<Learner> { learner });

            // Create the network.
            var model = new Model(trainer, network, dataType, device);

            // Setup minibatch sources.
            // Network will be trained using the training set,
            // and tested using the test set.
            var featuresName = "features";
            var targetsName = "labels";

            // name to variable.
            var nameToVariable = new Dictionary<string, Variable>
            {
                { featuresName, inputVariable },
                { targetsName, targetVariable },
            };

            // The order of the training data is randomize.
            var train = CreateMinibatchSource(trainFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, randomize: true);
            var trainingSource = new CntkMinibatchSource(train, nameToVariable);

            // Notice randomization is switched off for test data.
            var test = CreateMinibatchSource(testFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, randomize: false);
            var testSource = new CntkMinibatchSource(test, nameToVariable);

            // Train the model using the training set.
            model.Fit(trainingSource, epochs: 5, batchSize: 128);

            // Evaluate the model using the test set.
            var (loss, metric) = model.Evaluate(testSource);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");
        }

        MinibatchSource CreateMinibatchSource(string mapFilePath, string featuresName, string targetsName,
            int numberOfClasses, int[] inputShape, bool randomize)
        {
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var streamConfigurations = new StreamConfiguration[]
            {
                new StreamConfiguration(featuresName, inputSize),
                new StreamConfiguration(targetsName, numberOfClasses)
            };

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                mapFilePath, 
                streamConfigurations, 
                MinibatchSource.InfinitelyRepeat, 
                randomize);

            return minibatchSource;
        }
    }
}
