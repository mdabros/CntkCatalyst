using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples.DeepLearningFrancoisChollet
{
    /// <summary>
    /// Example from Chapter 5.1: Introduction to convnets:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/5.1-introduction-to-convnets.ipynb
    /// 
    /// This example needs manual download of the MNIST dataset in CNTK format.
    /// Instruction on how to download and convert the dataset can be found here:
    /// https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/MNIST
    /// </summary>
    [TestClass]
    public class Ch_51_Introduction_To_Convnets
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

            // Ensure reproducible results with CNTK.
            CNTKLib.SetFixedRandomSeed((uint)random.Next());
            CNTKLib.ForceDeterministicAlgorithms();

            // Create the architecture.
            var input = Layers.Input(inputShape, dataType);
            // scale input between 0 and 1.
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(0.00390625f, device), input);

            var network = scaledInput
                .Conv2D((3, 3), 32, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2), Padding.None)

                .Conv2D((3, 3), 32, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2), Padding.None)

                .Conv2D((3, 3), 32, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                
                .Dense(64, weightInit(), biasInit, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, weightInit(), biasInit, device, dataType)
                .Softmax();

            // Get input and target variables from network.
            var inputVariable = network.Arguments[0];
            var targetVariable = Variable.InputVariable(outputShape, dataType);

            // setup loss and learner.
            var lossFunc = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
            var metricFunc = Metrics.Accuracy(network.Output, targetVariable);

            // setup trainer.
            var learner = Learners.RMSProp(network.Parameters());
            var trainer = Trainer.CreateTrainer(network, lossFunc, metricFunc, new List<Learner> { learner });

            // Create the network.
            var model = new Model(trainer, network, dataType, device);

            // Write model summary.
            Trace.WriteLine(model.Summary());

            // Setup minibatch sources.
            // Network will be trained using the training set,
            // and tested using the test set.

            // setup name to variable map.
            var nameToVariable = new Dictionary<string, Variable>
            {
                { "features", inputVariable },
                { "labels", targetVariable },
            };

            // The order of the training data is randomize.
            var train = CreateMinibatchSource(trainFilePath, nameToVariable, randomize: true);
            var trainingSource = new CntkMinibatchSource(train, nameToVariable);

            // Notice randomization is switched off for test data.
            var test = CreateMinibatchSource(testFilePath, nameToVariable, randomize: false);
            var testSource = new CntkMinibatchSource(test, nameToVariable);

            // Train the model using the training set.
            model.Fit(trainingSource, epochs: 5, batchSize: 64);

            // Evaluate the model using the test set.
            var (loss, metric) = model.Evaluate(testSource);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");
        }

        MinibatchSource CreateMinibatchSource(string mapFilePath, Dictionary<string, Variable> nameToVariable,
            bool randomize)
        {
            var streamConfigurations = new List<StreamConfiguration>();
            foreach (var kvp in nameToVariable)
            {
                var size = kvp.Value.Shape.TotalSize;
                var name = kvp.Key;
                streamConfigurations.Add(new StreamConfiguration(name, size));
            }

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                mapFilePath,
                streamConfigurations,
                MinibatchSource.InfinitelyRepeat,
                randomize);

            return minibatchSource;
        }
    }
}
