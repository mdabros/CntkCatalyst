using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples.DeepLearningFrancoisChollet
{
    /// <summary>
    /// Example from Chapter 5.2: Using convnets with small datasets:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
    /// 
    /// This example needs manual download of the "dogs-vs-cats" dataset.
    /// Sources to download from:
    /// https://www.kaggle.com/c/dogs-vs-cats/data (needs an account)
    /// https://www.microsoft.com/en-us/download/details.aspx?id=54765
    /// </summary>
    [TestClass]
    public class Ch_52_Using_Convnets_With_Small_Datasets
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\CatsAndDogs";
            var mapFiles = PrepareMapFiles(baseDataDirectoryPath);

            // Define the input and output shape.
            var inputShape = new int[] { 150, 150, 3 };
            var numberOfClasses = 2;
            var outputShape = new int[] { numberOfClasses };

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.GlorotNormal(random.Next());
            var biasInit = Initializers.Zero();

            // Create the architecture.
            var network = Layers.Input(inputShape, dataType)
                .Conv2D((3, 3), 32, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 64, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 128, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 128, (1, 1), Padding.None, weightInit(), biasInit, device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Dense(512, weightInit(), biasInit, device, dataType)
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
            var featuresName = "features";
            var targetsName = "targets";

            // setup name to variable map.
            var nameToVariable = new Dictionary<string, Variable>
            {
                { featuresName, inputVariable },
                { targetsName, targetVariable },
            };
            var train = CreateMinibatchSource(mapFiles.trainFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: true);
            var trainingSource = new CntkMinibatchSource(train, nameToVariable);

            // Notice augmentation is switched off for validation data.
            var valid = CreateMinibatchSource(mapFiles.validFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: false);
            var validationSource = new CntkMinibatchSource(valid, nameToVariable);

            // Notice augmentation is switched off for test data.
            var test = CreateMinibatchSource(mapFiles.testFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: false);
            var testSource = new CntkMinibatchSource(test, nameToVariable);

            // Train the model using the training set.
            model.Fit(trainMinibatchSource: trainingSource,
                epochs: 100, batchSize: 32,
                validationMinibatchSource: validationSource);

            // Evaluate the model using the test set.
            var (loss, metric) = model.Evaluate(testSource);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            // Save model.
            model.Network.Save("cats_and_dogs_small_2.cntk");
        }

        MinibatchSource CreateMinibatchSource(string mapFilePath, string featuresName, string targetsName,
            int numberOfClasses, int[] inputShape, bool augmentation)
        {
            var transforms = new List<CNTKDictionary>();
            if (augmentation)
            {
                var randomSideTransform = CNTKLib.ReaderCrop(
                    cropType:"RandomSide",
                    cropSize: new Tuple<int, int>(0, 0),
                    sideRatio: new Tuple<float, float>(0.8f, 1.0f),
                    areaRatio: new Tuple<float, float>(0.0f, 0.0f),
                    aspectRatio: new Tuple<float, float>(1.0f, 1.0f),
                    jitterType: "uniRatio");

                transforms.Add(randomSideTransform);
            }

            var scaleTransform = CNTKLib.ReaderScale(inputShape[0], inputShape[1], inputShape[2]);
            transforms.Add(scaleTransform);

            var imageDeserializer = CNTKLib.ImageDeserializer(mapFilePath, targetsName, 
                (uint)numberOfClasses, featuresName, transforms);

            var minibatchSourceConfig = new MinibatchSourceConfig(new DictionaryVector() { imageDeserializer });
            return CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

        public static (string trainFilePath, string validFilePath, string testFilePath) PrepareMapFiles(
            string baseDataDirectoryPath)
        {
            var imageDirectoryPath = Path.Combine(baseDataDirectoryPath, "train");

            // Download data from one of these locations:
            // https://www.kaggle.com/c/dogs-vs-cats/data (needs an account)
            // https://www.microsoft.com/en-us/download/details.aspx?id=54765
            if (!Directory.Exists(imageDirectoryPath))
            {
                throw new ArgumentException($"Image data directory not found: {imageDirectoryPath}");
            }

            const int trainingSetSize = 1000;
            const int validationSetSize = 500;
            const int testSetSize = 500;

            const string trainFileName = "train_map.txt";
            const string validFileName = "validation_map.txt";
            const string testFileName = "test_map.txt";

            var fileNames = new string[] { trainFileName, validFileName, testFileName };
            var numberOfSamples = new int[] { trainingSetSize, validationSetSize, testSetSize };
            var counter = 0;

            for (int j = 0; j < fileNames.Length; j++)
            {
                var filename = fileNames[j];
                using (var distinationFileWriter = new System.IO.StreamWriter(filename, false))
                {
                    for (int i = 0; i < numberOfSamples[j]; i++)
                    {
                        var catFilePath = Path.Combine(imageDirectoryPath, "cat", $"cat.{counter}.jpg");
                        var dogFilePath = Path.Combine(imageDirectoryPath, "dog", $"dog.{counter}.jpg");
                        counter++;

                        distinationFileWriter.WriteLine($"{catFilePath}\t0");
                        distinationFileWriter.WriteLine($"{dogFilePath}\t1");
                    }
                }
                Trace.WriteLine("Wrote " + Path.Combine(Directory.GetCurrentDirectory(), filename));
            }

            return (trainFileName, validFileName, testFileName);
        }
    }
}
