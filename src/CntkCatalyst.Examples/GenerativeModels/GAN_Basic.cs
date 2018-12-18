using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples.GenerativeModels
{
    /// <summary>
    /// Example based:
    /// https://cntk.ai/pythondocs/CNTK_206A_Basic_GAN.html
    /// 
    /// Training follows the original paper relatively closely:
    /// https://arxiv.org/pdf/1406.2661v1.pdf
    /// 
    /// This example needs manual download of the MNIST dataset in CNTK format.
    /// Instruction on how to download and convert the dataset can be found here:
    /// https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/MNIST
    /// </summary>
    [TestClass]
    public class GAN_Basic
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\Mnist";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "Train-28x28_cntk_text.txt");

            // Define the input and output shape.
            var inputShape = new int[] { 784 }; // 28 * 28 * 1
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.Xavier(random.Next());
            var biasInit = Initializers.Zero();

            //var inputDynamicAxis = new List<Axis> { Axis.DefaultDynamicAxis() };

            // setup generator
            var ganeratorInputShape = NDShape.CreateNDShape(new int[] { 100 });
            var generatorInput = Variable.InputVariable(ganeratorInputShape, dataType);
            var generatorNetwork = Generator(generatorInput, weightInit, biasInit, device, dataType);

            // setup discriminator
            var discriminatorInputShape = NDShape.CreateNDShape(inputShape);
            var discriminatorInput = Variable.InputVariable(discriminatorInputShape, dataType);
            var discriminatorInputScaled = discriminatorInput
                .ElementTimes(Constant.Scalar(2 * 0.00390625f, device))
                .Minus(Constant.Scalar(1.0f, device));

            var discriminatorNetwork = Discriminator(discriminatorInputScaled, weightInit, biasInit, device, dataType);
            var discriminatorNetworkFake = discriminatorNetwork
                .Clone(ParameterCloningMethod.Share, replacements:
                    new Dictionary<Variable, Variable>
                    {
                        { discriminatorInputScaled.Output, generatorNetwork.Output },
                    });

            var nameToVariable = new Dictionary<string, Variable>
            {
                { "features", discriminatorInput },
            };

            // The order of the training data is randomize.
            var train = CreateMinibatchSource(trainFilePath, nameToVariable, randomize: true);
            var trainingSource = new CntkMinibatchSource(train, nameToVariable);

            // setup generator loss: 1.0 - C.log(D_fake)
            var generatorLossFunc = CNTKLib.Minus(Constant.Scalar(1.0f, device), 
                CNTKLib.Log(discriminatorNetworkFake));
                        
            // setup discriminator loss: -(C.log(D_real) + C.log(1.0 - D_fake))
            var discriminatorLossFunc = CNTKLib.Negate(CNTKLib.Plus(CNTKLib.Log(discriminatorNetwork), 
                CNTKLib.Log(CNTKLib.Minus(Constant.Scalar(1.0f, device), discriminatorNetworkFake))));

            var generatorFitter = CreateFitter(generatorNetwork, generatorLossFunc, device);
            var discriminatorFitter = CreateFitter(discriminatorNetwork, discriminatorLossFunc, device);

            int epochs = 100;
            int batchSize = 1024;
            int discriminitorSteps = 1;

            var isSweepEnd = false;
            for (int epoch = 0; epoch < epochs;)
            {
                // Fit discriminator for k steps.
                for (int k = 0; k < discriminitorSteps; k++)
                {
                    var minibatchItems = trainingSource.GetNextMinibatch(batchSize, device);
                    var discriminatorMinibatch = minibatchItems.minibatch;
                    isSweepEnd = minibatchItems.isSweepEnd;

                    if (!isSweepEnd) // only fit when !isSweepEnd, to have full batch sizes.
                    {
                        // Add noise data to minibatch
                        var discriminatorNoiseBatchValue = Value.CreateBatch<float>(ganeratorInputShape,
                            NoiseSamples(random, batchSize, ganeratorInputShape.Dimensions), device);

                        discriminatorMinibatch.Add(generatorInput, discriminatorNoiseBatchValue);
                        discriminatorFitter.FitNextStep(discriminatorMinibatch, batchSize);

                        discriminatorNoiseBatchValue.Dispose();
                    }
                }

                var generatorNoiseBatchValue = Value.CreateBatch<float>(ganeratorInputShape,
                    NoiseSamples(random, batchSize, ganeratorInputShape.Dimensions), device);

                var generatorMinibatch = new Dictionary<Variable, Value>
                {
                    { generatorInput,  generatorNoiseBatchValue}
                };

                generatorFitter.FitNextStep(generatorMinibatch, batchSize);
                generatorNoiseBatchValue.Dispose();

                if (isSweepEnd)
                {
                    var generatorCurrentLoss = generatorFitter.CurrentLoss;
                    generatorFitter.ResetLossAndMetricAccumulators();

                    var discriminatorCurrentLoss = discriminatorFitter.CurrentLoss;
                    discriminatorFitter.ResetLossAndMetricAccumulators();


                    var traceOutput = $"Epoch: {epoch + 1:000} Generator Loss = {generatorCurrentLoss:F8}, Discriminator Loss = {discriminatorCurrentLoss:F8}";

                    ++epoch;

                    Trace.WriteLine(traceOutput);
                }
            }
        }

        static Fitter CreateFitter(Function network, Function loss, DeviceDescriptor device)
        {
            var learner = Learners.Adam(network.Parameters(), learningRate: 0.00005);
            var trainer = Trainer.CreateTrainer(network, loss, loss, new List<Learner> { learner });
            var fitter = new Fitter(trainer, device);

            return fitter;
        }

        Function Generator(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var generatorNetwork = input
                 .Dense(128, weightInit(), biasInit, device, dataType)
                 .ReLU()
                 .Dense(784, weightInit(), biasInit, device, dataType) // output corresponds to input shape: 28 * 28 = 784.
                 .Tanh();

            return generatorNetwork;
        }

        Function Discriminator(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var discriminatorNetwork = input
                 .Dense(128, weightInit(), biasInit, device, dataType)
                 .ReLU()
                 .Dense(1, weightInit(), biasInit, device, dataType)
                 .Sigmoid();

            return discriminatorNetwork;
        }

        float[] NoiseSamples(Random random, int sampleCount, IList<int> sampleShape, 
            float min = -1.0f, float max = 1.0f)
        {
            var totalElementCount = sampleCount * sampleShape.Aggregate((d1, d2) => d1 * d2);

            var samples = Enumerable.Range(0, totalElementCount)
                .Select(v => (float)random.NextDouble() * (max - min) + min)
                .ToArray();

            return samples;
        }

        MinibatchSource CreateMinibatchSource(string mapFilePath, Dictionary<string, Variable> nameToVariable,
            bool randomize)
        {
            var streamConfigurations = new List<StreamConfiguration>();
            foreach (var kvp in nameToVariable)
            {
                var size = kvp.Value.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
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
