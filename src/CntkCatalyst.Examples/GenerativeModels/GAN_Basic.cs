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
            // TODO: 
            // - Add visualization.
            // - Verify results.
            // - Clean up code.

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

            // Create minibatch source for providing the real images.
            var imageNameToVariable = new Dictionary<string, Variable> { { "features", discriminatorInput } };
            var imageMinibatchSource = CreateMinibatchSource(trainFilePath, imageNameToVariable, randomize: true);

            // Create minibatch source for providing the noise.
            var noiseNameToVariable = new Dictionary<string, Variable> { { "noise", generatorInput } };
            var noiseMinibatchSource = new UniformNoiseMinibatchSource(noiseNameToVariable, min: -1.0f, max: 1.0f, seed: random.Next());
            
            // Combine both sources in the composite minibatch source.
            var compositeMinibatchSource = new CompositeMinibatchSource(imageMinibatchSource, noiseMinibatchSource);

            // setup generator loss: 1.0 - C.log(D_fake)
            var generatorLossFunc = CNTKLib.Minus(Constant.Scalar(1.0f, device), 
                CNTKLib.Log(discriminatorNetworkFake));
                        
            // setup discriminator loss: -(C.log(D_real) + C.log(1.0 - D_fake))
            var discriminatorLossFunc = CNTKLib.Negate(CNTKLib.Plus(CNTKLib.Log(discriminatorNetwork), 
                CNTKLib.Log(CNTKLib.Minus(Constant.Scalar(1.0f, device), discriminatorNetworkFake))));

            var generatorFitter = CreateFitter(generatorNetwork, generatorLossFunc, device);
            var discriminatorFitter = CreateFitter(discriminatorNetwork, discriminatorLossFunc, device);

            int epochs = 10;
            int batchSize = 1024;
            int discriminitorSteps = 1;

            var isSweepEnd = false;
            for (int epoch = 0; epoch < epochs;)
            {
                // Fit discriminator for k steps.
                for (int k = 0; k < discriminitorSteps; k++)
                {
                    // Discriminator needs both real images and noise, 
                    // so uses the composite minibatch source.
                    var minibatchItems = compositeMinibatchSource.GetNextMinibatch(batchSize, device);
                    var discriminatorMinibatch = minibatchItems.minibatch;
                    isSweepEnd = minibatchItems.isSweepEnd;

                    if (!isSweepEnd) // Only fit when !isSweepEnd, to have full batch sizes.
                    {
                        discriminatorFitter.FitNextStep(discriminatorMinibatch, batchSize);
                    }

                    DisposeValues(discriminatorMinibatch);
                }

                // Generator only needs noise images, 
                // so uses the noise mini batch source separately.
                var (generatorMinibatch, sweep) = noiseMinibatchSource.GetNextMinibatch(batchSize, device);
                generatorFitter.FitNextStep(generatorMinibatch, batchSize);
                DisposeValues(generatorMinibatch);

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

            // Sample 6x6 images from generator.
            var predictor = new Predictor(generatorNetwork, device);
            var samples = 6 * 6;
            var batch = noiseMinibatchSource.GetNextMinibatch(samples, device);
            var examples = batch.minibatch;

            var image = predictor.PredictNextStep(examples);
            var imageData = image.SelectMany(t => t).ToArray();

            // Show examples
            var app = new System.Windows.Application();
            var window = new PlotWindowBitMap("Generated Images", imageData, 28, 28, 1, true);
            window.Show();
            app.Run(window);
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

        CntkMinibatchSource CreateMinibatchSource(string mapFilePath, Dictionary<string, Variable> nameToVariable,
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

            return new CntkMinibatchSource(minibatchSource, nameToVariable);
        }

        void DisposeValues(IDictionary<Variable, Value> minibatch)
        {
            foreach (var value in minibatch.Values)
            {
                value.Dispose();
            }
        }
    }
}
