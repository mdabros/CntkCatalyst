﻿using System;
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
    /// Example based on:
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
    public class GAN_BasicGAN
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\Mnist";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "Train-28x28_cntk_text.txt");

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.Xavier(random.Next());
            var biasInit = Initializers.Zero();

            // Ensure reproducible results with CNTK.
            CNTKLib.SetFixedRandomSeed((uint)random.Next());
            CNTKLib.ForceDeterministicAlgorithms();

            // Setup generator
            var ganeratorInputShape = NDShape.CreateNDShape(new int[] { 100 });
            var generatorInput = Variable.InputVariable(ganeratorInputShape, dataType);
            var generatorNetwork = Generator(generatorInput, weightInit, biasInit, device, dataType);

            // Setup discriminator            
            var discriminatorInputShape = NDShape.CreateNDShape(new int[] { 784 }); // 28 * 28 * 1.
            var discriminatorInput = Variable.InputVariable(discriminatorInputShape, dataType);
            // scale image input between -1.0 and 1.0.
            var discriminatorInputScaled = CNTKLib.Minus(
                CNTKLib.ElementTimes(Constant.Scalar(2 * 0.00390625f, device), discriminatorInput),
                Constant.Scalar(1.0f, device));

            var discriminatorNetwork = Discriminator(discriminatorInputScaled, weightInit, biasInit, device, dataType);

            // The discriminator must be used on both the real MNIST images and fake images generated by the generator function.
            // One way to represent this in the computational graph is to create a clone of the output of the discriminator function, 
            // but with substituted inputs. Setting method = share in the clone function ensures that both paths through the discriminator model 
            // use the same set of parameters.
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

            // Setup generator loss: 1.0 - C.log(D_fake)
            var generatorLossFunc = CNTKLib.Minus(Constant.Scalar(1.0f, device), 
                CNTKLib.Log(discriminatorNetworkFake));
                        
            // Setup discriminator loss: -(C.log(D_real) + C.log(1.0 - D_fake))
            var discriminatorLossFunc = CNTKLib.Negate(CNTKLib.Plus(CNTKLib.Log(discriminatorNetwork), 
                CNTKLib.Log(CNTKLib.Minus(Constant.Scalar(1.0f, device), discriminatorNetworkFake))));

            // Create fitters for the training loop.
            var generatorFitter = CreateFitter(generatorNetwork, generatorLossFunc, device);
            var discriminatorFitter = CreateFitter(discriminatorNetwork, discriminatorLossFunc, device);

            // Note, that the network needs to train for many epochs to show realistic results.
            int epochs = 700;
            int batchSize = 1024;
            
            // Controls how many steps the discriminator takes, 
            // each time the generator takes 1 step.
            // Default from the original paper is 1.
            int discriminatorSteps = 1;

            var isSweepEnd = false;
            for (int epoch = 0; epoch < epochs;)
            {
                for (int step = 0; step < discriminatorSteps; step++)
                {
                    // Discriminator needs both real images and noise, 
                    // so uses the composite minibatch source.
                    var minibatchItems = compositeMinibatchSource.GetNextMinibatch(batchSize, device);
                    isSweepEnd = minibatchItems.isSweepEnd;

                    discriminatorFitter.FitNextStep(minibatchItems.minibatch, batchSize);
                    DisposeValues(minibatchItems.minibatch);
                }

                // Generator only needs noise images, 
                // so uses the noise minibatch source separately.
                var noiseMinibatchItems = noiseMinibatchSource.GetNextMinibatch(batchSize, device);

                generatorFitter.FitNextStep(noiseMinibatchItems.minibatch, batchSize);
                DisposeValues(noiseMinibatchItems.minibatch);

                if (isSweepEnd)
                {
                    var generatorCurrentLoss = generatorFitter.CurrentLoss;
                    generatorFitter.ResetLossAndMetricAccumulators();

                    var discriminatorCurrentLoss = discriminatorFitter.CurrentLoss;
                    discriminatorFitter.ResetLossAndMetricAccumulators();

                    var traceOutput = $"Epoch: {epoch + 1:000} Generator Loss = {generatorCurrentLoss:F8}, Discriminator Loss = {discriminatorCurrentLoss:F8}";
                    Trace.WriteLine(traceOutput);

                    ++epoch;
                }
            }

            // Sample 6x6 images from generator.
            var samples = 6 * 6;
            var batch = noiseMinibatchSource.GetNextMinibatch(samples, device);
            var examples = batch.minibatch;

            var predictor = new Predictor(generatorNetwork, device);
            var images = predictor.PredictNextStep(examples);
            var imagesData = images.SelectMany(t => t).ToArray();

            // Show examples
            var app = new System.Windows.Application();
            var window = new PlotWindowBitMap("Generated Images", imagesData, 28, 28, 1, true);
            window.Show();
            app.Run(window);
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

        static Fitter CreateFitter(Function network, Function loss, DeviceDescriptor device)
        {
            var learner = Learners.MomentumSGD(network.Parameters(), learningRate: 0.00005, momentum: 0.9);
            var trainer = Trainer.CreateTrainer(network, loss, loss, new List<Learner> { learner });
            var fitter = new Fitter(trainer, device);

            return fitter;
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
