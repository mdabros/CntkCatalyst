﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples.GenerativeModels
{
    /// <summary>
    /// Example based on:
    /// https://cntk.ai/pythondocs/CNTK_206C_WGAN_LSGAN.html
    /// 
    /// Training follows the original paper relatively closely:
    /// Original GAN paper: https://arxiv.org/pdf/1406.2661v1.pdf
    /// WGAN paper: https://arxiv.org/pdf/1701.04862.pdf
    /// 
    /// This example needs manual download of the CIFAR-10 dataset in CNTK format.
    /// Instruction on how to download and convert the dataset can be found here:
    /// https://github.com/Microsoft/CNTK/tree/master/Examples/Image/DataSets/CIFAR-10
    /// </summary>
    [TestClass]
    public class GAN_WGAN
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\CIFAR10";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "Train_cntk_text.txt");

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.Xavier(random.Next(), scale: 0.02);
            var biasInit = Initializers.Zero();

            // Ensure reproducible results with CNTK.
            CNTKLib.SetFixedRandomSeed((uint)random.Next());
            CNTKLib.ForceDeterministicAlgorithms();

            // Setup generator
            var ganeratorInputShape = NDShape.CreateNDShape(new int[] { 100 });
            var generatorInput = Variable.InputVariable(ganeratorInputShape, dataType);
            var generatorNetwork = Generator(generatorInput, weightInit, biasInit, device, dataType);

            // Setup discriminator            
            var discriminatorInputShape = NDShape.CreateNDShape(new int[] { 32 * 32 * 3 });
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

            // Setup generator loss: -D_fake
            var generatorLossFunc = CNTKLib.Negate(discriminatorNetworkFake);
                        
            // Setup discriminator loss: D_real + D_fake
            var discriminatorLossFunc = CNTKLib.Plus(discriminatorNetwork, discriminatorNetworkFake);

            var generatorLearner = Learners.Adam(generatorNetwork.Parameters(), 
                learningRate: 0.00005, momentum: 0.0, varianceMomentum: 0.999, unitGain: false);
            var generatorFitter = CreateFitter(generatorLearner, generatorNetwork, generatorLossFunc, device);

            // Clip discriminator parameters.
            // W-GAN needs to clip the weights of the discriminator before every update,
            // in order to maintain K-Lipschitz continuity.
            // The suggested value of clipping threshold is 0.01.
            var maxClip = Constant.Scalar(dataType, 0.01);
            var minClip = Constant.Scalar(dataType, -0.01);
            var clippedDiscriminatorParameters = discriminatorNetwork.Parameters()
                .Select(p => CNTKLib.Clip(p, minClip, maxClip))
                .Select(f => f.Parameters().Single())
                .ToList();

            var discriminatorLearner = Learners.Adam(discriminatorNetwork.Parameters(),
                learningRate: 0.00005, momentum: 0.0, varianceMomentum: 0.999, unitGain: false);
            var discriminatorFitter = CreateFitter(discriminatorLearner, discriminatorNetwork, discriminatorLossFunc, device);

            int epochs = 30;
            int batchSize = 64;

            // Controls how many steps the discriminator takes, 
            // each time the generator takes 1 step.
            // Note that compared to original GANs, 
            // we train the discriminator many more times than the generator. 
            // The reason behind that is the output of the discriminator serves as an estimation of the EM distance. 
            // We want to train the discriminator until it can closely estimate the EM distance. 
            // In order to make sure that the discriminator has a sufficient good estimation at the very beginning of the training, 
            // we even train it for 100 iterations before train the generator.
            int initialDiscriminatorSteps = 10;
            int discriminatorSteps = 5;

            var isSweepEnd = false;
            for (int epoch = 0; epoch < epochs;)
            {
                var currentDiscriminatorSteps = epoch == 0 ? initialDiscriminatorSteps : discriminatorSteps;
                for (int step = 0; step < discriminatorSteps; step++)
                {
                    // Assign clipped parameters.
                    var parameters = discriminatorNetwork.Parameters();
                    for (int i = 0; i < parameters.Count; i++)
                    {
                        var parameter = parameters[i];
                        var parameterValue = new Value(parameter.GetValue());
                        var clipped = clippedDiscriminatorParameters[i];
                        var clippedValue = new Value(clipped.GetValue());

                        var input = new Dictionary<Variable, Value>()
                        { { parameter, parameterValue } };

                        var output = new Dictionary<Variable, Value>()
                        { { clipped, clippedValue } };

                        CNTKLib.Assign(parameter, clipped).Evaluate(input, output, device);
                    }

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
            var noise = batch.minibatch;

            var predictor = new Predictor(generatorNetwork, device);
            var images = predictor.PredictNextStep(noise);
            var imagesData = images.SelectMany(t => t).ToArray();

            // Show examples
            var app = new Application();
            var window = new PlotWindowBitMap("Generated Images", imagesData, 32, 32, 3, true);
            window.Show();
            app.Run(window);
        }

        Function Generator(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var generatorNetwork = input
                 .Dense(4 * 4 * 256, weightInit(), biasInit, device, dataType)
                 .BatchNorm(BatchNorm.Regular, device, dataType)
                 .ReLU()
                 
                 .Reshape(NDShape.CreateNDShape(new int[] { 4, 4, 256 }))

                 .ConvTranspose2D((5, 5), 128, (2, 2), Padding.Zeros, (8, 8), weightInit(), biasInit, device, dataType)
                 .BatchNorm(BatchNorm.Spatial, device, dataType)
                 .ReLU()

                 .ConvTranspose2D((5, 5), 64, (2, 2), Padding.Zeros, (16, 16), weightInit(), biasInit, device, dataType)
                 .BatchNorm(BatchNorm.Spatial, device, dataType)
                 .ReLU()

                 .ConvTranspose2D((5, 5), 3, (2, 2), Padding.Zeros, (32, 32), weightInit(), biasInit, device, dataType)
                 .Tanh();

            Trace.Write(Model.Summary(generatorNetwork));

            return generatorNetwork.Reshape(NDShape.CreateNDShape(new int[] { 32 * 32 * 3 }));
        }

        Function Discriminator(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var discriminatorNetwork = input
                 .Reshape(NDShape.CreateNDShape(new int[] { 32, 32, 3 }))

                 .Conv2D((5, 5), 64, (2, 2), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 //.BatchNorm(BatchNorm.Spatial, device, dataType)
                 .LeakyReLU(0.2)

                 .Conv2D((5, 5), 128, (2, 2), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .BatchNorm(BatchNorm.Spatial, device, dataType)
                 .LeakyReLU(0.2)

                 .Conv2D((5, 5), 256, (2, 2), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .BatchNorm(BatchNorm.Spatial, device, dataType)
                 .LeakyReLU(0.2)

                 .Dense(1, weightInit(), biasInit, device, dataType);

            Trace.Write(Model.Summary(discriminatorNetwork));

            return discriminatorNetwork;
        }

        CntkMinibatchSource CreateMinibatchSource(string mapFilePath, Dictionary<string, Variable> nameToVariable,
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

            return new CntkMinibatchSource(minibatchSource, nameToVariable);
        }

        static Fitter CreateFitter(Learner learner, Function network, Function loss, DeviceDescriptor device)
        {            
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
