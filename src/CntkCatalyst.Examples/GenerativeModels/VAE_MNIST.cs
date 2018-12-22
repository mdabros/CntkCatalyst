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
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/8.4-generating-images-with-vaes.ipynb
    /// 
    /// 
    /// </summary>
    [TestClass]
    public class VAE_MNIST
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\Mnist";
            var trainFilePath = Path.Combine(baseDataDirectoryPath, "Train-28x28_cntk_text.txt");
            var testFilePath = Path.Combine(baseDataDirectoryPath, "Test-28x28_cntk_text.txt");

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Setup initializers
            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.GlorotUniform(random.Next());
            var biasInit = Initializers.Zero();

            // Ensure reproducible results with CNTK.
            CNTKLib.SetFixedRandomSeed((uint)random.Next());
            CNTKLib.ForceDeterministicAlgorithms();

            // Setup encoder.
            var encoderInputShape = NDShape.CreateNDShape(new int[] { 28, 28, 1 });
            var encoderInput = Variable.InputVariable(encoderInputShape, dataType);
            var scaledEncoderInput = CNTKLib.ElementTimes(Constant.Scalar(0.00390625f, device), encoderInput);
            var encoderNetwork = Encoder(scaledEncoderInput, weightInit, biasInit, device, dataType);

            // Setup latent variables.
            var latentSize = 2;
            var z_mean = encoderNetwork.Dense(latentSize, weightInit(), biasInit, device, dataType);
            var z_log_var = encoderNetwork.Dense(latentSize, weightInit(), biasInit, device, dataType);
            var epsilon = CNTKLib.NormalRandom(new int[] { latentSize }, dataType);
            var z = CNTKLib.Plus(z_mean, CNTKLib.ElementTimes(CNTKLib.Exp(z_log_var), epsilon), "decoderInputNode");

            // Setup decoder            
            var decoderNetwork = Decoder(z, weightInit, biasInit, device, dataType);

            // Create minibatch source for providing the real images.
            var nameToVariable = new Dictionary<string, Variable> { { "features", decoderNetwork.Arguments[0] } };
            var trainMinibatchSource = CreateMinibatchSource(trainFilePath, nameToVariable, randomize: true);
            var testMinibatchSource = CreateMinibatchSource(testFilePath, nameToVariable, randomize: false);

            // Regularization metric
            var square_ = CNTKLib.Square(z_mean);
            var exp_ = CNTKLib.Exp(z_log_var);
            var constant_1 = Constant.Scalar(dataType, 1.0);
            var diff_ = CNTKLib.Plus(constant_1, z_log_var);
            diff_ = CNTKLib.Minus(diff_, square_);
            diff_ = CNTKLib.Minus(diff_, exp_);
            var constant_2 = Constant.Scalar(dataType, -5e-4);
            var regularization_metric = CNTKLib.ElementTimes(constant_2, CNTKLib.ReduceMean(diff_, Axis.AllStaticAxes()));

            // Overall loss function
            var crossentropy_loss = Losses.BinaryCrossEntropy(decoderNetwork.Output, scaledEncoderInput);
            crossentropy_loss = CNTKLib.ReduceMean(crossentropy_loss, Axis.AllStaticAxes());
            var loss = CNTKLib.Plus(crossentropy_loss, regularization_metric);

            // Setup trainer.
            var learner = Learners.Adam(decoderNetwork.Parameters(), learningRate: 0.001, momentum: 0.9);
            var trainer = Trainer.CreateTrainer(decoderNetwork, loss, loss, new List<Learner> { learner });
            // Create model.
            var model = new Model(trainer, decoderNetwork, dataType, device);

            Trace.WriteLine(model.Summary());

            // Train the model.
            model.Fit(trainMinibatchSource, batchSize: 16, epochs: 10,
                validationMinibatchSource: testMinibatchSource);

            //// Sample 15x15 images from the latent space.
            
            // Setup decoder input for prediction.
            var decoderInputNode = decoderNetwork.FindByName("decoderInputNode");
            var decoderInputVariable = Variable.InputVariable(decoderInputNode.Output.Shape, dataType);
            var replacements = new Dictionary<Variable, Variable>() { { decoderInputNode, decoderInputVariable } };
            var decoder = decoderNetwork.Clone(ParameterCloningMethod.Freeze, replacements);

            // Sample 15x15 samples from the latent space
            var minibatch = SampleMinibatchForGrid(device, decoderInputVariable, gridSize: 15);

            // Transform from latent space to images using the decoder.
            var predictor = new Predictor(decoder, device);
            var images = predictor.PredictNextStep(minibatch);
            var imagesData = images.SelectMany(t => t).ToArray();

            // Show the example images.
            var app = new Application();
            var window = new PlotWindowBitMap("Generated Images", imagesData, 28, 28, 1, true);
            window.Show();
            app.Run(window);
        }

        Function Encoder(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var encoderNetwork = input
                 .Conv2D((3, 3), 32, (1, 1), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .ReLU()

                 .Conv2D((3, 3), 64, (2, 2), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .ReLU()

                 .Conv2D((3, 3), 64, (1, 1), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .ReLU()

                 .Conv2D((3, 3), 64, (1, 1), Padding.Zeros, weightInit(), biasInit, device, dataType)
                 .ReLU()

                 .Dense(32, weightInit(), biasInit, device, dataType);

            return encoderNetwork;
        }

        Function Decoder(Function input, Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
            DeviceDescriptor device, DataType dataType)
        {
            var decoderNetwork = input
                .Dense(14 * 14 * 64, weightInit(), biasInit, device, dataType)
                .ReLU()

                .Reshape(NDShape.CreateNDShape(new int[] { 14, 14, 64 }))

                .ConvTranspose2D((3, 3), 32, (2, 2), Padding.Zeros, (28, 28), weightInit(), biasInit, device, dataType)
                .ReLU()

                .Conv2D((3, 3), 1, (1, 1), Padding.Zeros, weightInit(), biasInit, device, dataType)
                .Sigmoid();

            return decoderNetwork;
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

        static Dictionary<Variable, Value> SampleMinibatchForGrid(DeviceDescriptor device, Variable decoderInputVariable, int gridSize)
        {
            var data = new float[gridSize * gridSize * 2];
            var sample_start = -2f;
            var sample_interval_width = 4f;
            for (int i = 0, pos = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    data[pos++] = sample_start + (sample_interval_width / (gridSize - 1)) * i;
                    data[pos++] = sample_start + (sample_interval_width / (gridSize - 1)) * j;
                }
            }
            var value = Value.CreateBatch(decoderInputVariable.Shape, data, device);

            var minibatch = new Dictionary<Variable, Value>
            {
                { decoderInputVariable, value},
            };
            return minibatch;
        }
    }
}
