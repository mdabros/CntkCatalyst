using System;
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
    /// Original VAE papers:
    /// https://arxiv.org/pdf/1401.4082.pdf
    /// https://arxiv.org/pdf/1312.6114.pdf
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

            // Setup input dimensions and variables.
            var inputShape = NDShape.CreateNDShape(new int[] { 28, 28, 1 });
            var InputVariable = Variable.InputVariable(inputShape, dataType);
            var scaledInputVariable = CNTKLib.ElementTimes(Constant.Scalar(0.00390625f, device), InputVariable);

            const int latentSpaceSize = 2;
            // Setup the encoder, this encodes the input into a mean and variance parameter.
            var (mean, logVariance) = Encoder(scaledInputVariable, latentSpaceSize, weightInit, biasInit, device, dataType);

            // add latent space sampling. This will draw a latent point using a small random epsilon.
            var epsilon = CNTKLib.NormalRandom(new int[] { latentSpaceSize }, dataType);
            var latentSpaceSampler = CNTKLib.Plus(mean, CNTKLib.ElementTimes(CNTKLib.Exp(logVariance), epsilon));

            // add decoder, this decodes from latent space back to an image.            
            var encoderDecoderNetwork = Decoder(latentSpaceSampler, weightInit, biasInit, device, dataType);

            // Create minibatch source for providing the input images.
            var nameToVariable = new Dictionary<string, Variable> { { "features", encoderDecoderNetwork.Arguments[0] } };
            var minibatchSource = CreateMinibatchSource(trainFilePath, nameToVariable, randomize: true);

            // Reconstruction loss. Forces the decoded samples to match the initial inputs.
            var reconstructionLoss = Losses.BinaryCrossEntropy(encoderDecoderNetwork.Output, scaledInputVariable);

            // Regularization loss. Helps in learning well-formed latent spaces and reducing overfitting to the training data.            
            var regularizationLoss = RegularizationLoss(mean, logVariance, dataType);

            // Overall loss function. Sum of reconstruction- and regularization loss.
            var loss = CNTKLib.Plus(reconstructionLoss, regularizationLoss);

            // Setup trainer.
            var learner = Learners.Adam(encoderDecoderNetwork.Parameters(), learningRate: 0.001, momentum: 0.9);
            var trainer = Trainer.CreateTrainer(encoderDecoderNetwork, loss, loss, new List<Learner> { learner });
            // Create model.
            var model = new Model(trainer, encoderDecoderNetwork, dataType, device);

            Trace.WriteLine(model.Summary());

            // Train the model.
            model.Fit(minibatchSource, batchSize: 16, epochs: 10);

            //// Use the decoder to sample 15x15 images across the latent space.

            // Image generation only requires use of the decoder part of the network.
            // Clone and replace the latentSpaceSampler from training with a new decoder input variable. 
            var decoderInputVariable = Variable.InputVariable(latentSpaceSampler.Output.Shape, dataType);
            var replacements = new Dictionary<Variable, Variable>() { { latentSpaceSampler, decoderInputVariable } };
            var decoder = encoderDecoderNetwork.Clone(ParameterCloningMethod.Freeze, replacements);

            // Sample 15x15 samples from the latent space
            var minibatch = SampleMinibatchForGrid(device, decoderInputVariable, gridSize: 15);

            // Transform from points in latent space to images using the decoder.
            var predictor = new Predictor(decoder, device);
            var images = predictor.PredictNextStep(minibatch);
            var imagesData = images.SelectMany(t => t).ToArray();

            // Show examples.
            var app = new Application();
            var window = new PlotWindowBitMap("Generated Images", imagesData, 28, 28, 1, true);
            window.Show();
            app.Run(window);
        }

        (Function mean, Function logVariance) Encoder(Function input, int latentSpaceSize, 
            Func<CNTKDictionary> weightInit, CNTKDictionary biasInit,
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

                 .Dense(32, weightInit(), biasInit, device, dataType)
                 .ReLU();

            var mean = encoderNetwork.Dense(latentSpaceSize, weightInit(), biasInit, device, dataType);
            var logVariance = encoderNetwork.Dense(latentSpaceSize, weightInit(), biasInit, device, dataType);

            return (mean, logVariance);
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

        static Function RegularizationLoss(Function mean, Function logVariance, DataType dataType)
        {
            // kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            var difference = CNTKLib.Plus(Constant.Scalar(dataType, 1.0), logVariance);
            difference = CNTKLib.Minus(difference, CNTKLib.Square(mean));
            difference = CNTKLib.Minus(difference, CNTKLib.Exp(logVariance));

            var regularizationLoss = CNTKLib.ElementTimes(Constant.Scalar(dataType, -5e-4),
                CNTKLib.ReduceMean(difference, Axis.AllStaticAxes()));

            return regularizationLoss;
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

        static Dictionary<Variable, Value> SampleMinibatchForGrid(DeviceDescriptor device, 
            Variable decoderInputVariable, int gridSize)
        {
            var latentGridData = new float[gridSize * gridSize * 2];

            // Sample latent space in a grid, from -2 to 2, with "gridSize" points in each direction.
            var sampleStart = -2f;
            var sampleIntervalWidth = 4f;

            var index = 0;
            for (int x = 0; x < gridSize; x++)
            {
                for (int y = 0; y < gridSize; y++)
                {
                    latentGridData[index++] = sampleStart + (sampleIntervalWidth / (gridSize - 1)) * x;
                    latentGridData[index++] = sampleStart + (sampleIntervalWidth / (gridSize - 1)) * y;
                }
            }
            var value = Value.CreateBatch(decoderInputVariable.Shape, latentGridData, device);

            var minibatch = new Dictionary<Variable, Value>
            {
                { decoderInputVariable, value},
            };

            return minibatch;
        }
    }
}
