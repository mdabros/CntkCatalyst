using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test.Models
{
    [TestClass]
    public class ModelTest
    {
        [TestMethod]
        public void Model_Use_Case()
        {
            var inputShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            (var observations, var targets) = CreateArtificialData(inputShape, outputShape, observationCount: 10000);

            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            var random = new Random(232);
            Func<CNTKDictionary> weightInit = () => Initializers.GlorotNormal(random.Next());
            var biasInit = Initializers.Zero();

            // Create the architecture.
            var network = Layers.Input(inputShape, dataType)
                .Dense(512, weightInit(), biasInit, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, weightInit(), biasInit, device, dataType)
                .Softmax();

            // setup input and target variables.
            var inputVariable = network.Arguments[0];
            var targetVariable = Variable.InputVariable(network.Output.Shape, dataType);

            // loss
            var lossFunc = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
            var metricFunc = Metrics.Accuracy(network.Output, targetVariable);

            // setup trainer.
            var learner = Learners.MomentumSGD(network.Parameters());
            var trainer = CNTKLib.CreateTrainer(network, lossFunc, metricFunc, new LearnerVector { learner });

            var model = new Model(trainer, network, dataType, device);

            // setup name to data.
            var nameToData = new Dictionary<string, MemoryMinibatchData>
            {
                { "observations", observations },
                { "targets", targets }
            };

            // setup name to variable
            var nameToVariable = new Dictionary<string, Variable>
            {
                { "observations", inputVariable },
                { "targets", targetVariable },
            };

            var trainSource = new MemoryMinibatchSource(nameToVariable, nameToData, seed: 232, randomize: true);

            model.Fit(trainSource, batchSize: 32, epochs: 10);
            
            (var loss, var metric) = model.Evaluate(trainSource);

            Trace.WriteLine($"Final evaluation - Loss: {loss}, Metric: {metric}");
        }

        static (MemoryMinibatchData observations, MemoryMinibatchData targets) CreateArtificialData(int[] inputShape, int[] outputShape, int observationCount)
        {
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var random = new Random(32);

            var observationsData = new float[observationCount * inputSize];
            observationsData = observationsData.Select(v => (float)random.NextDouble()).ToArray();

            var observations = new MemoryMinibatchData(observationsData, inputShape.ToArray(), observationCount);

            var targetsData = new float[observationCount];
            targetsData = targetsData.Select(d => (float)random.Next(outputShape.Single())).ToArray();
            var oneHotTargetsData = EncodeOneHot(targetsData);

            var targets = new MemoryMinibatchData(oneHotTargetsData, outputShape, observationCount);

            return (observations, targets);
        }

        /// <summary>
        /// Encodes targets in a one-of-n structure. Target vector with two classes [0, 1, 1, 0] becomes a matrix:
        /// 1 0
        /// 0 1
        /// 0 1
        /// 1 0
        /// Primary use is for classification
        /// </summary>
        static float[] EncodeOneHot(float[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = targets.Distinct().OrderBy(v => v)
                .ToDictionary(v => v, v => index++);

            var distinctTargets = targetNameToTargetIndex.Count;
            var oneHot = new float[targets.Length * distinctTargets];

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                var targetIndex = targetNameToTargetIndex[target];
                var oneHotIndex = i * distinctTargets + targetIndex;
                oneHot[oneHotIndex] = 1.0f;
            }

            return oneHot;
        }
    }
}
