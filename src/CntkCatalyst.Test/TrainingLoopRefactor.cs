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
    public class TrainingLoopRefactor
    {
        [TestMethod]
        public void Standard_Cntk_Loop()
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

            var trainMinibatchSource = new MemoryMinibatchSource(observations, targets, seed: 232, randomize: true);

            // setup input and target variables.
            var inputVariable = network.Arguments[0];
            var targetVariable = Variable.InputVariable(network.Output.Shape, dataType);
            
            // loss
            var loss = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
            var metric = Metrics.Accuracy(network.Output, targetVariable);

            // setup learner.
            var learner = Learners.MomentumSGD(network.Parameters());

            // setup trainer.
            var trainer = CNTKLib.CreateTrainer(network, loss, metric, new LearnerVector { learner });

            // variables for training loop.            
            var inputMap = new Dictionary<Variable, Value>();

            var lossSum = 0f;
            var metricSum = 0f;
            var totalSampleCount = 0;
            var epochs = 10;
            var batchSize = 32;

            for (int epoch = 0; epoch < epochs;)
            {
                var (minibatch, isSweepEnd) = trainMinibatchSource.GetNextMinibatch(batchSize, device);
                
                var obserationsStreamInfo = trainMinibatchSource.StreamInfo(trainMinibatchSource.FeaturesName);
                var targetsStreamInfo = trainMinibatchSource.StreamInfo(trainMinibatchSource.TargetsName);

                using (var observationsData = minibatch[obserationsStreamInfo].data)
                using (var targetsData = minibatch[targetsStreamInfo].data)
                {
                    inputMap.Add(inputVariable, observationsData);
                    inputMap.Add(targetVariable, targetsData);

                    trainer.TrainMinibatch(inputMap, false, device);

                    var lossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var metricValue = (float)trainer.PreviousMinibatchEvaluationAverage();

                    // Accumulate loss/metric.
                    lossSum += lossValue * batchSize;
                    metricSum += metricValue * batchSize;
                    totalSampleCount += batchSize;

                    if (isSweepEnd)
                    {
                        var currentLoss = lossSum / totalSampleCount;
                        var currentMetric = metricSum / totalSampleCount;

                        var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";

                        ++epoch;
                        lossSum = 0;
                        metricSum = 0;
                        totalSampleCount = 0;

                        Trace.WriteLine(traceOutput);
                    }

                    // Ensure cleanup
                    inputMap.Clear();
                }
            }
        }

        [TestMethod]
        public void Fitter_Loop()
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

            var minibatchSource = new MemoryMinibatchSource(observations, targets, seed: 232, randomize: true);

            // setup input and target variables.
            var inputVariable = network.Arguments[0];
            var targetVariable = Variable.InputVariable(network.Output.Shape, dataType);

            // loss
            var loss = Losses.CategoricalCrossEntropy(network.Output, targetVariable);
            var metric = Metrics.Accuracy(network.Output, targetVariable);

            // setup learner.
            var learner = Learners.MomentumSGD(network.Parameters());

            // setup trainer.
            var trainer = CNTKLib.CreateTrainer(network, loss, metric, new LearnerVector { learner });

            // setup streaminfo to variable map.
            var streamInfoToVariable = new Dictionary<StreamInformation, Variable>
            {
                { minibatchSource.StreamInfo(minibatchSource.FeaturesName), inputVariable },
                { minibatchSource.StreamInfo(minibatchSource.TargetsName), targetVariable },
            };

            // setup Fitter
            var fitter = new Fitter(trainer, streamInfoToVariable, device);

            // variables for training loop.            
            var inputMap = new Dictionary<Variable, Value>();

            var epochs = 10;
            int batchSize = 32;

            for (int epoch = 0; epoch < epochs;)
            {
                var (minibatch, isSweepEnd) = minibatchSource.GetNextMinibatch(batchSize, device);
                fitter.Step(minibatch);

                if (isSweepEnd)
                {
                    var currentLoss = fitter.CurrentLoss;
                    var currentMetric = fitter.CurrentMetric;
                    fitter.ResetLossAccumulation();

                    var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";

                    ++epoch;

                    Trace.WriteLine(traceOutput);
                }
            }
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

        public sealed class Fitter
        {
            IDictionary<StreamInformation, Variable> m_streamInfoToVariable;
            IDictionary<Variable, Value> m_input;

            float m_lossSum = 0f;
            float m_metricSum = 0f;
            int m_totalSampleCount = 0;

            public Fitter(Trainer trainer, IDictionary<StreamInformation, Variable> streamInfoToVariable,
                DeviceDescriptor device)
            {
                Trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
                m_streamInfoToVariable = streamInfoToVariable ?? throw new ArgumentNullException(nameof(streamInfoToVariable));
                Device = device ?? throw new ArgumentNullException(nameof(device));

                m_input = new Dictionary<Variable, Value>();
            }

            public Trainer Trainer { get; }
            public DeviceDescriptor Device { get; }

            public double CurrentLoss => m_lossSum / m_totalSampleCount;
            public double CurrentMetric => m_metricSum / m_totalSampleCount;

            public void Step(IDictionary<StreamInformation, MinibatchData> minibatch)
            {
                var batchSize = AssignDataFromMinibatch(minibatch);

                Trainer.TrainMinibatch(m_input, false, Device);

                AccumulateLossAndMetric(batchSize);
                
                m_input.Clear();
                DisposeMiniBatchData(minibatch);
            }

            public void ResetLossAccumulation()
            {
                m_lossSum = 0;
                m_metricSum = 0;
                m_totalSampleCount = 0;
            }

            int AssignDataFromMinibatch(IDictionary<StreamInformation, MinibatchData> minibatch)
            {
                var batchSize = 0;
                foreach (var kvp in m_streamInfoToVariable)
                {
                    var streamInfo = kvp.Key;
                    var variable = kvp.Value;

                    var minibatchData = minibatch[streamInfo];
                    batchSize = (int)minibatchData.numberOfSamples;

                    var data = minibatch[streamInfo].data;
                    m_input.Add(variable, data);
                }

                return batchSize;
            }

            void AccumulateLossAndMetric(int batchSize)
            {
                var lossValue = (float)Trainer.PreviousMinibatchLossAverage();
                var metricValue = (float)Trainer.PreviousMinibatchEvaluationAverage();

                // Accumulate loss/metric.
                m_lossSum += lossValue * batchSize;
                m_metricSum += metricValue * batchSize;
                m_totalSampleCount += batchSize;
            }

            static void DisposeMiniBatchData(IDictionary<StreamInformation, MinibatchData> minibatch)
            {
                // Dispose data.
                foreach (var data in minibatch.Values)
                {
                    data.data.Dispose();
                    data.Dispose();
                }
            }
        }
    }
}
