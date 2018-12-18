using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using CNTK;

namespace CntkCatalyst
{
    public class Model
    {
        Function m_loss;
        Function m_metric;
        Trainer m_trainer;
        IDictionary<StreamInformation, Variable> m_streamToVariable;

        const string m_lossName = "Loss";
        const string m_metricName = "Metric";

        const string m_validationLossName = "Validation loss";
        const string m_validationMetricName = "Validation metric";

        readonly DataType m_dataType;
        readonly DeviceDescriptor m_device;

        public Model(Variable network,
            DataType dataType,
            DeviceDescriptor device)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            m_device = device ?? throw new ArgumentNullException(nameof(device));
            m_dataType = dataType;
        }

        public Function Network;

        // TODO: Move compile parameters to constructor
        public void Compile(Trainer trainer,
            IDictionary<StreamInformation, Variable> streamToVariable)
        {
            m_streamToVariable = streamToVariable;
            m_trainer = trainer;

            // Set loss and metric.
            m_loss = trainer.LossFunction();
            m_metric = trainer.EvaluationFunction();
        }

        public Dictionary<string, List<double>> Fit(IMinibatchSource trainMinibatchSource = null, int batchSize = 32, int epochs = 1,
            IMinibatchSource validationMinibatchSource = null)

        {
            // setup fitter.
            //TODO: stream infos from minibatch source should be used, not predefined, these could be from another source.
            var fitter = new Fitter(m_trainer, m_streamToVariable, m_device);

            // store epoch history
            var lossValidationHistory = new Dictionary<string, List<double>>
            {
                { m_lossName, new List<double>() },
                { m_metricName, new List<double>() },
                { m_validationLossName, new List<double>() },
                { m_validationMetricName, new List<double>() },
            };

            for (int epoch = 0; epoch < epochs; )
            {
                var (minibatch, isSweepEnd) = trainMinibatchSource.GetNextMinibatch(batchSize, m_device);

                fitter.FitNextStep(minibatch);

                if (isSweepEnd)
                {
                    var currentLoss = fitter.CurrentLoss;
                    lossValidationHistory[m_lossName].Add(currentLoss);

                    var currentMetric = fitter.CurrentMetric;
                    lossValidationHistory[m_metricName].Add(currentMetric);

                    var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";
                    fitter.ResetLossAccumulation();

                    ++epoch;

                    if (validationMinibatchSource != null)
                    {
                        (var validationLoss, var validationMetric) = Evaluate(validationMinibatchSource, batchSize);
                        traceOutput += $" - ValidationLoss = {validationLoss:F8}, ValidationMetric = {validationMetric:F8}";

                        lossValidationHistory[m_validationLossName].Add(validationLoss);
                        lossValidationHistory[m_validationMetricName].Add(validationMetric);
                    }

                    Trace.WriteLine(traceOutput);
                }
            }

            return lossValidationHistory;
        }

        public (double loss, double metric) Evaluate(IMinibatchSource minibatchSource, int batchSize = 32)
        {
            // create loss and metric evaluators.

            //TODO: stream infos from minibatch source should be used, not predefined, these could be from another source.
            using (var lossEvaluator = new BatchEvalutator(CNTKLib.CreateEvaluator(m_loss), m_streamToVariable, m_device))
            using (var metricEvaluator = new BatchEvalutator(CNTKLib.CreateEvaluator(m_metric), m_streamToVariable, m_device))
            {
                bool isSweepEnd = false;

                // TODO: Check if batchSize is larger than sample count.
                //var evaluationBatchSize = x.SampleCount < batchSize ? x.SampleCount : batchSize;
                var evaluationBatchSize = batchSize;

                while (!isSweepEnd)
                {
                    var nextMinibatch = minibatchSource.GetNextMinibatch(evaluationBatchSize, m_device);
                    var minibatch = nextMinibatch.minibatch;
                    isSweepEnd = nextMinibatch.isSweepEnd;

                    lossEvaluator.EvalauteNextStep(minibatch);
                    metricEvaluator.EvalauteNextStep(minibatch);
                }

                var finalLoss = lossEvaluator.CurrentMetric;
                var finalMetric = metricEvaluator.CurrentMetric;

                return ((float)finalLoss, (float)finalMetric);
            }
        }

        public IList<IList<float>> Predict(IMinibatchSource minibatchSource)
        {
            var predictions = new List<IList<float>>();

            //TODO: stream infos from minibatch source should be used, not predefined, these could be from another source.
            var predictor = new Predictor(Network, m_streamToVariable, m_device);

            bool isSweepEnd = false;

            while (!isSweepEnd)
            {
                const int evaluationBatchSize = 1;
                var nextMinibatch = minibatchSource.GetNextMinibatch(evaluationBatchSize, m_device);
                var minibatch = nextMinibatch.minibatch;
                isSweepEnd = nextMinibatch.isSweepEnd;

                var batchPredictions = predictor.PredictNextStep(minibatch);
                predictions.AddRange(batchPredictions);
            }

            return predictions;
        }

        /// <summary>
        /// Currently will only list all layers with empty name.
        /// </summary>
        /// <returns></returns>
        public string Summary()
        {
            var sb = new StringBuilder();
            sb.AppendLine("---------------------");
            sb.AppendLine("Model Summary");
            sb.AppendLine("Input Shape= " + Network.Arguments[0].Shape.AsString());
            sb.AppendLine("Output Shape= " + Network.Output.Shape.AsString());
            sb.AppendLine("=====================");
            sb.AppendLine("");

            var totalParameterCount = 0;

            // Finds all layers with empty name.
            // TODO: Figure out of to list all layers regardless of name.
            var layers = Network.FindAllWithName(string.Empty)
                .Reverse().ToList();

            foreach (var layer in layers)
            {                
                var outputShape = layer.Output.Shape;
                var layerParameterCount = 0;

                if (layer.Parameters().Any())
                {
                    layerParameterCount = layer.Parameters().First().Shape.TotalSize;
                }

                sb.AppendLine($"Layer Name='{layer.Name}' Output Shape={outputShape.AsString(),-30}" + 
                    $" Param #:{layerParameterCount}");

                totalParameterCount += layerParameterCount;
            }

            sb.AppendLine();
            sb.AppendLine("=====================");
            sb.AppendLine($"Total Number of Parameters: {totalParameterCount:N0}");
            sb.AppendLine("---------------------");

            return sb.ToString();
        }
    }
}
