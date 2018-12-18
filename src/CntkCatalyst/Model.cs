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

        const string m_lossName = "Loss";
        const string m_metricName = "Metric";

        const string m_validationLossName = "Validation loss";
        const string m_validationMetricName = "Validation metric";

        readonly DataType m_dataType;
        readonly DeviceDescriptor m_device;

        public Model(Trainer trainer,
            Variable network,
            DataType dataType,
            DeviceDescriptor device)
        {
            m_trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            Network = network ?? throw new ArgumentNullException(nameof(network));
            m_device = device ?? throw new ArgumentNullException(nameof(device));
            m_dataType = dataType;

            m_loss = trainer.LossFunction();
            m_metric = trainer.EvaluationFunction();
        }

        public Function Network;

        public Dictionary<string, List<double>> Fit(IMinibatchSource trainMinibatchSource = null, int batchSize = 32, int epochs = 1,
            IMinibatchSource validationMinibatchSource = null)

        {
            // Setup fitter.
            var fitter = new Fitter(m_trainer, m_device);

            // TODO: Refactor to callback style reporting for each metric, instead of returning a dictionary.
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

                fitter.FitNextStep(minibatch, batchSize);

                if (isSweepEnd)
                {
                    var currentLoss = fitter.CurrentLoss;
                    lossValidationHistory[m_lossName].Add(currentLoss);

                    var currentMetric = fitter.CurrentMetric;
                    lossValidationHistory[m_metricName].Add(currentMetric);

                    var traceOutput = $"Epoch: {epoch + 1:000} Loss = {currentLoss:F8}, Metric = {currentMetric:F8}";
                    fitter.ResetLossAndMetricAccumulators();

                    ++epoch;

                    if (validationMinibatchSource != null)
                    {
                        (var validationLoss, var validationMetric) = Evaluate(validationMinibatchSource);
                        traceOutput += $" - ValidationLoss = {validationLoss:F8}, ValidationMetric = {validationMetric:F8}";

                        lossValidationHistory[m_validationLossName].Add(validationLoss);
                        lossValidationHistory[m_validationMetricName].Add(validationMetric);
                    }

                    Trace.WriteLine(traceOutput);
                }
            }

            return lossValidationHistory;
        }

        public (double loss, double metric) Evaluate(IMinibatchSource minibatchSource)
        {
            // create loss and metric evaluators.
            using (var lossEvaluator = new MetricEvaluator(m_loss, m_device))
            using (var metricEvaluator = new MetricEvaluator(m_metric, m_device))
            {
                bool isSweepEnd = false;

                while (!isSweepEnd)
                {
                    // TODO: Add Support for other evaluation batch sizes.
                    const int evaluationBatchSize = 1;
                    var nextMinibatch = minibatchSource.GetNextMinibatch(evaluationBatchSize, m_device);
                    var minibatch = nextMinibatch.minibatch;
                    isSweepEnd = nextMinibatch.isSweepEnd;

                    lossEvaluator.EvalauteNextStep(minibatch, evaluationBatchSize);
                    metricEvaluator.EvalauteNextStep(minibatch, evaluationBatchSize);
                }

                var finalLoss = lossEvaluator.CurrentMetric;
                var finalMetric = metricEvaluator.CurrentMetric;

                return ((float)finalLoss, (float)finalMetric);
            }
        }

        public IList<IList<float>> Predict(IMinibatchSource minibatchSource)
        {
            var predictions = new List<IList<float>>();
            var predictor = new Predictor(Network, m_device);

            bool isSweepEnd = false;

            while (!isSweepEnd)
            {
                // TODO: Add Support for other prediction batch sizes.
                const int predictionBatchSize = 1;
                var nextMinibatch = minibatchSource.GetNextMinibatch(predictionBatchSize, m_device);
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
