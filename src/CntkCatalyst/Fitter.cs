using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public sealed class Fitter
    {
        IDictionary<StreamInformation, Variable> m_streamInfoToVariable;
        IDictionary<Variable, Value> m_input;

        double m_lossSum = 0f;
        double m_metricSum = 0f;
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

        public void FitNextStep(IDictionary<StreamInformation, MinibatchData> minibatch)
        {
            var batchSize = AssignDataFromMinibatch(minibatch);

            Trainer.TrainMinibatch(m_input, false, Device);

            AccumulateLossAndMetric(batchSize);

            m_input.Clear();
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
            var lossValue = Trainer.PreviousMinibatchLossAverage();
            var metricValue = Trainer.PreviousMinibatchEvaluationAverage();

            // Accumulate loss/metric.
            m_lossSum += lossValue * batchSize;
            m_metricSum += metricValue * batchSize;
            m_totalSampleCount += batchSize;
        }
    }
}
