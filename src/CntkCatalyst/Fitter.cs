using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public sealed class Fitter
    {
        double m_lossSum = 0f;
        double m_metricSum = 0f;
        int m_totalSampleCount = 0;

        public Fitter(Trainer trainer, DeviceDescriptor device)
        {
            Trainer = trainer ?? throw new ArgumentNullException(nameof(trainer));
            Device = device ?? throw new ArgumentNullException(nameof(device));
        }

        public Trainer Trainer { get; }
        public DeviceDescriptor Device { get; }

        public double CurrentLoss => m_lossSum / m_totalSampleCount;
        public double CurrentMetric => m_metricSum / m_totalSampleCount;

        public void FitNextStep(IDictionary<Variable, Value> minibatch, int batchSize)
        {
            Trainer.TrainMinibatch(minibatch, false, Device);
            AccumulateLossAndMetric(batchSize);
        }

        public void ResetLossAccumulation()
        {
            m_lossSum = 0;
            m_metricSum = 0;
            m_totalSampleCount = 0;
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
