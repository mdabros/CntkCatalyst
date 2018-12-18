using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class MetricEvaluator : IDisposable
    {
        double m_metricSum = 0f;
        int m_totalSampleCount = 0;
        bool disposed = false;

        Evaluator m_evaluator;

        public MetricEvaluator(Function metric, DeviceDescriptor device)
        {            
            Metric = metric ?? throw new ArgumentNullException(nameof(metric));
            Device = device ?? throw new ArgumentNullException(nameof(device));

            m_evaluator = CNTKLib.CreateEvaluator(metric);
        }

        public Function Metric { get; }
        public DeviceDescriptor Device { get; }

        public double CurrentMetric => m_metricSum / m_totalSampleCount;

        public void EvalauteNextStep(IDictionary<Variable, Value> minibatch, int batchSize)
        {
            using (var inputMap = new UnorderedMapVariableMinibatchData())
            {
                AssignDataFromMinibatch(minibatch, inputMap, batchSize);

                m_metricSum += m_evaluator.TestMinibatch(inputMap, Device) * batchSize;
                m_totalSampleCount += batchSize;

                inputMap.Clear();
            }
        }

        public void ResetLossAccumulation()
        {
            m_metricSum = 0;
            m_totalSampleCount = 0;
        }

        void AssignDataFromMinibatch(IDictionary<Variable, Value> minibatch, 
            UnorderedMapVariableMinibatchData inputMap,
            int batchSize)
        {
            foreach (var kvp in minibatch)
            {
                var variable = kvp.Key;
                var value = kvp.Value;

                inputMap.Add(variable, new MinibatchData(value, (uint)batchSize));
            }
        }

        // Public implementation of Dispose pattern callable by consumers.
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // Protected implementation of Dispose pattern.
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                m_evaluator.Dispose();
            }

            disposed = true;
        }
    }
}
