using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class BatchEvalutator : IDisposable
    {
        IDictionary<StreamInformation, Variable> m_streamInfoToVariable;

        double m_metricSum = 0f;
        int m_totalSampleCount = 0;
        bool disposed = false;

        public BatchEvalutator(Evaluator evaluator, IDictionary<StreamInformation, Variable> streamInfoToVariable,
            DeviceDescriptor device)
        {
            Evalautor = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
            m_streamInfoToVariable = streamInfoToVariable ?? throw new ArgumentNullException(nameof(streamInfoToVariable));
            Device = device ?? throw new ArgumentNullException(nameof(device));
        }

        public Evaluator Evalautor { get; }
        public DeviceDescriptor Device { get; }

        public double CurrentMetric => m_metricSum / m_totalSampleCount;

        public void EvalauteNextStep(IDictionary<StreamInformation, MinibatchData> minibatch)
        {
            using (var inputMap = new UnorderedMapVariableMinibatchData())
            {
                var batchSize = AssignDataFromMinibatch(minibatch, inputMap);

                m_metricSum += Evalautor.TestMinibatch(inputMap, Device) * batchSize;
                m_totalSampleCount += batchSize;

                inputMap.Clear();
            }
        }

        public void ResetLossAccumulation()
        {
            m_metricSum = 0;
            m_totalSampleCount = 0;
        }

        int AssignDataFromMinibatch(IDictionary<StreamInformation, MinibatchData> minibatch, UnorderedMapVariableMinibatchData inputMap)
        {
            var batchSize = 0;
            foreach (var kvp in m_streamInfoToVariable)
            {
                var streamInfo = kvp.Key;
                var variable = kvp.Value;

                var minibatchData = minibatch[streamInfo];
                batchSize = (int)minibatchData.numberOfSamples;

                var data = minibatch[streamInfo];
                inputMap.Add(variable, data);
            }

            return batchSize;
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
                Evalautor.Dispose();
            }

            disposed = true;
        }
    }
}
