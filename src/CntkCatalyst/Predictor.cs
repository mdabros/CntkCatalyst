using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class Predictor
    {
        IDictionary<StreamInformation, Variable> m_streamInfoToVariable;

        double m_metricSum = 0f;
        int m_totalSampleCount = 0;
        bool disposed = false;

        IDictionary<Variable, Value> m_input;
        IDictionary<Variable, Value> m_output;

        public Predictor(Function network, IDictionary<StreamInformation, Variable> streamInfoToVariable,
            DeviceDescriptor device)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            m_streamInfoToVariable = streamInfoToVariable ?? throw new ArgumentNullException(nameof(streamInfoToVariable));
            Device = device ?? throw new ArgumentNullException(nameof(device));

            m_input = new Dictionary<Variable, Value>();
            m_output = new Dictionary<Variable, Value>();
        }

        public Function Network { get; }
        public DeviceDescriptor Device { get; }

        public double CurrentMetric => m_metricSum / m_totalSampleCount;

        public IList<IList<float>> PredictNextStep(IDictionary<StreamInformation, MinibatchData> minibatch)
        {
            var batchSize = AssignDataFromMinibatch(minibatch);

            var outputVar = Network.Output;
            m_output.Add(outputVar, null);

            Network.Evaluate(m_input, m_output, Device);
            var outputVal = m_output[outputVar];

            var batchPrediction = outputVal.GetDenseData<float>(outputVar);

            // Ensure cleanup, call erase.
            m_output.Clear();
            m_input.Clear();

            return batchPrediction;
        }

        public void ResetLossAccumulation()
        {
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
    }
}
