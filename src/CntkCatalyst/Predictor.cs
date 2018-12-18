using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class Predictor
    {
        IDictionary<Variable, Value> m_output;

        public Predictor(Function network, DeviceDescriptor device)
        {
            Network = network ?? throw new ArgumentNullException(nameof(network));
            Device = device ?? throw new ArgumentNullException(nameof(device));

            m_output = new Dictionary<Variable, Value>();
        }

        public Function Network { get; }
        public DeviceDescriptor Device { get; }

        /// <summary>
        /// Currently only supports single output from network (can be multi-target, but only assigned to one output variable).
        /// TODO: Add support for multiple outputs.
        /// </summary>
        /// <param name="minibatch"></param>
        /// <returns></returns>
        public IList<IList<float>> PredictNextStep(IDictionary<Variable, Value> minibatch)
        {
            var outputVar = Network.Output;
            m_output.Add(outputVar, null);

            Network.Evaluate(minibatch, m_output, Device);
            var outputVal = m_output[outputVar];

            var batchPrediction = outputVal.GetDenseData<float>(outputVar);

            // Ensure cleanup.
            m_output.Clear();

            return batchPrediction;
        }
    }
}
