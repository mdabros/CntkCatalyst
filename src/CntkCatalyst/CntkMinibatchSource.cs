using System;
using System.Linq;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;
        readonly IDictionary<string, Variable> m_nameToVariable;

        public CntkMinibatchSource(MinibatchSource minibatchSource, IDictionary<string, Variable> nameToVariable)
        {
            m_nameToVariable = nameToVariable ?? throw new ArgumentNullException(nameof(nameToVariable));
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
        }

        public (IDictionary<Variable, Value> minibatch, bool isSweepEnd) GetNextMinibatch(int minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            var streamInfoToMinibatchData = m_minibatchSource.GetNextMinibatch((uint)minibatchSizeInSamples, device);
            var isSweepEnd = streamInfoToMinibatchData.Values.Any(a => a.sweepEnd);

            var minibatch = AssignDataFromMinibatch(streamInfoToMinibatchData);

            return (minibatch, isSweepEnd);
        }

        Dictionary<Variable, Value> AssignDataFromMinibatch(IDictionary<StreamInformation, MinibatchData> streamInfoToMinibatchData)
        {
            var minibatch = new Dictionary<Variable, Value>();

            foreach (var kvp in m_nameToVariable)
            {
                var name = kvp.Key;
                var variable = kvp.Value;
                var streamInfo = m_minibatchSource.StreamInfo(name);

                var minibatchData = streamInfoToMinibatchData[streamInfo];
                var value = streamInfoToMinibatchData[streamInfo].data;
                minibatch.Add(variable, value);
            }

            return minibatch;
        }
    }
}
