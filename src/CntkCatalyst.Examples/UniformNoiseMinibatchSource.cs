using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace CntkCatalyst.Examples
{
    public class UniformNoiseMinibatchSource : IMinibatchSource
    {
        readonly float m_min;
        readonly float m_max;
        readonly Random m_random;

        readonly IDictionary<string, Variable> m_nameToVariable;
        readonly IDictionary<string, float[]> m_nameToData;

        public UniformNoiseMinibatchSource(IDictionary<string, Variable> nameToVariable, float min, float max, int seed)
        {
            m_nameToVariable = nameToVariable ?? throw new ArgumentNullException(nameof(nameToVariable));

            m_min = min;
            m_max = max;
            m_random = new Random(seed);

            // Initialize data dictionary.
            m_nameToData = m_nameToVariable.ToDictionary(v => v.Key, v => new float[0]);
        }

        public (IDictionary<Variable, Value> minibatch, bool isSweepEnd) GetNextMinibatch(int minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            var minibatch = new Dictionary<Variable, Value>();
            foreach (var kvp in m_nameToVariable)
            {
                var name = kvp.Key;
                var variable = kvp.Value;
                var sampleShape = variable.Shape;
                var totalElementCount = minibatchSizeInSamples * sampleShape.TotalSize;

                var data = m_nameToData[name];
                if(data.Length != totalElementCount)
                {
                    Array.Resize(ref data, totalElementCount);
                    m_nameToData[name] = data;
                }

                for (int i = 0; i < data.Length; i++)
                {
                    data[i] = SampleRandomUniform();
                }
                
                var value = Value.CreateBatch<float>(sampleShape, data, device);
                minibatch.Add(variable, value);
            }

            // since the samples are randomly generated,
            // there is no limit to the number of samples,
            // hence, the sweep never ends.
            var isSweepEnd = false;

            return (minibatch, isSweepEnd);
        }

        float SampleRandomUniform()
        {
            return (float)m_random.NextDouble() * (m_max - m_min) + m_min;
        }
    }
}
