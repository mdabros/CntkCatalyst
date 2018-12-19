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

        public UniformNoiseMinibatchSource(IDictionary<string, Variable> nameToVariable, float min, float max, int seed)
        {
            m_nameToVariable = nameToVariable ?? throw new ArgumentNullException(nameof(nameToVariable));

            m_min = min;
            m_max = max;
            m_random = new Random(seed);
        }

        public (IDictionary<Variable, Value> minibatch, bool isSweepEnd) GetNextMinibatch(int minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            var minibatch = new Dictionary<Variable, Value>();
            foreach (var kvp in m_nameToVariable)
            {
                var variable = kvp.Value;
                var sampleShape = variable.Shape;
                var totalElementCount = minibatchSizeInSamples * sampleShape.Dimensions.Aggregate((d1, d2) => d1 * d2);

                // Consider reusing sample arrays, to avoid load on GC.
                var samples = Enumerable.Range(0, totalElementCount)
                    .Select(v => SampleRandomUniform())
                    .ToArray();
                
                var value = Value.CreateBatch<float>(sampleShape, samples, device);
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
