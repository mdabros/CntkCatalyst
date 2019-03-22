using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace CntkCatalyst.Examples
{
    public class RandomOneHotMinibatchSource : IMinibatchSource
    {
        readonly int m_classCount;
        readonly Random m_random;

        readonly IDictionary<string, Variable> m_nameToVariable;
        readonly IDictionary<string, float[]> m_nameToData;

        public RandomOneHotMinibatchSource(IDictionary<string, Variable> nameToVariable, int classCount, int seed)
        {
            m_nameToVariable = nameToVariable ?? throw new ArgumentNullException(nameof(nameToVariable));

            m_classCount = classCount;

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

                for (int i = 0; i < minibatchSizeInSamples; i++)
                {
                    // Using one-hot encoding, so each minibatch sample
                    // will contain zeros and a single 1 indicating the class.
                    var classValueIndex = m_random.Next(0, m_classCount);
                    var miniBatchIndex = i * sampleShape.TotalSize;
                    var classIndex = miniBatchIndex + classValueIndex;
                    data[classIndex] = 1;
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
    }
}
