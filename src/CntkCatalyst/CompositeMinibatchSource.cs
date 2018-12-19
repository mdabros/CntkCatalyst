using System;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class CompositeMinibatchSource : IMinibatchSource
    {
        readonly IMinibatchSource[] m_minibatchSources;

        public CompositeMinibatchSource(params IMinibatchSource[] minibatchSources)
        {
            m_minibatchSources = minibatchSources ?? throw new ArgumentNullException(nameof(minibatchSources));
        }

        public (IDictionary<Variable, Value> minibatch, bool isSweepEnd) GetNextMinibatch(int minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            var minibatch = new Dictionary<Variable, Value>();
            var isSweepEnd = false;

            foreach (var source in m_minibatchSources)
            {
                var (batch, isCurrentSourceSweepEnd) = source.GetNextMinibatch(minibatchSizeInSamples, device);

                foreach (var item in batch)
                {
                    minibatch.Add(item.Key, item.Value);
                }
                
                if(!isSweepEnd)
                {
                    // set isSweepEnd = true, 
                    // if at least one source has reached the end.
                    isSweepEnd = isCurrentSourceSweepEnd;
                }
            }

            return (minibatch, isSweepEnd);
        }
    }
}
