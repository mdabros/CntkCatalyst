using System;
using System.Linq;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;

        public CntkMinibatchSource(MinibatchSource minibatchSource)
        {
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
        }

        public (IDictionary<StreamInformation, MinibatchData> minibatch, bool isSweepEnd) GetNextMinibatch(
            int minibatchSizeInSamples, DeviceDescriptor device)
        {
            var minibatch = m_minibatchSource.GetNextMinibatch((uint)minibatchSizeInSamples, device);
            var isSweepEnd = minibatch.Values.Any(a => a.sweepEnd);

            return (minibatch, isSweepEnd);
        }

        public StreamInformation StreamInfo(string streamName)
        {
            return m_minibatchSource.StreamInfo(streamName);
        }
    }
}
