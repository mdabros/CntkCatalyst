using System;
using System.Linq;
using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;

        public CntkMinibatchSource(MinibatchSource minibatchSource, string featuresName, string labelsName)
        {
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
            FeaturesName = featuresName;
            TargetsName = labelsName;
        }

        public string FeaturesName { get; }
        public string TargetsName { get; }


        public (IDictionary<StreamInformation, MinibatchData> minibatch, bool isSweepEnd) GetNextMinibatch(
            int minibatchSizeInSamples, DeviceDescriptor device)
        {
            var minibatch = m_minibatchSource.GetNextMinibatch((uint)minibatchSizeInSamples, device);
            var isSweepEnd = minibatch.Values.Any(a => a.sweepEnd);

            return (minibatch, isSweepEnd);
        }

        //public (IDictionary<Variable, MinibatchData> minibatch, bool isSweepEnd) NextMinibatch(uint minibatchSizeInSamples, DeviceDescriptor device)
        //{
        //    var minibatch = new Dictionary<Variable, MinibatchData>();

        //    var minibatchData = m_minibatchSource.GetNextMinibatch(minibatchSizeInSamples, device);
        //    var isSweepEnd = minibatchData.Values.Any(a => a.sweepEnd);

        //    var obserationsStreamInfo = m_minibatchSource.StreamInfo(FeaturesName);
        //    var targetsStreamInfo = m_minibatchSource.StreamInfo(TargetsName);

        //    var observations = minibatchData[obserationsStreamInfo];
        //    var targets = minibatchData[targetsStreamInfo];

        //    minibatch.Add(m_inputVariable, observationsData);
        //    minibatch.Add(m_targetVariable, targetsData);
        //}

        public StreamInformation StreamInfo(string streamName)
        {
            return m_minibatchSource.StreamInfo(streamName);
        }
    }
}
