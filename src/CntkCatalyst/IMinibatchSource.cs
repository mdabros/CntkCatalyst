using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public interface IMinibatchSource
    {
        IDictionary<StreamInformation, MinibatchData> GetNextMinibatch(int minibatchSizeInSamples, 
            DeviceDescriptor device);

        StreamInformation StreamInfo(string streamName);

        string FeaturesName { get; }
        string TargetsName { get; }
    }
}