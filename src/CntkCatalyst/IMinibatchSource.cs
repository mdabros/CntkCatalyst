using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    public interface IMinibatchSource
    {
        (IDictionary<Variable, Value> minibatch, bool isSweepEnd) GetNextMinibatch(
            int minibatchSizeInSamples, DeviceDescriptor device);
    }
}