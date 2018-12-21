using System;
using System.Linq;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Conv2D(this Function input,
            ValueTuple<int, int> filterShape,
            int filterCount,
            ValueTuple<int, int> filterStride,
            Padding padding,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType)
        {
            var filterSizes = new int[]
            {
                filterShape.Item1,
                filterShape.Item2,
                NDShape.InferredDimension, // Infer number of channels in input.
                filterCount
            };

            var filterStrides = new int[]
            {
                filterStride.Item1,
                filterStride.Item2,
            };

            return Conv(input, filterSizes, filterCount, filterStrides, 
                padding, weightInitializer, biasInitializer, 
                device, dataType);
        }
    }
}
