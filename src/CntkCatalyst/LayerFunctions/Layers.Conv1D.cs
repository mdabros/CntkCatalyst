using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Conv1D(this Function input,
            int filterShape,
            int filterCount,
            int filterStride,
            Padding padding,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType)
        {
            var filterSizes = new int[]
            {
                filterShape,
                NDShape.InferredDimension, // Infer number of channels in signal.
                filterCount
            };

            var filterStrides = new int[] { filterStride };

            return Conv(input, filterSizes, filterCount, filterStrides, padding,
                weightInitializer, biasInitializer, device, dataType);
        }
    }
}
