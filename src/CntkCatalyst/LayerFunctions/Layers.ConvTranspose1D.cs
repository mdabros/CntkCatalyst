using System;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function ConvTranspose1D(this Function input,
            int filterShape,
            int filterCount,
            int filterStride,
            Padding padding,
            int outputShape,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType)
        {
            // Notice that the order of the filter arguments
            // are different compared to conventional convolution.
            var filterSizes = new int[]
            {
                filterShape,
                filterCount,
                NDShape.InferredDimension // Infer number of channels in input.
            };

            var filterStrides = new int[]
            {
                filterStride
            };

            var outputSizes = new int[]
            {
                outputShape,
                filterCount,
            };

            return ConvTranspose(input, filterSizes, filterCount, filterStrides,
                padding, outputSizes, weightInitializer, biasInitializer,
                device, dataType);
        }
    }
}
