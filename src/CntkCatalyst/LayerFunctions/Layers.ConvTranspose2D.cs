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
        public static Function ConvTranspose2D(this Function input,
            ValueTuple<int, int> filterShape,
            int filterCount,
            ValueTuple<int, int> strideShape,
            Padding padding,
            ValueTuple<int, int> outputShape,
            CNTKDictionary weightInitializer,
            CNTKDictionary biasInitializer,
            DeviceDescriptor device,
            DataType dataType)
        {
            // Notice that the order of the filter arguments
            // are different compared to conventional convolution.
            var filterSizes = new int[]
            {
                filterShape.Item1,
                filterShape.Item2,
                filterCount,
                NDShape.InferredDimension // Infer number of channels in input.
            };

            var filterStrides = new int[]
            {
                strideShape.Item1,
                strideShape.Item2,
            };

            var outputSizes = new int[]
            {
                outputShape.Item1,
                outputShape.Item2,
                filterCount,
            };

            return ConvTranspose(input, filterSizes, filterCount, filterStrides,
                padding, outputSizes, weightInitializer, biasInitializer,
                device, dataType);
        }
    }
}
