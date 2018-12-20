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

            var weights = new Parameter(NDShape.CreateNDShape(filterSizes), dataType,
                   weightInitializer, device);

            // Currently, only sharing=true is supported by CNTK. So these are hardcoded.
            // sharing dimensions follows stride dimensions. 1D, 2D, 3D, etc.
            var sharing = CntkUtilities.CreateFilledBoolVector(filterStrides.Length, true);

            // Padding dimensions follows stride dimensions. 1D, 2D, 3D, etc.
            var usePadding = padding.ToBoolean();
            var autoPadding = CntkUtilities.CreateFilledBoolVector(filterStrides.Length, usePadding);
            autoPadding.Add(false); // auto-padding must be false for the channel dimension.

            var result = CNTKLib.ConvolutionTranspose(weights, input, filterStrides, sharing, autoPadding, NDShape.CreateNDShape(outputSizes));

            if (biasInitializer != null)
            {
                // Bias dimensions should be defined for filter dimensions.
                // For instance for 2D case: (1, 1, filterChannels).
                var biasShape = filterStrides.Select(s => 1).ToList();
                biasShape.Add(filterCount);

                var bias = new Parameter(NDShape.CreateNDShape(biasShape.ToArray()),
                    dataType, biasInitializer, device);

                result = CNTKLib.Plus(result, bias);
            }

            return result;
        }
    }
}
