using System.Linq;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    public static partial class Layers
    {
        internal static Function ConvTranspose(this Function input,
                    int[] filterShape,
                    int filterCount,
                    int[] filterStride,
                    Padding padding, // TODO: Consider if padding should be decided pr. dimension.
                    int[] outputShape,
                    CNTKDictionary weightInitializer,
                    CNTKDictionary biasInitializer,
                    DeviceDescriptor device,
                    DataType dataType)
        {
            var weights = new Parameter(NDShape.CreateNDShape(filterShape), dataType,
                weightInitializer, device);

            var strideShape = NDShape.CreateNDShape(filterStride);

            // Currently, only sharing=true is supported by CNTK. So these are hardcoded.
            // sharing dimensions follows stride dimensions. 1D, 2D, 3D, etc.
            var sharing = CntkUtilities.CreateFilledBoolVector(filterStride.Length, true);

            // Padding dimensions follows stride dimensions. 1D, 2D, 3D, etc.
            var usePadding = padding.ToBoolean();
            var autoPadding = CntkUtilities.CreateFilledBoolVector(filterStride.Length, usePadding);
            // Auto-padding must be false for the channel dimension.
            // This seems to be different compared to regular convolution which does not require this. 
            autoPadding.Add(false);

            var outputNDShape = NDShape.CreateNDShape(outputShape);

            // TODO: Consider if we want to surface the additional options for Convolution:
            // - dilation
            // - maxTempMemSizeInSamples

            // Default for dilation seems to be a shape of size (1) with value 1
            var dilation = NDShape.CreateNDShape(new[] { 1 });

            // Following are defaults extrapolated from CNTK code
            var reductionRank = 1u;
            var maxTempMemSizeInSamples = 0u;

            var result = CNTKLib.ConvolutionTranspose(
                weights, input, filterStride, sharing, autoPadding,
                outputNDShape, dilation, reductionRank, 
                maxTempMemSizeInSamples, string.Empty);

            if (biasInitializer != null)
            {
                // Bias dimensions should be defined for filter dimensions.
                // For instance for 2D case: (1, 1, filterChannels).
                var biasShape = filterStride.Select(s => 1).ToList();
                biasShape.Add(filterCount);

                var bias = new Parameter(NDShape.CreateNDShape(biasShape.ToArray()),
                    dataType, biasInitializer, device);

                result = CNTKLib.Plus(result, bias);
            }

            return result;
        }
    }
}
