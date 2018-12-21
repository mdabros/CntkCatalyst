using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function BatchNorm(this Function input,
            BatchNorm batchNorm,
            DeviceDescriptor device,
            DataType dataType,
            double initialScaleValue = 1,
            double initialBiasValue = 0,
            int normalizationTimeConstant = 5000)
        {
            var inferredDimension1D = NDShape.CreateNDShape(new int[] { NDShape.InferredDimension });

            var scaleInitializer = CNTKLib.ConstantInitializer(initialScaleValue);
            var scaleParams = new Parameter(inferredDimension1D,
                dataType, scaleInitializer, device);

            var biasInitializer = CNTKLib.ConstantInitializer(initialBiasValue);
            var biasParams = new Parameter(inferredDimension1D,
                dataType, biasInitializer, device);

            const double zeroInit = 1.0;

            // Batch normalization initial state are constants.
            var runningMean = new Constant(inferredDimension1D, dataType, zeroInit, device);
            var runningInvStd = new Constant(inferredDimension1D, dataType, zeroInit, device);
            var runningCount = new Constant(NDShape.CreateNDShape(new[] { 1 }), dataType, zeroInit, device);

            bool spatial = batchNorm == LayerFunctions.BatchNorm.Spatial;
            
            // Allows to smooth batch estimates with the running statistics. 
            // However, this has not been found useful so far in our experiments (from CNTK team).
            const double blendTimeConstant = 0.0;
            
            // Epsilon is added to the variance to avoid division by 0.
            const double epsilon = 0.00001;
            bool useCudnn = device.Type == DeviceKind.GPU;
            const bool disableRegularization = false;

            // TODO: Consider if we want to surface the additional options for BatchNorm:
            // - blendTimeConstant
            // - epsilon
            // - useCudnn
            // - disableRegularization
            // - name
            return CNTKLib.BatchNormalization(input,
                scaleParams, biasParams,
                runningMean, runningInvStd, runningCount,
                spatial,
                normalizationTimeConstant, blendTimeConstant,
                epsilon,
                useCudnn,
                disableRegularization);
        }
    }
}
