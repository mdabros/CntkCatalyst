using System;
using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        static class RecurrentOperators
        {
            public const string LSTM = "lstm";
            public const string GRU = "gru";
            public const string RNN_Tanh = "rnnTanh";
            public const string RNN_ReLU = "rnnReLU";
        }
        
        public static Function LSTMStack(this Function input, int units, int layerCount, 
            CNTKDictionary weightInitializer, 
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name = "")
        {                
            return CreateOptimizedRNNStack(input, RecurrentOperators.LSTM, units, layerCount, 
                weightInitializer, bidirectional, device, dataType, name);
        }

        public static Function GRUStack(this Function input, int units, int layerCount,
            CNTKDictionary weightInitializer,
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name = "")
        {
            return CreateOptimizedRNNStack(input, RecurrentOperators.GRU, units, layerCount,
                weightInitializer, bidirectional, device, dataType, name);
        }

        public static Function SimpleReluRNNStack(this Function input, int units, int layerCount,
            CNTKDictionary weightInitializer,
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name = "")
        {
            return CreateOptimizedRNNStack(input, RecurrentOperators.RNN_ReLU, units, layerCount,
                weightInitializer, bidirectional, device, dataType, name);
        }

        public static Function SimpleTanhRNNStack(this Function input, int units, int layerCount,
            CNTKDictionary weightInitializer,
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name = "")
        {
            return CreateOptimizedRNNStack(input, RecurrentOperators.RNN_Tanh, units, layerCount,
                weightInitializer, bidirectional, device, dataType, name);
        }

        static Function CreateOptimizedRNNStack(Function input, string recurrentOperator, int units,
            int layerCount,
            CNTKDictionary weightInitializer,
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name)
        {
            if (device.Type != DeviceKind.GPU)
            {
                throw new NotSupportedException($"OptimizedRNNStack only supports GPU. Device was: {device.Type}");
            }

            // TODO: Investigate initialization:
            // All weights are contained in a single matrix that should have hiddenDims rows 
            // and as many columns as needed to hold all parameters. Since this can be cumbersome to determine, 
            // you can have the dimension inferred automatically. 
            // To make sure that random initialization uses the correct fan-in, specify initOutputRank=-1:

            var weighthape = new int[] { units, NDShape.InferredDimension };
            var weights = new Parameter(weighthape, dataType, weightInitializer, device);

            return CNTKLib.OptimizedRNNStack(input, weights, (uint)units, (uint)layerCount,
                bidirectional, recurrentOperator, name);
        }
    }
}
