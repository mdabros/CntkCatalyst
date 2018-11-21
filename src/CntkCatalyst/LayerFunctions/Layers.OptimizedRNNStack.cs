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

        public static Function SimpleRNNWithRelu(this Function input, int units, int layerCount,
            CNTKDictionary weightInitializer,
            bool bidirectional,
            DeviceDescriptor device,
            DataType dataType,
            string name = "")
        {
            return CreateOptimizedRNNStack(input, RecurrentOperators.RNN_ReLU, units, layerCount,
                weightInitializer, bidirectional, device, dataType, name);
        }

        public static Function SimpleRNNWithTanh(this Function input, int units, int layerCount,
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

            var inputShape = input.Output.Shape;
            var weights = new Parameter(inputShape, dataType, weightInitializer, device);

            return CNTKLib.OptimizedRNNStack(input, weights, (uint)units, (uint)layerCount,
                bidirectional, recurrentOperator, name);
        }
    }
}
