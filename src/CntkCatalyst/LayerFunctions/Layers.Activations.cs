using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function ReLU(this Function input)
        {
            return CNTKLib.ReLU(input);
        }

        public static Function LeakyReLU(this Function input, double alpha)
        {
            return CNTKLib.LeakyReLU(input, alpha);
        }

        public static Function Sigmoid(this Function input)
        {
            return CNTKLib.Sigmoid(input);
        }

        public static Function Tanh(this Function input)
        {
            return CNTKLib.Tanh(input);
        }

        public static Function Softmax(this Function input)
        {
            return CNTKLib.Softmax(input);
        }
    }
}
