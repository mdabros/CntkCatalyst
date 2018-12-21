using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Flatten(this Function input)
        {
            return CNTKLib.Flatten(input);
        }

        public static Function Reshape(this Function input, NDShape shape)
        {
            return CNTKLib.Reshape(input, shape);
        }
    }
}
