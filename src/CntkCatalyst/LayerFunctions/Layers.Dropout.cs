using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function Dropout(this Function input, double dropoutRate, int seed)
        {
            return CNTKLib.Dropout(input, dropoutRate, (uint)seed);
        }
    }
}
