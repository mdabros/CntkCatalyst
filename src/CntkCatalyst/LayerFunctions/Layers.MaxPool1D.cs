using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function MaxPool1D(this Function input,
            int poolShape,
            int strideShape,
            Padding padding) 
        {
            var usePadding = padding.ToBoolean();

            var poolSizes = new int[] { poolShape };
            var strideSizes = new int[] { strideShape };
            var paddingSizes = new bool[] { usePadding };

            return CNTKLib.Pooling(input,
                PoolingType.Max,
                poolSizes,
                strideSizes,
                paddingSizes);
        }
    }
}
