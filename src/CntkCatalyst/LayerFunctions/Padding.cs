using System;

namespace CntkCatalyst.LayerFunctions
{
    public enum Padding
    {
        /// <summary>
        /// Add no padding. This is sometimes also refered to as Valid.
        /// This indicates that the kernel is applied only to "valid" regions of the input, 
        /// that is regions where the kernel is completely within the input. 
        /// This corresponds to no padding added, 
        /// and will therefore always shrink the output dimensions compared to the input dimensions,
        /// with about half the kernel size.
        /// </summary>
        None,

        /// <summary>
        /// Add zero-padding to keep output dimensions the same as input dimensions.
        /// Note that this only applies when the kernel stride is 1. 
        /// Strides larger than one does not affect the amount of padding 
        /// since this is based on the kernel size only. 
        /// So using this padding mode with a stride larger than one, 
        /// will not result in the output dimensions matching the input dimensions.
        /// </summary>
        Zeros
    }

    public static class PaddingExtensions
    {
        public static bool ToBoolean(this Padding padding)
        {
            switch (padding)
            {
                case Padding.None:
                    return false;
                case Padding.Zeros:
                    return true;
                default:
                    throw new ArgumentException($"Unknown padding type: {padding}");
            }
        }
    }
}
