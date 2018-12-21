using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CntkCatalyst.LayerFunctions
{
    public enum BatchNorm
    {
        /// <summary>
        /// This is used for convolutions.
        /// spatially-pooled batch-normalization, 
        /// where normalization values will be tied across all pixel positions pr. channel.
        /// </summary>
        Spatial,

        /// <summary>
        /// This is used for dense, fully connected layers.
        /// Regular batch-normalization where all elements of the input tensor 
        /// will be normalized independently.
        /// </summary>
        Regular
    }
}
