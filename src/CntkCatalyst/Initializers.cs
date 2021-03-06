using CNTK;

namespace CntkCatalyst
{
    /// <summary>
    /// Initializer factory for CNTK
    /// </summary>
    public static class Initializers
    {
        public static CNTKDictionary Zero()
        {
            return CNTKLib.ConstantInitializer(0);
        }

        public static CNTKDictionary One()
        {
            return CNTKLib.ConstantInitializer(1);
        }

        public static CNTKDictionary None()
        {
            return null;
        }

        public static CNTKDictionary Bilinear(int kernelWidth, int kernelHeight)
        {
            return CNTKLib.BilinearInitializer((uint)kernelWidth, (uint)kernelHeight);
        }

        public static CNTKDictionary Uniform(int seed)
        {
            return CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale, (uint)seed);
        }

        public static CNTKDictionary Uniform(int seed, double scale)
        {
            return CNTKLib.UniformInitializer(scale, (uint)seed);
        }

        public static CNTKDictionary Normal(int seed)
        {
            return CNTKLib.NormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Normal(int seed, double scale)
        {
            return CNTKLib.NormalInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Normal(int seed, double scale, 
            int outputRank)
        {
            return CNTKLib.NormalInitializer(scale, outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Normal(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.NormalInitializer(scale, outputRank, filterRank, (uint)seed);
        }

        public static CNTKDictionary TruncatedNormal(int seed)
        {
            return CNTKLib.TruncatedNormalInitializer(CNTKLib.DefaultParamInitScale,
                (uint)seed);
        }

        public static CNTKDictionary TruncatedNormal(int seed, double scale)
        {
            return CNTKLib.TruncatedNormalInitializer(scale, (uint)seed);
        }

        public static CNTKDictionary Xavier(int seed)
        {
            return CNTKLib.XavierInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Xavier(int seed, double scale)
        {
            return CNTKLib.XavierInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Xavier(int seed, double scale,
            int outputRank)
        {
            return CNTKLib.XavierInitializer(scale,
                outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary Xavier(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.XavierInitializer(scale,
                outputRank,
                filterRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotNormal(int seed)
        {
            return CNTKLib.GlorotNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotNormal(int seed, double scale)
        {
            return CNTKLib.GlorotNormalInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotNormal(int seed, double scale, 
            int outputRank)
        {
            return CNTKLib.GlorotNormalInitializer(scale,
                outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotNormal(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.GlorotNormalInitializer(scale,
                outputRank,
                filterRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotUniform(int seed)
        {
            return CNTKLib.GlorotUniformInitializer(CNTKLib.DefaultParamInitScale, 
                CNTKLib.SentinelValueForInferParamInitRank, 
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotUniform(int seed, double scale)
        {
            return CNTKLib.GlorotUniformInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotUniform(int seed, double scale,
            int outputRank)
        {
            return CNTKLib.GlorotUniformInitializer(scale,
                outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary GlorotUniform(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.GlorotUniformInitializer(scale,
                outputRank,
                filterRank,
                (uint)seed);
        }

        public static CNTKDictionary HeNormal(int seed)
        {
            return CNTKLib.HeNormalInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeNormal(int seed, double scale)
        {
            return CNTKLib.HeNormalInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeNormal(int seed, double scale,
            int outputRank)
        {
            return CNTKLib.HeNormalInitializer(scale,
                outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeNormal(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.HeNormalInitializer(scale,
                outputRank,
                filterRank,
                (uint)seed);
        }

        public static CNTKDictionary HeUniform(int seed)
        {
            return CNTKLib.HeUniformInitializer(CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeUniform(int seed, double scale)
        {
            return CNTKLib.HeUniformInitializer(scale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeUniform(int seed, double scale,
            int outputRank)
        {
            return CNTKLib.HeUniformInitializer(scale,
                outputRank,
                CNTKLib.SentinelValueForInferParamInitRank,
                (uint)seed);
        }

        public static CNTKDictionary HeUniform(int seed, double scale,
            int outputRank, int filterRank)
        {
            return CNTKLib.HeUniformInitializer(scale,
                outputRank,
                filterRank,
                (uint)seed);
        }
    }
}
