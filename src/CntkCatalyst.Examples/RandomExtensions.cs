using System;

namespace CntkCatalyst.Examples
{
    public delegate float SampleRandomUniform(Random random, float min, float max);

    public static class RandomExtensions
    {
        public static float SampleRandomUniformF32(Random random, float min, float max)
        {
            return (float)(random.NextDouble() * (max - min) + min);
        }

        public static float SampleRandomUniformInt32(Random random, float min, float max)
        {
            var maxInclusive = max + 1;
            return random.Next((int)min, (int)maxInclusive);
        }
    }
}
