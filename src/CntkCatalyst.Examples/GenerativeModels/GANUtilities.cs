using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace CntkCatalyst.Examples.GenerativeModels
{
    public static class GANUtilities
    {
        public static Value CreateNoiseSamplesValue(Random random, int sampleCount, NDShape sampleShape,
            DeviceDescriptor device, float min = -1.0f, float max = 1.0f)
        {
            var samples = CreateNoiseSamples(random, sampleCount, sampleShape.Dimensions, min, max);
            var value = Value.CreateBatch<float>(sampleShape, samples, device);

            return value;
        }

        public static float[] CreateNoiseSamples(Random random, int sampleCount, IList<int> sampleShape,
            float min = -1.0f, float max = 1.0f)
        {
            var totalElementCount = sampleCount * sampleShape.Aggregate((d1, d2) => d1 * d2);

            var samples = Enumerable.Range(0, totalElementCount)
                .Select(v => (float)random.NextDouble() * (max - min) + min)
                .ToArray();

            return samples;
        }
    }
}
