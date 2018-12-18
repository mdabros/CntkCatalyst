using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace CntkCatalyst
{
    public class MemoryMinibatchSource : IMinibatchSource
    {
        readonly Random m_random;
        readonly int[] m_currentSweepIndeces;

        int m_currentBatchStartIndex = -1;
        int[] m_batchIndeces = Array.Empty<int>();

        readonly float[] m_minibatch;
        bool m_randomize;

        IDictionary<string, MemoryMinibatchData> m_data;

        readonly Dictionary<string, StreamInformation> m_streamInfos;

        public MemoryMinibatchSource(IDictionary<string, MemoryMinibatchData> data,
            int seed,
            bool randomize)
        {
            m_data = data ?? throw new ArgumentNullException(nameof(data));

            if(data.Count <= 0)
            {
                throw new ArgumentException("No items added to data dictionary");
            }

            var sampleCount = data.First().Value.SampleCount;
            foreach (var item in data)
            {
                if(sampleCount != item.Value.SampleCount)
                {
                    throw new ArgumentException($"Sample count not consistent: " 
                        + string.Join(",", data.Select(v => $"{v.Key}: samples: {v.Value.SampleCount}")));
                }
            }

            m_currentSweepIndeces = Enumerable.Range(0, sampleCount).ToArray();
            m_random = new Random(seed);
            m_randomize = randomize;
            m_minibatch = Array.Empty<float>();

            m_streamInfos = m_data.ToDictionary(v => v.Key, v => new StreamInformation { m_name = v.Key });
        }

        public (IDictionary<StreamInformation, MinibatchData> minibatch, bool isSweepEnd) GetNextMinibatch(
            int minibatchSizeInSamples, DeviceDescriptor device)
        {
            CheckIfNewSweepAndShuffle();

            UpdateBatchIndeces(minibatchSizeInSamples, m_currentBatchStartIndex);

            var minibatchItems = NextBatch(device);

            m_currentBatchStartIndex += minibatchSizeInSamples;

            // Start over if sweep end
            var isSweepEnd = m_currentBatchStartIndex >= m_currentSweepIndeces.Length;
            if (isSweepEnd)
            {
                m_currentBatchStartIndex = -1;
            }

            var minibatch = minibatchItems.ToDictionary(v => v.Key, 
                v => new MinibatchData(v.Value, (uint)minibatchSizeInSamples, isSweepEnd));

            return (minibatch, isSweepEnd);
        }

        public StreamInformation StreamInfo(string streamName)
        {
            return m_streamInfos[streamName];
        }

        Dictionary<StreamInformation, Value> NextBatch(DeviceDescriptor device)
        {
            var minibatch = new Dictionary<StreamInformation, Value>();

            foreach (var item in m_data)
            {
                var sampleShape = item.Value.SampleShape;
                var samplesData = CopyMinibatchSamples(item.Value);
                var name = item.Key;

                var value = Value.CreateBatch<float>(sampleShape, samplesData, device, true);
                minibatch.Add(StreamInfo(name), value);
            }

            return minibatch;
        }

        float[] CopyMinibatchSamples(MemoryMinibatchData data)
        {
            var sampleSize = data.SampleShape.Aggregate((v1, v2) => v1 * v2);
            var batchSize = m_batchIndeces.Length;
            var minibatchItem = new float[batchSize * sampleSize];

            for (int i = 0; i < m_batchIndeces.Length; i++)
            {
                var batchIndex = m_batchIndeces[i];

                var startIndex = batchIndex * sampleSize;
                Array.Copy(data.Data, startIndex, minibatchItem,
                    i * sampleSize, sampleSize);
            }

            return minibatchItem;
        }

        void CheckIfNewSweepAndShuffle()
        {
            if (m_currentBatchStartIndex < 0)
            {
                if (m_randomize)
                {
                    // Start new sweep by shuffling indexes in place.
                    Shuffle(m_currentSweepIndeces, m_random);
                }
                m_currentBatchStartIndex = 0;
            }
        }

        void UpdateBatchIndeces(int minibatchSizeInSamples, int batchStartIndex)
        {
            if (m_batchIndeces.Length != minibatchSizeInSamples)
            {
                m_batchIndeces = new int[minibatchSizeInSamples];
            }

            for (int i = 0; i < minibatchSizeInSamples; i++)
            {
                // Repeat the start so we can fulfill the requested batch size
                var sweepIndex = (i + batchStartIndex) % m_currentSweepIndeces.Length;
                m_batchIndeces[i] = m_currentSweepIndeces[sweepIndex];
            }
        }

        static void Shuffle<TIndex>(TIndex[] array, Random random)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = random.Next(n);
                --n;
                TIndex temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
