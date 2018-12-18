using System.Collections.Generic;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test
{
    [TestClass]
    public class MemoryMinibatchSourceTest
    {
        float[] m_observationsData = new float[]
        {
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4,
            5, 5, 5, 5 ,5,
            6, 6, 6, 6, 6,
            7, 7, 7, 7, 7,
            8, 8, 8, 8, 8,
            9, 9, 9, 9, 9,
        };

        float[] m_targetData = new float[]
        {
            1, 1, 1, 2, 2, 2, 3, 3, 3
        };

        [TestMethod]
        public void MemoryMinibatchSource_GetNextMinibatch()
        {
            var observationsShape = new int[] { 5 };
            var observations = new MemoryMinibatchData(m_observationsData, observationsShape, 9);
            var targetsShape = new int[] { 1 };
            var targets = new MemoryMinibatchData(m_targetData, targetsShape, 9);

            // setup name to data map.
            var nameToData = new Dictionary<string, MemoryMinibatchData>
            {
                { "observations", observations },
                { "targets", targets }
            };

            var nameToVariable = new Dictionary<string, Variable>
            {
                { "observations", Variable.InputVariable(observationsShape, DataType.Float) },
                { "targets", Variable.InputVariable(targetsShape, DataType.Float) }
            };

            var sut = new MemoryMinibatchSource(nameToVariable, nameToData, 5, false);
            var device = DeviceDescriptor.CPUDevice;

            for (int i = 0; i < 30; i++)
            {
                var (minibatch, isSweepEnd) = sut.GetNextMinibatch(3, device);
                var obs = minibatch[nameToVariable["observations"]].GetDenseData<float>(nameToVariable["observations"]);
                var tar = minibatch[nameToVariable["targets"]].GetDenseData<float>(nameToVariable["targets"]);
            }
        }
    }
}
