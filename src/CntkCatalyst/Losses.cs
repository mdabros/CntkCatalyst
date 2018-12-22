using CNTK;

namespace CntkCatalyst
{
    public static class Losses
    {
        public static Function MeanSquaredError(Variable predictions, Variable targets)
        {
            var errors = CNTKLib.Minus(targets, predictions);
            var squaredErrors = CNTKLib.Square(errors);
            return ReduceMeanAll(squaredErrors);
        }

        public static Function MeanAbsoluteError(Variable predictions, Variable targets)
        {
            var errors = CNTKLib.Minus(targets, predictions);
            var absoluteErrors = CNTKLib.Abs(errors);
            return ReduceMeanAll(absoluteErrors);
        }

        public static Function CategoricalCrossEntropy(Variable predictions, Variable targets)
        {
            var erros = CNTKLib.CrossEntropyWithSoftmax(predictions, targets);
            return ReduceMeanAll(erros);
        }

        public static Function BinaryCrossEntropy(Variable predictions, Variable targets)
        {
            var errors = CNTKLib.BinaryCrossEntropy(predictions, targets);
            return ReduceMeanAll(errors);
        }

        static Function ReduceMeanAll(Function errors)
        {
            var allAxes = Axis.AllStaticAxes();
            return CNTKLib.ReduceMean(errors, allAxes);
        }
    }
}
