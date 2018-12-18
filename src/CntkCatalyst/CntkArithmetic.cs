using CNTK;

namespace CntkCatalyst
{
    public static class CntkArithmetic
    {
        public static Function ElementTimes(this Function leftOperand, Function rightOperand)
        {
            return CNTKLib.ElementTimes(leftOperand, rightOperand);
        }

        public static Function ElementTimes(this Variable leftOperand, Variable rightOperand)
        {
            return CNTKLib.ElementTimes(leftOperand, rightOperand);
        }

        public static Function Minus(this Function leftOperand, Function rightOperand)
        {
            return CNTKLib.Minus(leftOperand, rightOperand);
        }

        public static Function Minus(this Variable leftOperand, Variable rightOperand)
        {
            return CNTKLib.Minus(leftOperand, rightOperand);
        }

        public static Function Plus(this Function leftOperand, Function rightOperand)
        {
            return CNTKLib.Plus(leftOperand, rightOperand);
        }

        public static Function Plus(this Variable leftOperand, Variable rightOperand)
        {
            return CNTKLib.Plus(leftOperand, rightOperand);
        }
    }
}
