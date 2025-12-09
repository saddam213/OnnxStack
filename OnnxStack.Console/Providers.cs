using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Model;

namespace OnnxStack.Console
{
    public static class Providers
    {
        public static OnnxExecutionProvider CPU()
        {
            return new OnnxExecutionProvider("CPU", configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }

        public static OnnxExecutionProvider DirectML(int deviceId)
        {
            return new OnnxExecutionProvider("DirectML", configuration =>
            {
                var sessionOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL
                };

                sessionOptions.AppendExecutionProvider_DML(deviceId);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            });
        }

    }
}
