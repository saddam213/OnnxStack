using OnnxStack.Core.Config;
using OnnxStack.TextGeneration.Models;
using OnnxStack.TextGeneration.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class TextGenerationExample : IExampleRunner
    {
        public TextGenerationExample()
        {
        }

        public int Index => 40;

        public string Name => "Text Generation Demo";

        public string Description => "Text Generation Example";

        public async Task RunAsync()
        {
            var pipeline = TextGenerationPipeline.CreatePipeline("D:\\Repositories\\phi2_onnx", executionProvider: ExecutionProvider.Cuda);

            await pipeline.LoadAsync();
 
            while (true)
            {
                OutputHelpers.WriteConsole("Enter Prompt: ", ConsoleColor.Gray);
                var promptOptions = new PromptOptionsModel(OutputHelpers.ReadConsole(ConsoleColor.Cyan));
                var searchOptions = new SearchOptionsModel();
                await foreach (var token in pipeline.RunAsync(promptOptions, searchOptions))
                {
                    OutputHelpers.WriteConsole(token.Content, ConsoleColor.Yellow, false);
                }
            }
        }
    }
}
