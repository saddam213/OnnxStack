using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxStack.Core.Config;

namespace OnnxStack.TextGeneration.Common
{
    public class TextGenerationModel : IDisposable //: OnnxModelSession
    {
        private Model _model;
        private Tokenizer _tokenizer;
        private readonly TextGenerationModelConfig _configuration;
        public TextGenerationModel(TextGenerationModelConfig configuration)
        {
            _configuration = configuration;
        }

        public Model Model => _model;
        public Tokenizer Tokenizer => _tokenizer;


        /// <summary>
        /// Loads the model session.
        /// </summary>
        public async Task LoadAsync()
        {
            if (_model is not null)
                return; // Already Loaded

            await Task.Run(() =>
            {
                _model = new Model(_configuration.OnnxModelPath);
                _tokenizer = new Tokenizer(_model);
            });
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _tokenizer?.Dispose();
            _model?.Dispose();
            _model = null;
            _tokenizer = null;
        }


        public static TextGenerationModel Create(TextGenerationModelConfig configuration)
        {
            return new TextGenerationModel(configuration);
        }

        public static TextGenerationModel Create(string modelPath, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new TextGenerationModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelPath
            };
            return new TextGenerationModel(configuration);
        }


    }
}
