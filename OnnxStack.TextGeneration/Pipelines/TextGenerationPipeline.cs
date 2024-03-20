using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.TextGeneration.Common;
using OnnxStack.TextGeneration.Models;
using System.Runtime.CompilerServices;

namespace OnnxStack.TextGeneration.Pipelines
{

    public class TextGenerationPipeline
    {
        private readonly string _name;
        private readonly ILogger _logger;
        private readonly TextGenerationModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="TextGenerationPipeline"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="model">The text generation model.</param>
        /// <param name="logger">The logger.</param>
        public TextGenerationPipeline(string name, TextGenerationModel model, ILogger logger = default)
        {
            _name = name;
            _logger = logger;
            _model = model;
        }


        /// <summary>
        /// Gets the name.
        /// </summary>
        /// <value>
        public string Name => _name;


        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <returns></returns>
        public Task LoadAsync()
        {
            return _model.LoadAsync();
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.Yield();
            _model?.Dispose();
        }


        /// <summary>
        /// Runs the text generation pipeline
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public IAsyncEnumerable<TokenModel> RunAsync(PromptOptionsModel promptOptions, SearchOptionsModel searchOptions, CancellationToken cancellationToken = default)
        {
            return RunInternalAsync(promptOptions, searchOptions, cancellationToken);
        }


        /// <summary>
        /// Runs the text generation pipeline
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async IAsyncEnumerable<TokenModel> RunInternalAsync(PromptOptionsModel promptOptions, SearchOptionsModel searchOptions, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var timestamp = _logger?.LogBegin("Run text generation pipeline stream...");
            var sequences = await EncodePrompt(promptOptions, cancellationToken);

            using (var generatorParams = new GeneratorParams(_model.Model))
            {
                ApplySearchOptions(generatorParams, searchOptions);
                generatorParams.SetInputSequences(sequences);

                using (var tokenizerStream = _model.Tokenizer.CreateStream())
                using (var generator = new Generator(_model.Model, generatorParams))
                {
                    while (!generator.IsDone())
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        yield return await Task.Run(() =>
                        {
                            generator.ComputeLogits();
                            generator.GenerateNextTokenTop();

                            var tokenId = generator.GetSequence(0)[^1];
                            return new TokenModel(tokenId, tokenizerStream.Decode(tokenId));
                        }, cancellationToken);
                    }
                }
            }
            _logger?.LogEnd("Text generation pipeline stream complete.", timestamp);
        }


        /// <summary>
        /// Encodes the prompt.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<Sequences> EncodePrompt(PromptOptionsModel promptOptions, CancellationToken cancellationToken = default)
        {
            return await Task.Run(() => _model.Tokenizer.Encode(promptOptions.Prompt), cancellationToken);
        }


        /// <summary>
        /// Applies the search options to the GeneratorParams instance.
        /// </summary>
        /// <param name="generatorParams">The generator parameters.</param>
        /// <param name="searchOptions">The search options.</param>
        private static void ApplySearchOptions(GeneratorParams generatorParams, SearchOptionsModel searchOptions)
        {
            generatorParams.SetSearchOption("top_p", searchOptions.TopP);
            generatorParams.SetSearchOption("top_k", searchOptions.TopK);
            generatorParams.SetSearchOption("temperature", searchOptions.Temperature);
            generatorParams.SetSearchOption("repetition_penalty", searchOptions.RepetitionPenalty);
            generatorParams.SetSearchOption("past_present_share_buffer", searchOptions.PastPresentShareBuffer);
            generatorParams.SetSearchOption("num_return_sequences", searchOptions.NumReturnSequences);
            generatorParams.SetSearchOption("no_repeat_ngram_size", searchOptions.NoRepeatNgramSize);
            generatorParams.SetSearchOption("min_length", searchOptions.MinLength);
            generatorParams.SetSearchOption("max_length", searchOptions.MaxLength);
            generatorParams.SetSearchOption("length_penalty", searchOptions.LengthPenalty);
            generatorParams.SetSearchOption("early_stopping", searchOptions.EarlyStopping);
            generatorParams.SetSearchOption("do_sample", searchOptions.DoSample);
            generatorParams.SetSearchOption("diversity_penalty", searchOptions.DiversityPenalty);
        }


        /// <summary>
        /// Creates the pipeline from a TextGenerationModelSet.
        /// </summary>
        /// <param name="modelSet">The model set.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static TextGenerationPipeline CreatePipeline(TextGenerationModelSet modelSet, ILogger logger = default)
        {
            var textGenerationModel = new TextGenerationModel(modelSet.TextGenerationConfig.ApplyDefaults(modelSet));
            return new TextGenerationPipeline(modelSet.Name, textGenerationModel, logger);
        }


        /// <summary>
        /// Creates the pipeline from the specified file.
        /// </summary>
        /// <param name="modelFile">The model file.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns></returns>
        public static TextGenerationPipeline CreatePipeline(string modelFile, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML, ILogger logger = default)
        {
            var name = Path.GetFileNameWithoutExtension(modelFile);
            var configuration = new TextGenerationModelSet
            {
                Name = name,
                IsEnabled = true,
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                TextGenerationConfig = new TextGenerationModelConfig
                {
                    OnnxModelPath = modelFile
                }
            };
            return CreatePipeline(configuration, logger);
        }
    }
}
