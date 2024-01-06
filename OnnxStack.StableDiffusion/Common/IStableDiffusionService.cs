using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IStableDiffusionService
    {
        IReadOnlyCollection<StableDiffusionModelSet> ModelSets { get; }

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> LoadModelAsync(IOnnxModelSetConfig model);


        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(IOnnxModel model);

        /// <summary>
        /// Determines whether the specified model is loaded
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if the specified model is loaded; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(IOnnxModel model);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="DenseTensor<float>"/></returns>
        Task<DenseTensor<float>> GenerateAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="SixLabors.ImageSharp.Image<Rgba32>"/></returns>
        Task<Image<Rgba32>> GenerateAsImageAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="byte[]"/></returns>
        Task<byte[]> GenerateAsBytesAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates the StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The Scheduler options.</param>
        /// <param name="progressCallback">The callback used to provide progess of the current InferenceSteps.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>The diffusion result as <see cref="System.IO.Stream"/></returns>
        Task<Stream> GenerateAsStreamAsync(ModelOptions model, PromptOptions prompt, SchedulerOptions options, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<BatchResult> GenerateBatchAsync(ModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<Image<Rgba32>> GenerateBatchAsImageAsync(ModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<byte[]> GenerateBatchAsBytesAsync(ModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates a batch of StableDiffusion image using the prompt and options provided.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="batchOptions">The batch options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        IAsyncEnumerable<Stream> GenerateBatchAsStreamAsync(ModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions, BatchOptions batchOptions, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default);
    }
}