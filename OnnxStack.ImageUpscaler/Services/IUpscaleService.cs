using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;
using System.Threading.Tasks;

namespace OnnxStack.ImageUpscaler.Services
{
    public interface IUpscaleService
    {

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> LoadModelAsync(UpscaleModelSet model);

        /// <summary>
        /// Unloads the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns></returns>
        Task<bool> UnloadModelAsync(UpscaleModelSet model);

        /// <summary>
        /// Determines whether [is model loaded] [the specified model options].
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <returns>
        ///   <c>true</c> if [is model loaded] [the specified model options]; otherwise, <c>false</c>.
        /// </returns>
        bool IsModelLoaded(UpscaleModelSet modelOptions);

        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> GenerateAsync(UpscaleModelSet modelOptions, InputImage inputImage);

        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<Image<Rgba32>> GenerateAsImageAsync(UpscaleModelSet modelOptions, InputImage inputImage);


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<byte[]> GenerateAsByteAsync(UpscaleModelSet modelOptions, InputImage inputImage);


        /// <summary>
        /// Generates the upscaled image.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <returns></returns>
        Task<Stream> GenerateAsStreamAsync(UpscaleModelSet modelOptions, InputImage inputImage);


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        Task<DenseTensor<float>> GenerateAsync(UpscaleModelSet modelOptions, VideoInput videoInput);


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        Task<byte[]> GenerateAsByteAsync(UpscaleModelSet modelOptions, VideoInput videoInput);


        /// <summary>
        /// Generates the upscaled video.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="videoInput">The video input.</param>
        /// <returns></returns>
        Task<Stream> GenerateAsStreamAsync(UpscaleModelSet modelOptions, VideoInput videoInput);
    }
}
