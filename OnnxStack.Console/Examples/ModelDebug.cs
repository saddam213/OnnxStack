using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class ModelDebug : IExampleRunner
    {
        private readonly string _outputDirectory;

        public ModelDebug()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ModelDebug));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 1;

        public string Name => "Model Debug";

        public string Description => "Model Debug";

        public async Task RunAsync()
        {
            // Execution provider
            //var provider = Providers.AMDGPU(0);
            var provider = Providers.DirectML(0);

            // Create Pipeline
            //var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\Debug\\stable-diffusion-1.5_io32_amdgpu");
            var pipeline = FluxPipeline.CreatePipeline(provider, "M:\\Models\\Flux\\Flux_kontext-f16-onnx", ModelType.Instruct);
            // var pipeline = LatentConsistencyPipeline.CreatePipeline("M:\\Models\\LCM_Dreamshaper_v7-onnx");
            // var pipeline = LatentConsistencyXLPipeline.CreatePipeline("D:\\Repositories\\Latent-Consistency-xl-Olive-Onnx");
            // var pipeline = StableDiffusionXLPipeline.CreatePipeline("C:\\Repositories\\sdxl-turbo-ryzenai-0912\\sdxl-turbo-ryzenai-0912", ModelType.Turbo, executionProvider: Core.Config.ExecutionProvider.RyzenAI);
            // var pipeline = CogVideoPipeline.CreatePipeline("M:\\BaseModels\\CogVideoX-2b\\_Test",);
            // var pipeline = HunyuanPipeline.CreatePipeline("M:\\Models\\HunyuanDiT-onnx");
            // var pipeline = StableDiffusionXLPipeline.CreatePipeline("M:\\Models\\stable-diffusion-instruct-pix2pix-onnx", ModelType.Turbo);
            // var inputImage = await OnnxImage.FromFileAsync(@"C:\\Users\\Deven\\Downloads\flux_amuse.png");


            // Create ControlNet
            // var controlNet = ControlNetModel.Create("M:\\Models\\ControlNet-onnx\\SD\\Depth\\model.onnx");

            // Create FeatureExtractor
            // var featureExtractor = FeatureExtractorPipeline.CreatePipeline("M:\\Models\\FeatureExtractor-onnx\\SD\\Depth\\model.onnx", sampleSize: 512, normalizeOutput: true);

            // Create Depth Image
            // var controlImage = await featureExtractor.RunAsync(inputImage);

            // Save Depth Image
            // await controlImage.SaveAsync(Path.Combine(_outputDirectory, $"Depth.png"));

            var inputImage = await OnnxImage.FromFileAsync("C:\\Users\\Administrator\\Pictures\\Sample4.png");
            inputImage.Resize(256, 256);

            // Prompt
            var generateOptions = new GenerateOptions
            {
            
                Prompt = "Add sunglasses and a hat to the woman",
                Diffuser = DiffuserType.ImageToImage,
                InputImage = inputImage,
                OptimizationType = OptimizationType.None,
                IsLowMemoryComputeEnabled = true,
                IsLowMemoryDecoderEnabled = true,
                IsLowMemoryEncoderEnabled = true,
                IsLowMemoryTextEncoderEnabled = true
            };

            // Scheduler
            generateOptions.SchedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                //Seed = 1445933838,
                Width = 768, Height = 768,
                InferenceSteps = 28,
                GuidanceScale = 0,
                GuidanceScale2 = 2.5f,
            };


            // Run pipeline
            var result = await pipeline.GenerateAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Save Image File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"{pipeline.GetType().Name}-{generateOptions.SchedulerOptions.Seed}.png"));

            //Unload
            await pipeline.UnloadAsync();

        }
    }
}
