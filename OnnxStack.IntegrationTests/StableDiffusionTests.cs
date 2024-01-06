using FluentAssertions;
using FluentAssertions.Execution;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;
using Xunit.Abstractions;

namespace OnnxStack.IntegrationTests;

/// <summary>
/// These tests could be run via a self-hosted runner in Github Actions to run the tests on your own GPU as part of the CI/CD pipeline.
/// Maybe something like https://www.youtube.com/watch?v=rVq-SCNyxVc
/// </summary>
[Collection("IntegrationTests")]
public class StableDiffusionTests
{
    private readonly IStableDiffusionService _stableDiffusion;
    private readonly ILogger<StableDiffusionTests> _logger;
    private const string StableDiffusionModel = "StableDiffusion 1.5";
    private const string LatentConsistencyModel = "LCM-Dreamshaper-V7";

    public StableDiffusionTests(ITestOutputHelper testOutputHelper)
    {
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole());               //necessary for showing logs when running in docker
        services.AddLogging(builder => builder.AddXunit(testOutputHelper)); //necessary for showing logs when running in IDE
        services.AddOnnxStackStableDiffusion();
        var provider = services.BuildServiceProvider();
        _stableDiffusion = provider.GetRequiredService<IStableDiffusionService>();
        _logger = provider.GetRequiredService<ILogger<StableDiffusionTests>>();
    }
    
    [Theory]
    [InlineData(StableDiffusionModel)]
    [InlineData(LatentConsistencyModel)]
    public async Task GivenAStableDiffusionModel_WhenLoadModel_ThenModelIsLoaded(string modelName)
    {
        //arrange
        var model = _stableDiffusion.ModelSets.Single(m => m.Name == modelName);
        
        //act
        _logger.LogInformation("Attempting to load model {0}", model.Name);
        var isModelLoaded = await _stableDiffusion.LoadModelAsync(model);

        //assert
        isModelLoaded.Should().BeTrue();
    }

    [Theory]
    [InlineData(StableDiffusionModel, SchedulerType.EulerAncestral, 10, 7.0f)]
    [InlineData(LatentConsistencyModel, SchedulerType.LCM, 4, 1.0f)]
    public async Task GivenTextToImage_WhenInference_ThenImageGenerated(string modelName, SchedulerType schedulerType,
        int inferenceSteps, float guidanceScale)

    {
        //arrange
        var model = _stableDiffusion.ModelSets.Single(m => m.Name == modelName);
        _logger.LogInformation("Attempting to load model: {0}", model.Name);
        await _stableDiffusion.LoadModelAsync(model);

        var prompt = new PromptOptions
        {
            Prompt = "an astronaut riding a horse in space",
            NegativePrompt = "blurry,ugly,cartoon",
            DiffuserType = DiffuserType.TextToImage
        };

        var scheduler = new SchedulerOptions
        {
            Width = 512,
            Height = 512,
            SchedulerType = schedulerType,
            InferenceSteps = inferenceSteps,
            GuidanceScale = guidanceScale,
            Seed = 1
        };

        var steps = 0;

        //act
        var image = await _stableDiffusion.GenerateAsImageAsync(model, prompt, scheduler, (currentStep) =>
        {
            _logger.LogInformation($"Step {currentStep}/{inferenceSteps}");
            steps++;
        });

        var imagesDirectory = Path.Combine(Directory.GetCurrentDirectory(), "images");
        if (!Directory.Exists(imagesDirectory))
        {
            _logger.LogInformation($"Creating directory {imagesDirectory}");
            Directory.CreateDirectory(imagesDirectory);
        }
        else
        {
            _logger.LogInformation($"Directory {imagesDirectory} already exists");
        }

        var fileName =
            $"{imagesDirectory}/{nameof(GivenTextToImage_WhenInference_ThenImageGenerated)}-{DateTime.Now:yyyyMMddHHmmss}.png";
        _logger.LogInformation($"Saving generated image to {fileName}");
        await image.SaveAsPngAsync(fileName);

        //assert
        using (new AssertionScope())
        {
            steps.Should().Be(inferenceSteps);
            image.Should().NotBeNull();
            image.Size.IsEmpty.Should().BeFalse();
            image.Width.Should().Be(512);
            image.Height.Should().Be(512);

            File.Exists(fileName).Should().BeTrue();
        }
    }
}