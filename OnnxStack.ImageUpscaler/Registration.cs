using Microsoft.Extensions.DependencyInjection;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.ImageUpscaler.Config;

namespace OnnxStack.ImageUpscaler
{
    public static class Registration
    {
        /// <summary>
        /// Register OnnxStack ImageUpscaler services
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        public static void AddOnnxStackImageUpscaler(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddOnnxStack();
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(TryLoadAppSettings());
        }


        /// <summary>
        /// Register OnnxStack ImageUpscaler services, AddOnnxStack() must be called before
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        /// <param name="configuration">The configuration.</param>
        public static void AddOnnxStackImageUpscaler(this IServiceCollection serviceCollection, ImageUpscalerConfig configuration)
        {
            serviceCollection.RegisterServices();
            serviceCollection.AddSingleton(configuration);
        }


        /// <summary>
        /// Registers the services.
        /// </summary>
        /// <param name="serviceCollection">The service collection.</param>
        private static void RegisterServices(this IServiceCollection serviceCollection)
        {
            serviceCollection.AddSingleton<IVideoService, VideoService>();
        }


        /// <summary>
        /// Try load ImageUpscalerConfig from application settings.
        /// </summary>
        /// <returns></returns>
        private static ImageUpscalerConfig TryLoadAppSettings()
        {
            try
            {
                return ConfigManager.LoadConfiguration<ImageUpscalerConfig>();
            }
            catch
            {
                return new ImageUpscalerConfig();
            }
        }
    }
}