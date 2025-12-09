using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OnnxStack.Core;
using OnnxStack.Device.Services;
using System.Reflection;

namespace OnnxStack.Console
{
    internal class Program
    {
        static async Task Main(string[] _)
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Logging.ClearProviders();
            builder.Logging.AddConsole();
            builder.Logging.SetMinimumLevel(LogLevel.Trace);

            // Add OnnxStack
            builder.Services.AddOnnxStack();
            //builder.Services.AddOnnxStackConfig<StableDiffusionConfig>();

            // Add AppService
            builder.Services.AddHostedService<AppService>();
            builder.Services.AddSingleton<IHardwareSettings>(new HardwareSettings());
            builder.Services.AddSingleton<IHardwareService, HardwareService>();

            // Add Runners
            var exampleRunners = Assembly.GetExecutingAssembly()
                .GetTypes()
                .Where(type => typeof(IExampleRunner).IsAssignableFrom(type) && !type.IsInterface)
                .ToList();
            builder.Services.AddSingleton(exampleRunners.AsEnumerable());
            foreach (var exampleRunner in exampleRunners)
            {
                builder.Services.AddSingleton(exampleRunner);
            }

            // Start
            await builder.Build().RunAsync();
        }

    }

    public class HardwareSettings : IHardwareSettings
    {
        public int ProcessId { get; set; } = Environment.ProcessId;
        public bool UseLegacyDeviceDetection { get; set; }
    }
}